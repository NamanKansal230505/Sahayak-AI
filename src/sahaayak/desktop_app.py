"""SahaayakAI desktop app — iris cursor + Claude-CLI agent in one window-less GUI.

Layout:
    * Floating circular button (top-right by default, draggable, always on top)
      Click it -> input dialog -> goal -> agent runs -> status panel streams.
    * Background iris tracker (MediaPipe FaceMesh) drives the cursor whenever
      the agent is idle. Auto-pauses while the agent is running so the two
      do not fight for the mouse.
    * Status panel (top-right, docked under the button) streams per-turn
      agent output and exposes a STOP button.
    * F12 kill switch stops both subsystems instantly.
    * PyAutoGUI corner-fail-safe still aborts hard.

The agent backend is the `claude -p` CLI (Claude Code Max subscription).
We shell out once per turn — same duct tape as scripts/agent_claude.py,
just integrated into the GUI loop.

Launch:
    python -m sahaayak.main --app
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

from PyQt6.QtCore import (
    QObject,
    QPoint,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sahaayak.safety.confidence_gate import ConfidenceGate
from sahaayak.safety.kill_switch import KillSwitch
from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

ACTION_RE = re.compile(r'\{[^{}]*"action"\s*:[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)


# ───────────────────────────────────────────────────────── iris tracker ──
class IrisWorker(QThread):
    """MediaPipe FaceMesh cursor driver. Sleeps when paused."""

    error = pyqtSignal(str)
    double_blink = pyqtSignal()  # emits when both eyes blink twice in <600 ms

    def __init__(self, gate: ConfidenceGate, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._gate = gate
        self._stop = False
        self._paused = False
        self._gain = 2.5
        self._click_cooldown_s = 1.2
        self._last_click_ts = 0.0
        # Double-blink state.
        self._both_closed_now = False
        self._both_closure_starts: deque[float] = deque(maxlen=4)
        self._double_blink_window_s = 0.6
        self._double_blink_cooldown_s = 1.5
        self._last_double_blink_ts = 0.0

    def request_stop(self) -> None:
        self._stop = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def run(self) -> None:  # noqa: PLR0915 - cohesive single loop is clearer
        try:
            import cv2  # noqa: PLC0415
            import mediapipe as mp  # noqa: PLC0415
            import pyautogui  # noqa: PLC0415

            from sahaayak.core.kalman_filter import GazeKalmanFilter  # noqa: PLC0415
        except ImportError as exc:
            self.error.emit(f"iris deps missing: {exc}")
            return

        screen_w, screen_h = pyautogui.size()
        cam_w = cam_h = 640.0  # downstream coords are in 640x480 frame space
        cam_cx, cam_cy = cam_w / 2, cam_h / 2

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        kalman = GazeKalmanFilter(process_noise=0.05, measurement_noise=0.5)

        # Wink-detection landmark indices.
        right_upper, right_lower = 159, 145
        left_upper, left_lower = 386, 374
        # Iris centres.
        right_iris, left_iris = 473, 468
        # EAR baseline window.
        ear_history_l: deque[float] = deque(maxlen=30)
        ear_history_r: deque[float] = deque(maxlen=30)

        try:
            while not self._stop:
                if self._paused:
                    self.msleep(80)
                    continue
                ok, frame = cap.read()
                if not ok:
                    self.msleep(20)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if not res.multi_face_landmarks:
                    continue
                lm = res.multi_face_landmarks[0].landmark

                ix = (lm[left_iris].x + lm[right_iris].x) / 2 * cam_w
                iy = (lm[left_iris].y + lm[right_iris].y) / 2 * cam_h
                tx = screen_w / 2 + (ix - cam_cx) * self._gain
                ty = screen_h / 2 + (iy - cam_cy) * self._gain
                sx, sy = kalman.update(tx, ty, ts=time.monotonic())
                # Pad away from corners so we never trip the FAILSAFE accidentally.
                pad = 60
                sx = max(pad, min(screen_w - pad, int(sx)))
                sy = max(pad, min(screen_h - pad, int(sy)))

                if not self._gate.is_blocked():
                    try:
                        pyautogui.moveTo(sx, sy, _pause=False)
                    except Exception:  # noqa: BLE001, S110 - never crash the worker on a stray cursor failure
                        pass

                # Wink-to-click.
                r_gap = abs(lm[right_upper].y - lm[right_lower].y)
                l_gap = abs(lm[left_upper].y - lm[left_lower].y)
                ear_history_l.append(l_gap)
                ear_history_r.append(r_gap)
                now_t = time.monotonic()
                if (
                    len(ear_history_l) >= 10
                    and now_t - self._last_click_ts > self._click_cooldown_s
                    and not self._gate.is_blocked()
                ):
                    base_l = sorted(ear_history_l)[int(len(ear_history_l) * 0.9)]
                    base_r = sorted(ear_history_r)[int(len(ear_history_r) * 0.9)]
                    ratio_l = l_gap / base_l if base_l > 0 else 1.0
                    ratio_r = r_gap / base_r if base_r > 0 else 1.0
                    if ratio_l < 0.6 and ratio_r > 0.85:
                        try:
                            pyautogui.click(button="left")
                        except Exception:  # noqa: BLE001, S110 - best-effort click
                            pass
                        self._last_click_ts = now_t
                    elif ratio_r < 0.6 and ratio_l > 0.85:
                        try:
                            pyautogui.click(button="right")
                        except Exception:  # noqa: BLE001, S110 - best-effort click
                            pass
                        self._last_click_ts = now_t

                    # Double-blink (both eyes close, open, close, open
                    # within `_double_blink_window_s`) → wakes the agent.
                    both_closed = ratio_l < 0.55 and ratio_r < 0.55
                    if both_closed and not self._both_closed_now:
                        self._both_closure_starts.append(now_t)
                        if (
                            len(self._both_closure_starts) >= 2
                            and self._both_closure_starts[-1] - self._both_closure_starts[-2]
                            <= self._double_blink_window_s
                            and now_t - self._last_double_blink_ts > self._double_blink_cooldown_s
                        ):
                            self._last_double_blink_ts = now_t
                            self._both_closure_starts.clear()
                            self.double_blink.emit()
                    self._both_closed_now = both_closed
        finally:
            try:
                cap.release()
                face_mesh.close()
            except Exception:  # noqa: BLE001, S110 - cleanup is best-effort
                pass


# ──────────────────────────────────────────────────────── agent worker ──
class AgentWorker(QThread):
    """Runs one Claude-CLI agent task. Emits per-turn status updates."""

    log = pyqtSignal(str)
    finished_with = pyqtSignal(bool, str)  # success, summary

    def __init__(
        self,
        goal: str,
        max_iterations: int = 20,
        turn_timeout: int = 120,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._goal = goal
        self._max_iter = max_iterations
        self._timeout = turn_timeout
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: PLR0912, PLR0915 - integrated agent loop
        try:
            import pyautogui  # noqa: PLC0415
        except ImportError as exc:
            self.finished_with.emit(False, f"pyautogui missing: {exc}")
            return

        prompt_template = (
            "You are a screen-control agent. Look at the screenshot using the Read "
            "tool, then respond with EXACTLY one JSON action describing the SINGLE "
            "next step. Be terse — every word costs subscription quota.\n\n"
            "GOAL: {goal}\n\n"
            "SCREEN SIZE (original pixels — your coordinates use this space): "
            "{w} x {h}\n\n"
            "SCREENSHOT FILE: {path}\n"
            "(Use the Read tool on this path to view the current screen.)\n\n"
            "PREVIOUS ACTIONS (oldest -> newest):\n{history}\n\n"
            "# Allowed actions (respond with ONE on the LAST line, no markdown fence):\n"
            '{{"action": "click", "x": INT, "y": INT, "button": "left|right" (default left), "double": BOOL}}\n'
            '{{"action": "type", "text": STR}}\n'
            '{{"action": "press", "key": STR}}\n'
            '{{"action": "hotkey", "keys": [STR, STR, ...]}}\n'
            '{{"action": "scroll", "direction": "up|down", "amount": INT}}\n'
            '{{"action": "wait", "seconds": NUM, "reason": STR}} (capped at 5)\n'
            '{{"action": "done", "summary": STR}}\n'
            '{{"action": "failed", "reason": STR}}\n\n'
            "# Safety\n"
            "For ANY irreversible action (sending email/message, deleting data, "
            "payments, closing unsaved work, system / install changes), respond with "
            "{{\"action\": \"failed\", \"reason\": \"needs explicit user confirmation: <what>\"}} "
            "and stop.\n\n"
            "Prefer keyboard shortcuts over clicks (Ctrl+L, Win+R, Win+E, Alt+F4, "
            "Win+D). After 3 failed attempts at the same subgoal, return failed."
        )

        history: list[str] = []
        last_path: Path | None = None

        for i in range(1, self._max_iter + 1):
            if self._stop:
                self.log.emit("[stopped by user]")
                self.finished_with.emit(False, "stopped by user")
                return

            # Capture fresh screenshot.
            if last_path is not None and last_path.exists():
                try:
                    last_path.unlink()
                except OSError:
                    pass
            screen_w, screen_h = pyautogui.size()
            img = pyautogui.screenshot()
            fd, raw_path = tempfile.mkstemp(prefix="sahaayak_screen_", suffix=".png")
            import os as _os  # noqa: PLC0415
            _os.close(fd)
            img.save(raw_path, format="PNG")
            last_path = Path(raw_path)

            history_text = (
                "\n".join(f"  {n + 1}. {h}" for n, h in enumerate(history))
                if history
                else "  (none — first action)"
            )
            prompt = prompt_template.format(
                goal=self._goal,
                w=screen_w,
                h=screen_h,
                path=str(last_path).replace("\\", "/"),
                history=history_text,
            )

            self.log.emit(f"[turn {i:02d}/{self._max_iter}] thinking...")

            t0 = time.perf_counter()
            try:
                cmd = [  # noqa: S607 - claude is the user's CLI, must be on PATH
                    "claude", "-p", prompt,
                    "--output-format", "json",
                    "--allowed-tools", "Read",
                ]
                proc = subprocess.run(  # noqa: S603 - args composed from constants
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self._timeout,
                )
            except subprocess.TimeoutExpired:
                self.log.emit(f"  ⚠ claude turn exceeded {self._timeout}s — aborting.")
                self.finished_with.emit(False, "turn timeout")
                return
            except FileNotFoundError:
                self.log.emit("  ⚠ `claude` CLI not found on PATH.")
                self.finished_with.emit(False, "claude CLI missing")
                return

            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                snippet = (proc.stderr or proc.stdout or "")[:300]
                self.log.emit(f"  ⚠ claude exited {proc.returncode}: {snippet}")
                self.finished_with.emit(False, f"claude error {proc.returncode}")
                return

            try:
                payload = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                self.log.emit(f"  ⚠ could not parse claude JSON: {exc}")
                self.finished_with.emit(False, "claude JSON parse error")
                return

            text = payload.get("result", "") or ""
            cost = payload.get("total_cost_usd", 0.0)
            self.log.emit(f"  ({elapsed:.1f}s, ~${cost:.4f})")

            matches = ACTION_RE.findall(text)
            if not matches:
                self.log.emit("  ⚠ no parseable action — agent ended turn without one.")
                history.append("(no action — try again)")
                continue
            try:
                action = json.loads(matches[-1])
            except json.JSONDecodeError:
                self.log.emit("  ⚠ action JSON malformed.")
                history.append("(malformed action)")
                continue

            name = action.get("action")
            if name == "done":
                summary = action.get("summary", "")
                self.log.emit(f"  ✅ DONE: {summary}")
                self.finished_with.emit(True, summary)
                return
            if name == "failed":
                reason = action.get("reason", "")
                self.log.emit(f"  ❌ FAILED: {reason}")
                self.finished_with.emit(False, reason)
                return

            try:
                result = _execute_action(action)
            except Exception as exc:  # noqa: BLE001
                self.log.emit(f"  ⚠ {exc}")
                history.append(f"{name}: ERROR {exc}")
                continue

            self.log.emit(f"  -> {result}")
            history.append(result)

        self.log.emit(f"[reached max iterations {self._max_iter}]")
        self.finished_with.emit(False, "max iterations")


def _execute_action(action: dict[str, Any]) -> str:
    import pyautogui  # noqa: PLC0415

    name = action.get("action")
    if name == "click":
        x, y = int(action["x"]), int(action["y"])
        button = action.get("button") or "left"
        if action.get("double"):
            pyautogui.doubleClick(x, y, button=button)
            return f"double-click {button} at ({x},{y})"
        pyautogui.click(x, y, button=button)
        return f"click {button} at ({x},{y})"
    if name == "type":
        text = str(action["text"])
        pyautogui.typewrite(text, interval=0.015)
        return f"typed {len(text)} chars"
    if name == "press":
        key = str(action["key"])
        pyautogui.press(key)
        return f"pressed {key}"
    if name == "hotkey":
        keys = [str(k) for k in action["keys"]]
        pyautogui.hotkey(*keys)
        return f"hotkey {'+'.join(keys)}"
    if name == "scroll":
        amount = int(action.get("amount") or 5)
        clicks = -amount if action["direction"] == "down" else amount
        pyautogui.scroll(clicks)
        return f"scroll {action['direction']} x{amount}"
    if name == "wait":
        seconds = max(0.1, min(float(action.get("seconds") or 1), 5.0))
        time.sleep(seconds)
        return f"wait {seconds:.1f}s"
    raise ValueError(f"Unknown action: {name!r}")


# ────────────────────────────────────────────────── floating button ──
class FloatingButton(QWidget):
    """Frameless circular always-on-top button. Drag to move, click to fire."""

    clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(QSize(72, 72))
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._drag_offset: QPoint | None = None
        self._dragged = False
        self._busy = False
        self.setToolTip("SahaayakAI — click to ask the agent")

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self._dragged = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is None:
            return
        new_pos = event.globalPosition().toPoint() - self._drag_offset
        if (new_pos - self.pos()).manhattanLength() > 4:
            self._dragged = True
        self.move(new_pos)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            if not self._dragged:
                self.clicked.emit()
            self._drag_offset = None

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Outer ring
        ring_color = QColor(240, 90, 70) if self._busy else QColor(34, 197, 94)
        pen = QPen(ring_color, 4)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(11, 18, 32, 230)))
        painter.drawEllipse(4, 4, self.width() - 8, self.height() - 8)
        # Glyph
        painter.setPen(QColor("#f9fafb"))
        font = QFont("Segoe UI Emoji", 22, QFont.Weight.Bold)
        painter.setFont(font)
        glyph = "⏳" if self._busy else "🤖"
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, glyph)


# ────────────────────────────────────────────────────── status panel ──
class StatusPanel(QWidget):
    """Frameless docked log panel with a STOP button."""

    stop_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFixedWidth(380)
        self.setMinimumHeight(360)
        self.setStyleSheet(
            "QWidget { background: rgba(11, 18, 32, 235); color: #e5e7eb; "
            "font-family: 'Segoe UI', sans-serif; font-size: 11pt; }"
            "QPushButton#stop { background: #dc2626; color: white; padding: 8px 14px; "
            "border-radius: 6px; font-weight: bold; }"
            "QPushButton#stop:hover { background: #ef4444; }"
            "QTextEdit { background: rgba(0, 0, 0, 0.4); border: 0; padding: 6px; "
            "font-family: Consolas, monospace; font-size: 10pt; }"
            "QLabel#header { font-size: 13pt; font-weight: bold; padding: 6px 8px; }"
            "QLabel#goal { color: #fbbf24; padding: 0 8px 6px 8px; }"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self._header = QLabel("SahaayakAI agent")
        self._header.setObjectName("header")
        v.addWidget(self._header)
        self._goal_label = QLabel("")
        self._goal_label.setObjectName("goal")
        self._goal_label.setWordWrap(True)
        v.addWidget(self._goal_label)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self._log)
        row = QHBoxLayout()
        row.addStretch()
        self._stop_btn = QPushButton("STOP")
        self._stop_btn.setObjectName("stop")
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        row.addWidget(self._stop_btn)
        v.addLayout(row)

    def reset(self, goal: str) -> None:
        self._goal_label.setText(f"Goal: {goal}")
        self._log.clear()
        self._stop_btn.setEnabled(True)
        self._stop_btn.setText("STOP")

    def append(self, text: str) -> None:
        self._log.append(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def mark_done(self, success: bool, summary: str) -> None:
        msg = ("✅ Done — " if success else "❌ Failed — ") + (summary or "")
        self._log.append(msg)
        self._stop_btn.setText("CLOSE")
        self._stop_btn.setEnabled(True)


# ──────────────────────────────────────────────────────── input dialog ──
# ───────────────────────────────────────── speech-to-text (local Whisper) ──
class MicRecorderWorker(QThread):
    """Records mic audio. Stops on request_stop() or auto-stop on silence.

    With ``auto_stop_on_silence=True``, the recorder keeps going until it
    has heard speech (RMS above ``silence_threshold``) and then heard
    ``silence_seconds`` of quiet — perfect for hands-free dictation
    triggered by a double-blink.
    """

    finished_with = pyqtSignal(str)  # wav path
    error = pyqtSignal(str)
    level = pyqtSignal(float)  # rolling RMS for live UI feedback

    SAMPLE_RATE = 16000  # whisper expects 16 kHz mono

    def __init__(
        self,
        max_seconds: int = 30,
        auto_stop_on_silence: bool = False,
        silence_threshold: float = 600.0,
        silence_seconds: float = 1.5,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._max_s = max_seconds
        self._stop = False
        self._auto_stop = auto_stop_on_silence
        self._silence_thresh = float(silence_threshold)
        self._silence_window_s = float(silence_seconds)

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: PLR0912 - cohesive recorder loop
        try:
            import wave  # noqa: PLC0415

            import numpy as np  # noqa: PLC0415
            import sounddevice as sd  # noqa: PLC0415
        except ImportError as exc:
            self.error.emit(f"mic deps missing: {exc}")
            return

        chunks: list = []
        latest_rms: list[float] = [0.0]

        def _cb(indata, _frames, _t, _status) -> None:
            chunks.append(indata.copy())
            arr = indata.astype(np.float32)
            latest_rms[0] = float(np.sqrt(np.mean(arr * arr)))

        speech_seen = False
        last_speech_ts = 0.0
        try:
            stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=_cb,
                blocksize=1024,
            )
            stream.start()
            t0 = time.monotonic()
            while not self._stop and (time.monotonic() - t0) < self._max_s:
                self.msleep(80)
                rms = latest_rms[0]
                self.level.emit(rms)
                if self._auto_stop:
                    now = time.monotonic()
                    if rms > self._silence_thresh:
                        speech_seen = True
                        last_speech_ts = now
                    elif speech_seen and (now - last_speech_ts) > self._silence_window_s:
                        break
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"recording failed: {exc}")
            return

        if not chunks:
            self.error.emit("no audio captured")
            return

        audio = np.concatenate(chunks, axis=0)
        fd, path = tempfile.mkstemp(prefix="sahaayak_mic_", suffix=".wav")
        import os as _os  # noqa: PLC0415
        _os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        self.finished_with.emit(path)


# Module-level cache so the Whisper model loads exactly once across dialog opens.
_WHISPER_MODEL: object | None = None


def _get_whisper_model() -> object:
    """Load and cache the Whisper model on first call (~150 MB download once).

    NOTE: must be called once from the MAIN thread before any worker thread
    uses it. ctranslate2 (the engine inside faster-whisper) binds its
    internal DLL/thread pool to the thread that first instantiates the
    model — calling it cold from a QThread on Windows raises
    "Dynamic DLL initialization failed" on the first transcribe.
    """
    global _WHISPER_MODEL  # noqa: PLW0603
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    from faster_whisper import WhisperModel  # noqa: PLC0415
    # `base` is a good speed/accuracy balance on CPU; int8 keeps it under 200 MB RAM.
    # cpu_threads=1 disables ctranslate2's internal worker pool — slightly
    # slower per call but avoids the cross-thread DLL init crash on Windows.
    logger.info("Loading Whisper 'base' (int8) on main thread — may take ~5s the first time")
    _WHISPER_MODEL = WhisperModel(
        "base", device="cpu", compute_type="int8",
        cpu_threads=1, num_workers=1,
    )
    logger.info("Whisper ready.")
    return _WHISPER_MODEL


def warmup_whisper_on_main_thread() -> None:
    """Force model load + a tiny dummy transcribe on the main thread.

    Call this once at app startup (NOT from a QThread). It bakes the
    DLL/thread state on the main thread so subsequent transcribe calls
    from any QThread are safe on Windows.
    """
    import os as _os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    try:
        model = _get_whisper_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper warmup failed at load: %s", exc)
        return

    # Dummy 1-second silence so the inference path also initialises here.
    fd, path = tempfile.mkstemp(prefix="sahaayak_warmup_", suffix=".wav")
    _os.close(fd)
    sr = 16000
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(sr, dtype=np.int16).tobytes())
    try:
        list(model.transcribe(path)[0])  # type: ignore[union-attr]
        logger.info("Whisper warmup transcribe complete.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper warmup transcribe failed: %s", exc)
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


class WhisperWorker(QThread):
    """Transcribes a WAV via a subprocess running scripts/whisper_transcribe.py.

    Subprocess isolation avoids the Windows ctranslate2 + torch DLL conflict
    we hit when loading faster-whisper inline (see warmup_whisper_on_main_thread
    docstring). Each call costs ~3-5s of subprocess startup but works reliably.
    """

    finished_with = pyqtSignal(str)  # transcribed text
    error = pyqtSignal(str)

    SCRIPT = (
        Path(__file__).resolve().parents[2] / "scripts" / "whisper_transcribe.py"
    )

    def __init__(self, wav_path: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._wav_path = wav_path

    def run(self) -> None:
        if not self.SCRIPT.exists():
            self.error.emit(f"helper missing: {self.SCRIPT}")
            return
        try:
            proc = subprocess.run(  # noqa: S603 - args composed from constants
                [sys.executable, str(self.SCRIPT), self._wav_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.error.emit("transcribe timeout (>120s)")
            return
        finally:
            try:
                Path(self._wav_path).unlink()
            except OSError:
                pass

        if proc.returncode != 0:
            snippet = (proc.stderr or "")[-300:].strip()
            self.error.emit(f"transcribe failed: {snippet}")
            return
        text = (proc.stdout or "").strip()
        self.finished_with.emit(text)


def _play_wake_sound() -> None:
    """Best-effort short beep to confirm hands-free activation."""
    try:
        import winsound  # noqa: PLC0415

        winsound.Beep(880, 90)
        winsound.Beep(1320, 90)
    except Exception:  # noqa: BLE001, S110 - non-critical UX flourish
        pass


class GoalDialog(QDialog):
    """Goal input with optional push-to-talk dictation via local Whisper.

    Args:
        auto_dictate: If True, immediately start recording on open, play a
            wake sound, and auto-submit when silence is detected. Used by
            the double-blink hands-free trigger.
    """

    def __init__(self, parent: QWidget | None = None, auto_dictate: bool = False) -> None:
        super().__init__(parent)
        self._auto_dictate = auto_dictate
        self.setWindowTitle("SahaayakAI — what should I do?")
        self.setFixedWidth(620)
        self.setStyleSheet(
            "QDialog { background: #0b1220; color: #e5e7eb; }"
            "QLabel { color: #e5e7eb; font-size: 12pt; }"
            "QLabel#status { color: #9ca3af; font-size: 10pt; padding: 2px 0; }"
            "QLineEdit { background: #1f2937; color: white; padding: 10px; "
            "font-size: 14pt; border-radius: 6px; }"
            "QPushButton#mic { background: #1f2937; color: white; padding: 10px 14px; "
            "font-size: 16pt; border-radius: 6px; min-width: 56px; }"
            "QPushButton#mic:hover { background: #374151; }"
            "QPushButton#mic[recording=\"true\"] { background: #dc2626; }"
        )
        v = QVBoxLayout(self)
        v.addWidget(QLabel("Tell the agent what to do:"))
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText('e.g. "open notepad" or "go to web.whatsapp.com in chrome"')
        row.addWidget(self.input)
        self._mic_btn = QPushButton("🎤")
        self._mic_btn.setObjectName("mic")
        self._mic_btn.setToolTip("Click to speak (push-to-talk). Click again to stop.")
        self._mic_btn.clicked.connect(self._on_mic)
        row.addWidget(self._mic_btn)
        v.addLayout(row)
        self._status = QLabel("")
        self._status.setObjectName("status")
        v.addWidget(self._status)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)
        self._recorder: MicRecorderWorker | None = None
        self._whisper: WhisperWorker | None = None
        # When True, the dialog clicks OK itself once transcription completes.
        self._auto_submit_on_text = False

        if self._auto_dictate:
            # Schedule wake sound + auto-record after the dialog has shown.
            QTimer.singleShot(0, self._kickoff_auto_dictate)

    def goal(self) -> str:
        return self.input.text().strip()

    # ----- speech-to-text -----
    def _kickoff_auto_dictate(self) -> None:
        """Hands-free path: beep, start recording with VAD silence-stop."""
        _play_wake_sound()
        if self._recorder is not None and self._recorder.isRunning():
            return
        self._recorder = MicRecorderWorker(
            max_seconds=15,
            auto_stop_on_silence=True,
            silence_threshold=600.0,
            silence_seconds=1.5,
        )
        self._recorder.finished_with.connect(self._on_audio_ready)
        self._recorder.error.connect(self._on_mic_error)
        self._auto_submit_on_text = True
        self._set_mic_state("listening")
        self._recorder.start()

    def _on_mic(self) -> None:
        if self._recorder is not None and self._recorder.isRunning():
            # Currently recording → stop and transcribe.
            self._recorder.request_stop()
            self._set_mic_state("transcribing")
            return
        if self._whisper is not None and self._whisper.isRunning():
            return  # busy transcribing
        # Idle → start recording (manual push-to-talk; no auto-submit).
        self._auto_submit_on_text = False
        self._recorder = MicRecorderWorker(max_seconds=30)
        self._recorder.finished_with.connect(self._on_audio_ready)
        self._recorder.error.connect(self._on_mic_error)
        self._set_mic_state("recording")
        self._recorder.start()

    def _on_audio_ready(self, wav_path: str) -> None:
        self._set_mic_state("transcribing")
        self._whisper = WhisperWorker(wav_path)
        self._whisper.finished_with.connect(self._on_text_ready)
        self._whisper.error.connect(self._on_mic_error)
        self._whisper.start()

    def _on_text_ready(self, text: str) -> None:
        if text:
            current = self.input.text()
            if current:
                self.input.setText(current + " " + text)
            else:
                self.input.setText(text)
            self._status.setText(f"✓ {len(text)} chars dictated")
        else:
            self._status.setText("(no speech detected)")
        self._set_mic_state("idle")
        # Hands-free flow: auto-OK after a short pause if we got text.
        if self._auto_submit_on_text and text:
            QTimer.singleShot(700, self.accept)
        self._auto_submit_on_text = False

    def _on_mic_error(self, msg: str) -> None:
        self._status.setText(f"⚠ {msg}")
        self._set_mic_state("idle")

    def _set_mic_state(self, state: str) -> None:
        if state == "recording":
            self._mic_btn.setText("●")
            self._mic_btn.setProperty("recording", "true")
            self._status.setText("🔴 Recording... click 🎤 again to stop (max 30 s)")
        elif state == "listening":
            self._mic_btn.setText("●")
            self._mic_btn.setProperty("recording", "true")
            self._status.setText("👂 Listening... will auto-stop on silence")
        elif state == "transcribing":
            self._mic_btn.setText("⏳")
            self._mic_btn.setProperty("recording", "false")
            self._status.setText("Transcribing locally with Whisper...")
        else:  # idle
            self._mic_btn.setText("🎤")
            self._mic_btn.setProperty("recording", "false")
        # Force stylesheet refresh.
        self._mic_btn.style().unpolish(self._mic_btn)
        self._mic_btn.style().polish(self._mic_btn)


# ─────────────────────────────────────────────────────────── main app ──
class SahaayakApp(QObject):
    def __init__(self, qapp: QApplication) -> None:
        super().__init__()
        self._qapp = qapp
        self._gate = ConfidenceGate()
        self._kill = KillSwitch(self._gate, on_toggle=self._on_kill_toggle)
        self._kill.install()

        self._iris = IrisWorker(self._gate)
        self._iris.error.connect(self._on_iris_error)
        self._iris.double_blink.connect(self._on_double_blink)
        self._iris.start()

        # Floating button — top-right corner.
        screen = qapp.primaryScreen().availableGeometry()
        self._button = FloatingButton()
        self._button.move(screen.right() - 96, screen.top() + 36)
        self._button.clicked.connect(self._on_button_click)
        self._button.show()

        # Status panel — under the button.
        self._panel = StatusPanel()
        self._panel.move(screen.right() - 396, screen.top() + 120)
        self._panel.stop_requested.connect(self._on_stop)
        # Hidden until first agent run.
        self._agent: AgentWorker | None = None
        # Whisper now runs in a subprocess (see WhisperWorker) so no
        # in-process warmup is needed — and the warmup itself trips the
        # Windows torch/ctranslate2 DLL conflict on this machine.

    # ------------------------------------------------------------ events
    def _on_button_click(self) -> None:
        self._open_goal_dialog(auto_dictate=False)

    def _on_double_blink(self) -> None:
        # Hands-free trigger — open the dialog already recording.
        if self._agent is not None and self._agent.isRunning():
            return  # don't interrupt a running agent
        self._open_goal_dialog(auto_dictate=True)

    def _open_goal_dialog(self, auto_dictate: bool) -> None:
        if self._agent is not None and self._agent.isRunning():
            self._panel.show()
            self._panel.raise_()
            return

        # Pause iris while user types and while agent runs.
        self._iris.pause()

        dlg = GoalDialog(auto_dictate=auto_dictate)
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        screen = self._qapp.primaryScreen().availableGeometry()
        dlg.move(
            screen.center().x() - dlg.width() // 2,
            screen.center().y() - 100,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._iris.resume()
            return

        goal = dlg.goal()
        if not goal:
            self._iris.resume()
            return

        self._panel.reset(goal)
        self._panel.append("Starting agent...  (F12 = abort, mouse-into-corner = fail-safe)")
        self._panel.show()
        self._button.set_busy(True)

        self._agent = AgentWorker(goal, max_iterations=20)
        self._agent.log.connect(self._panel.append)
        self._agent.finished_with.connect(self._on_agent_done)
        self._agent.start()

    def _on_stop(self) -> None:
        if self._agent is not None and self._agent.isRunning():
            self._agent.request_stop()
            self._panel.append("[STOP requested — finishing current turn...]")
        else:
            self._panel.hide()

    def _on_agent_done(self, success: bool, summary: str) -> None:
        self._panel.mark_done(success, summary)
        self._button.set_busy(False)
        # Re-enable iris cursor and auto-hide the panel after 6s.
        self._iris.resume()
        QTimer.singleShot(6000, self._panel.hide)

    def _on_iris_error(self, msg: str) -> None:
        logger.warning("Iris worker error: %s", msg)

    def _on_kill_toggle(self, engaged: bool) -> None:
        if engaged:
            self._panel.append("[KILL SWITCH engaged — F12 again to release]")
            if self._agent is not None and self._agent.isRunning():
                self._agent.request_stop()

    # ------------------------------------------------------------ teardown
    def shutdown(self) -> None:
        self._iris.request_stop()
        self._iris.wait(2000)
        if self._agent is not None and self._agent.isRunning():
            self._agent.request_stop()
            self._agent.wait(2000)
        self._kill.uninstall()


def main(argv: list[str] | None = None) -> int:
    qapp = QApplication(argv if argv is not None else sys.argv)
    qapp.setQuitOnLastWindowClosed(False)  # frameless tool windows; don't auto-quit

    sahaayak = SahaayakApp(qapp)
    rc = qapp.exec()
    sahaayak.shutdown()
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
