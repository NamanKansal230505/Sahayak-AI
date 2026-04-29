"""Live cursor-control demo using MediaPipe FaceMesh.

Mirrors the approach of soumyagautam/Eye-Mouse-Tracking but adds the
SahaayakAI safety surfaces: kill switch (F12), confidence gate, time
limit, and PyAutoGUI fail-safe.

Pipeline:
    webcam -> mirror -> MediaPipe FaceMesh (468 landmarks + iris)
           -> iris pixel coords -> screen coords (proportional)
           -> Kalman smoothing -> CursorController.move()
           -> wink-y-distance check -> CursorController.click()

Why this exists alongside `demo_cursor.py`: that one uses our hand-rolled
OpenCV-Haar + OpenVINO-iris pipeline; this one uses MediaPipe end-to-end
because MediaPipe's full face mesh is far more reliable than my
half-implemented BlazeFace decoder. The OpenVINO story for SahaayakAI
production lives in `--run` (iris IR) and Phi-3 (LLM).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import pyautogui  # noqa: E402

from sahaayak.control.cursor_controller import CursorController  # noqa: E402
from sahaayak.core.kalman_filter import GazeKalmanFilter  # noqa: E402
from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: E402
from sahaayak.safety.kill_switch import KillSwitch  # noqa: E402

# Right eye (user's right, image's left after mirror) eyelid y-pair
RIGHT_EYE_UPPER = 159
RIGHT_EYE_LOWER = 145
# Left eye eyelid y-pair
LEFT_EYE_UPPER = 386
LEFT_EYE_LOWER = 374
# Iris centres (refined landmarks)
RIGHT_IRIS_CENTRE = 473
LEFT_IRIS_CENTRE = 468
# Eye-corner landmarks (used for eye-relative gaze normalisation)
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--mode", choices=("eye", "head"), default="eye",
                        help="'eye' = iris-relative-to-eye-corners (no head motion needed); "
                             "'head' = absolute iris pixel position in frame (head-as-joystick).")
    parser.add_argument("--gain", type=float, default=2.5,
                        help="Cursor gain. Sensible defaults: 2.5 for --mode eye, 1.5 for --mode head.")
    parser.add_argument("--no-clicks", action="store_true",
                        help="Disable wink-to-click for safe testing.")
    parser.add_argument("--smooth", type=float, default=0.18,
                        help="EMA factor on the iris signal. Lower = smoother but laggier (default 0.18).")
    parser.add_argument("--deadzone", type=float, default=0.015,
                        help="Iris movement smaller than this (normalised) is ignored (default 0.015).")
    args = parser.parse_args()

    screen_w, screen_h = pyautogui.size()
    print(f"Screen: {screen_w}x{screen_h}")

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # required for iris landmarks 468-477
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    gate = ConfidenceGate(min_confidence=0.4, grace_ms=400)
    cursor = CursorController(gate=gate, smoothing_steps=2,
                              screen_size=(screen_w, screen_h))
    kill = KillSwitch(gate)
    kill.install()
    kalman = GazeKalmanFilter(process_noise=0.08, measurement_noise=0.4)

    print(f"\nMode: {args.mode.upper()}  (gain={args.gain})")
    if args.mode == "eye":
        print("  Look around with your EYES — the cursor follows iris-in-eye direction.")
        print("  Head can stay still. Higher gain = farther cursor reach per eye degree.")
    else:
        print("  Move your HEAD to steer (head-as-joystick mode).")
    print("  - WINK LEFT  eye -> LEFT click")
    print("  - WINK RIGHT eye -> RIGHT click")
    print("  - LOOK UP   (top 200 px) for >0.5s -> SCROLL UP")
    print("  - LOOK DOWN (bottom 200 px) for >0.5s -> SCROLL DOWN")
    print("  - F12 kill switch / mouse-into-corner FAIL-SAFE / Ctrl+C")
    for i in (3, 2, 1):
        print(f"  starting in {i}...")
        time.sleep(1)
    print("GO\n")

    deadline = time.monotonic() + args.seconds
    moves = clicks_l = clicks_r = 0
    scrolls_up = scrolls_dn = 0
    last_log = time.monotonic()
    last_click_ts = 0.0
    # Scroll-zone state: cursor must remain in zone for `arm_ms` before
    # scrolling starts; subsequent scrolls fire every `tick_ms`.
    scroll_zone_h = 200
    scroll_arm_s = 0.5
    scroll_tick_s = 0.3
    scroll_zone_entered_ts: float | None = None
    scroll_last_tick_ts = 0.0
    scroll_zone_kind: str | None = None  # "up" | "down" | None
    # EMA state on the iris signal — smoothes upstream of all other filters.
    ema_x: float | None = None
    ema_y: float | None = None
    # Deadzone anchor: cursor target only updates when iris drifts beyond it.
    anchor_x: float | None = None
    anchor_y: float | None = None
    click_cooldown_s = 1.2
    # MediaPipe landmark .y is normalised [0,1]. The reference repo uses
    # absolute pixel diffs so we follow that convention here.
    wink_close_threshold = 0.012  # closed-eye eyelid gap (normalised)
    wink_open_threshold = 0.018   # other eye must be wider than this

    try:
        while time.monotonic() < deadline:
            ok, frame = cam.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            del frame
            res = face_mesh.process(rgb)
            del rgb
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark

            if args.mode == "eye":
                # Iris position relative to its own eye's corners. Drives cursor
                # by EYE direction, not head pose — much less head movement.
                # Right eye normalised offset (-1..+1 across the eye)
                r_w = lm[RIGHT_EYE_INNER].x - lm[RIGHT_EYE_OUTER].x
                r_cx = (lm[RIGHT_EYE_INNER].x + lm[RIGHT_EYE_OUTER].x) / 2
                r_h = abs(lm[RIGHT_EYE_LOWER].y - lm[RIGHT_EYE_UPPER].y) or 1e-6
                r_cy = (lm[RIGHT_EYE_UPPER].y + lm[RIGHT_EYE_LOWER].y) / 2
                r_off_x = (lm[RIGHT_IRIS_CENTRE].x - r_cx) / max(abs(r_w), 1e-6)
                r_off_y = (lm[RIGHT_IRIS_CENTRE].y - r_cy) / r_h
                # Left eye normalised offset
                l_w = lm[LEFT_EYE_INNER].x - lm[LEFT_EYE_OUTER].x
                l_cx = (lm[LEFT_EYE_INNER].x + lm[LEFT_EYE_OUTER].x) / 2
                l_h = abs(lm[LEFT_EYE_LOWER].y - lm[LEFT_EYE_UPPER].y) or 1e-6
                l_cy = (lm[LEFT_EYE_UPPER].y + lm[LEFT_EYE_LOWER].y) / 2
                l_off_x = (lm[LEFT_IRIS_CENTRE].x - l_cx) / max(abs(l_w), 1e-6)
                l_off_y = (lm[LEFT_IRIS_CENTRE].y - l_cy) / l_h
                # Average both eyes for robustness.
                raw_x = (r_off_x + l_off_x) / 2
                raw_y = (r_off_y + l_off_y) / 2
                # Heavy EMA on the iris signal itself (BEFORE Kalman). With
                # eye-relative tracking the input range is tiny, so a single
                # noisy frame would whip the cursor across the screen.
                if ema_x is None or ema_y is None:
                    ema_x, ema_y = raw_x, raw_y
                ema_x = ema_x * (1 - args.smooth) + raw_x * args.smooth
                ema_y = ema_y * (1 - args.smooth) + raw_y * args.smooth
                # Deadzone: ignore drifts smaller than `args.deadzone`.
                if anchor_x is None or anchor_y is None:
                    anchor_x, anchor_y = ema_x, ema_y
                if abs(ema_x - anchor_x) > args.deadzone:
                    anchor_x = ema_x
                if abs(ema_y - anchor_y) > args.deadzone:
                    anchor_y = ema_y
                off_x, off_y = anchor_x, anchor_y
                tx = screen_w / 2 + off_x * args.gain * screen_w
                ty = screen_h / 2 + off_y * args.gain * screen_h
                ix_norm, iy_norm = off_x, off_y  # for the log line
            else:
                # Head-as-joystick: absolute iris pixel position in frame.
                ix_norm = (lm[LEFT_IRIS_CENTRE].x + lm[RIGHT_IRIS_CENTRE].x) / 2
                iy_norm = (lm[LEFT_IRIS_CENTRE].y + lm[RIGHT_IRIS_CENTRE].y) / 2
                tx = (ix_norm - 0.5) * args.gain * screen_w + screen_w / 2
                ty = (iy_norm - 0.5) * args.gain * screen_h + screen_h / 2
            sx, sy = kalman.update(tx, ty, ts=time.monotonic())
            # Pad away from screen corners so accidental gaze-to-corner does
            # NOT trigger PyAutoGUI's fail-safe. Manual fail-safe still works
            # — user has to physically grab the mouse and shove it past us.
            corner_pad = 60
            sx = max(corner_pad, min(screen_w - corner_pad, int(sx)))
            sy = max(corner_pad, min(screen_h - corner_pad, int(sy)))
            gate.update(0.95, time.monotonic())
            if cursor.move(sx, sy):
                moves += 1

            # --- gaze-zone scroll detection ---
            now_scroll = time.monotonic()
            if sy < scroll_zone_h:
                current_zone = "up"
            elif sy > screen_h - scroll_zone_h:
                current_zone = "down"
            else:
                current_zone = None

            if current_zone is None:
                scroll_zone_entered_ts = None
                scroll_zone_kind = None
            else:
                if current_zone != scroll_zone_kind:
                    scroll_zone_entered_ts = now_scroll
                    scroll_zone_kind = current_zone
                elif (now_scroll - (scroll_zone_entered_ts or now_scroll)) >= scroll_arm_s and (
                    now_scroll - scroll_last_tick_ts
                ) >= scroll_tick_s and not gate.is_blocked():
                    if current_zone == "up":
                        cursor.scroll(3)
                        scrolls_up += 1
                    else:
                        cursor.scroll(-3)
                        scrolls_dn += 1
                    scroll_last_tick_ts = now_scroll

            # Wink detection: |upper.y - lower.y| in normalised coords.
            r_gap = abs(lm[RIGHT_EYE_UPPER].y - lm[RIGHT_EYE_LOWER].y)
            l_gap = abs(lm[LEFT_EYE_UPPER].y - lm[LEFT_EYE_LOWER].y)
            now = time.monotonic()
            if (
                not args.no_clicks
                and now - last_click_ts > click_cooldown_s
                and not gate.is_blocked()
            ):
                # User's LEFT eye = mediapipe LEFT (post-mirror).
                if l_gap < wink_close_threshold and r_gap > wink_open_threshold:
                    cursor.click("left")
                    clicks_l += 1
                    last_click_ts = now
                    print(f"  *** LEFT WINK  -> LEFT CLICK  at ({sx},{sy})  "
                          f"L_gap={l_gap:.4f}  R_gap={r_gap:.4f}")
                elif r_gap < wink_close_threshold and l_gap > wink_open_threshold:
                    cursor.click("right")
                    clicks_r += 1
                    last_click_ts = now
                    print(f"  *** RIGHT WINK -> RIGHT CLICK at ({sx},{sy})  "
                          f"L_gap={l_gap:.4f}  R_gap={r_gap:.4f}")

            if now - last_log >= 1.0:
                print(f"  cursor=({sx:4d},{sy:4d})  iris=({ix_norm:.3f},{iy_norm:.3f})  "
                      f"L_gap={l_gap:.4f}  R_gap={r_gap:.4f}")
                last_log = now
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except pyautogui.FailSafeException:
        print("\nFAIL-SAFE triggered (mouse in screen corner). Stopping.")
    finally:
        cam.release()
        face_mesh.close()
        kill.uninstall()

    print(f"\nMoves: {moves}.  Clicks: left={clicks_l}, right={clicks_r}.  "
          f"Scrolls: up={scrolls_up}, down={scrolls_dn}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
