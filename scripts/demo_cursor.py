"""Live cursor-control demo: gaze drives the OS mouse pointer.

Pipeline:
    webcam frame -> EyeTracker (iris on NPU) -> head-as-joystick mapping
                 -> Kalman smoothing -> ConfidenceGate -> CursorController

Without a calibration profile the iris vector covers only a tiny region of
the camera frame, so we amplify motion relative to the frame centre. This
is "head-as-joystick" mode — good enough to *prove* live control. Run
`python -m sahaayak.main --calibrate` for proper per-pixel accuracy.

Safety:
    * Ctrl+C stops cleanly.
    * Demo auto-stops after `--seconds` (default 25).
    * F12 toggles the SahaayakAI kill switch (forces ConfidenceGate shut).
    * PyAutoGUI's FAILSAFE is on: shove the mouse into the top-left
      corner of any monitor to abort instantly.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import pyautogui  # noqa: E402

from sahaayak.control.cursor_controller import CursorController  # noqa: E402
from sahaayak.core.eye_tracker import EyeTracker  # noqa: E402
from sahaayak.core.kalman_filter import GazeKalmanFilter  # noqa: E402
from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: E402
from sahaayak.safety.kill_switch import KillSwitch  # noqa: E402
from sahaayak.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=25.0,
                        help="Auto-stop after this many seconds.")
    parser.add_argument("--device", default="NPU")
    parser.add_argument("--gain", type=float, default=8.0,
                        help="Head-motion amplification (default 8x).")
    args = parser.parse_args()

    config = load_config()
    screen_w, screen_h = pyautogui.size()
    print(f"Screen: {screen_w}x{screen_h}")

    tracker = EyeTracker(device=args.device, config=config)
    gate = ConfidenceGate(min_confidence=0.4, grace_ms=400)
    cursor = CursorController(gate=gate, smoothing_steps=3,
                              screen_size=(screen_w, screen_h))
    kill = KillSwitch(gate)
    kill.install()
    kalman = GazeKalmanFilter(process_noise=0.05, measurement_noise=0.5)

    print(f"\nLook at the webcam. Move your head a little to steer.")
    print(f"  - WINK LEFT  eye -> LEFT click")
    print(f"  - WINK RIGHT eye -> RIGHT click")
    print(f"  - F12  toggles the kill switch")
    print(f"  - Move mouse to a screen corner for instant FAIL-SAFE")
    print(f"  - Ctrl+C to stop early")
    for i in (3, 2, 1):
        print(f"  starting in {i}...")
        time.sleep(1)
    print("GO\n")

    cam_w, cam_h = 640.0, 480.0
    cam_cx, cam_cy = cam_w / 2, cam_h / 2
    deadline = time.monotonic() + args.seconds
    moves = 0
    rejected = 0
    clicks_l = 0
    clicks_r = 0
    last_log = time.monotonic()
    last_x = last_y = None
    last_iris: tuple[float, float] | None = None
    # Wink detection state
    ear_history_l: deque[float] = deque(maxlen=30)
    ear_history_r: deque[float] = deque(maxlen=30)
    last_click_ts: float = 0.0
    click_cooldown_s = 1.5
    wink_low_ratio = 0.6   # closed eye drops below 60% of its baseline
    wink_high_ratio = 0.85  # other eye must stay above 85% of its baseline
    try:
        for raw_frame, _ in tracker.stream():
            if time.monotonic() >= deadline:
                break
            # Mirror the frame so user's left = cursor's left.
            frame = cv2.flip(raw_frame, 1)
            del raw_frame
            result = tracker.process(frame)
            del frame
            if result is None:
                continue
            ix = (result.left_iris[0] + result.right_iris[0]) / 2.0
            iy = (result.left_iris[1] + result.right_iris[1]) / 2.0
            # Stability filter: reject sudden iris jumps >100 px (likely a
            # Haar false positive picking up a non-face region).
            if last_iris is not None:
                dx = ix - last_iris[0]
                dy = iy - last_iris[1]
                if (dx * dx + dy * dy) ** 0.5 > 100:
                    rejected += 1
                    last_iris = (ix, iy)
                    continue
            last_iris = (ix, iy)
            target_x = screen_w / 2 + (ix - cam_cx) * args.gain
            target_y = screen_h / 2 + (iy - cam_cy) * args.gain
            sx, sy = kalman.update(target_x, target_y, ts=result.timestamp)
            sx = max(2, min(screen_w - 2, int(sx)))
            sy = max(2, min(screen_h - 2, int(sy)))
            gate.update(result.confidence, result.timestamp)
            if cursor.move(sx, sy):
                moves += 1
                last_x, last_y = sx, sy

            # --- wink-to-click detection ---
            ear_history_l.append(result.left_ear)
            ear_history_r.append(result.right_ear)
            now = time.monotonic()
            if (
                len(ear_history_l) >= 10
                and now - last_click_ts > click_cooldown_s
                and not gate.is_blocked()
            ):
                # Open-eye baseline = 90th percentile of recent EARs.
                base_l = sorted(ear_history_l)[int(len(ear_history_l) * 0.9)]
                base_r = sorted(ear_history_r)[int(len(ear_history_r) * 0.9)]
                ratio_l = result.left_ear / base_l if base_l > 0 else 1.0
                ratio_r = result.right_ear / base_r if base_r > 0 else 1.0
                if ratio_l < wink_low_ratio and ratio_r > wink_high_ratio:
                    cursor.click("left")
                    clicks_l += 1
                    last_click_ts = now
                    print(f"  *** WINK_LEFT  -> LEFT CLICK  at ({sx},{sy})  "
                          f"(L={result.left_ear:.2f}/{base_l:.2f}={ratio_l:.2f}, "
                          f"R={result.right_ear:.2f}/{base_r:.2f}={ratio_r:.2f})")
                elif ratio_r < wink_low_ratio and ratio_l > wink_high_ratio:
                    cursor.click("right")
                    clicks_r += 1
                    last_click_ts = now
                    print(f"  *** WINK_RIGHT -> RIGHT CLICK at ({sx},{sy})  "
                          f"(L={result.left_ear:.2f}/{base_l:.2f}={ratio_l:.2f}, "
                          f"R={result.right_ear:.2f}/{base_r:.2f}={ratio_r:.2f})")

            if now - last_log >= 1.0:
                print(f"  t={now-deadline+args.seconds:5.1f}s  cursor=({sx:4d},{sy:4d})"
                      f"  iris=({ix:5.1f},{iy:5.1f})  EAR L={result.left_ear:.2f}"
                      f" R={result.right_ear:.2f}  conf={result.confidence:.2f}")
                last_log = now
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except pyautogui.FailSafeException:
        print("\nFAIL-SAFE triggered (mouse in screen corner). Stopping.")
    finally:
        kill.uninstall()
    print(f"\nMoved cursor {moves} times. Rejected {rejected} jumpy frames.")
    print(f"Clicks: left={clicks_l}, right={clicks_r}. "
          f"Last position: {last_x},{last_y}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
