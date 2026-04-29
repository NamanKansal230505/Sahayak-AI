"""Headless webcam demo: capture N frames, print iris coords + EAR + confidence.

No display is opened — the script proves the full live pipeline works against
your actual webcam on the configured device. Press Ctrl+C to stop early.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sahaayak.core.eye_tracker import EyeTracker  # noqa: E402
from sahaayak.utils.config import load_config  # noqa: E402
from sahaayak.utils.intel_device import detect_devices  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config()
    rep = detect_devices()
    print(f"OpenVINO {rep.runtime_version}")
    print(f"Devices: {[d.name for d in rep.devices]}")

    tracker = EyeTracker(device=args.device, config=config)
    print(f"Tracker on: {tracker.device}")
    print(f"Look at the webcam. Capturing {args.frames} frames in...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    print("GO\n")

    found = 0
    start = time.perf_counter()
    for i, (frame, result) in enumerate(tracker.stream()):
        if i >= args.frames:
            break
        if result is None:
            print(f"  frame {i:03d}  (no face)")
            continue
        found += 1
        print(
            f"  frame {i:03d}  iris L=({result.left_iris[0]:6.1f},{result.left_iris[1]:6.1f})"
            f"  R=({result.right_iris[0]:6.1f},{result.right_iris[1]:6.1f})"
            f"  EAR L={result.left_ear:.2f} R={result.right_ear:.2f}"
            f"  conf={result.confidence:.2f}"
        )
        # Privacy: do not save the frame.
        del frame
    dur = time.perf_counter() - start
    print(f"\nFaces found in {found}/{args.frames} frames ({100*found/args.frames:.0f}%)")
    print(f"Wallclock: {dur:.1f}s -> ~{args.frames / dur:.1f} FPS end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
