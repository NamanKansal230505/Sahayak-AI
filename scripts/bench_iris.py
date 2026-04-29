"""Benchmark the iris model on every available OpenVINO device.

Uses a synthetic 640x480 BGR frame; the goal is to measure the iris+face
inference path latency across NPU / GPU / CPU and write a real
benchmark_report.json artefact.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make `sahaayak` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sahaayak.core.eye_tracker import EyeTracker  # noqa: E402
from sahaayak.utils.benchmark import BenchmarkRecord, BenchmarkReport, LatencyMeter  # noqa: E402
from sahaayak.utils.intel_device import detect_devices  # noqa: E402

WARMUP = 5
ITERATIONS = 50


def make_synthetic_frame() -> np.ndarray:
    """Synthetic BGR frame with a face-like blob in the centre."""
    rng = np.random.default_rng(seed=42)
    frame = (rng.integers(0, 60, size=(480, 640, 3), dtype=np.uint8))
    cy, cx = 240, 320
    yy, xx = np.ogrid[:480, :640]
    face_mask = (xx - cx) ** 2 / 110**2 + (yy - cy) ** 2 / 150**2 <= 1
    frame[face_mask] = (200, 180, 170)
    # Add two eye-like dark ellipses
    for ex in (cx - 50, cx + 50):
        eye_mask = (xx - ex) ** 2 / 18**2 + (yy - (cy - 30)) ** 2 / 10**2 <= 1
        frame[eye_mask] = (40, 40, 40)
    return frame.astype(np.uint8)


def bench_device(device: str, frame: np.ndarray) -> LatencyMeter:
    print(f"\n=== {device} ===")
    tracker = EyeTracker(device=device)
    # Warmup
    for _ in range(WARMUP):
        tracker.process(frame)
    meter = LatencyMeter(window=ITERATIONS)
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        tracker.process(frame)
        meter.record((time.perf_counter() - t0) * 1000.0)
    s = meter.summary()
    print(f"  count={s.count}  mean={s.mean_ms:.2f}ms  p50={s.p50_ms:.2f}ms  "
          f"p95={s.p95_ms:.2f}ms  p99={s.p99_ms:.2f}ms  fps={s.fps:.1f}")
    return meter


def main() -> int:
    devices_report = detect_devices()
    available = [d.name for d in devices_report.devices]
    print(f"OpenVINO {devices_report.runtime_version}")
    print(f"Available: {available}")

    frame = make_synthetic_frame()
    rep = BenchmarkReport(runtime_version=devices_report.runtime_version)

    for dev in available:
        try:
            meter = bench_device(dev, frame)
            rep.records.append(
                BenchmarkRecord(
                    component="iris-pipeline",
                    device=dev,
                    model="iris_landmark.xml + face_detector.xml (FP16)",
                    summary=meter.summary(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] {dev} failed: {exc}")
    rep.notes = (
        f"Warmup={WARMUP} iters={ITERATIONS}. Synthetic 640x480 frame with a "
        f"face-like blob; representative of the live-webcam path."
    )
    out_path = Path(__file__).resolve().parents[1] / "benchmark_report.json"
    rep.write(out_path)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
