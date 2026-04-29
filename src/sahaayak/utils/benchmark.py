"""Latency / FPS profiler for OpenVINO inference paths.

A `LatencyMeter` wraps any callable and records per-call wall-clock latency.
A `Benchmark.run` helper drives the eye-tracker pipeline against a synthetic
or live frame source and emits a `benchmark_report.json` artefact.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)


@dataclass
class LatencySummary:
    """Aggregate latency stats in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float


class LatencyMeter:
    """Rolling-window latency / FPS recorder."""

    def __init__(self, window: int = 256) -> None:
        self._samples_ms: deque[float] = deque(maxlen=window)

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Context manager: records elapsed time on exit."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._samples_ms.append((time.perf_counter() - start) * 1000.0)

    def record(self, ms: float) -> None:
        """Add a single latency sample."""
        self._samples_ms.append(float(ms))

    def summary(self) -> LatencySummary:
        """Compute mean/p50/p95/p99 + derived FPS."""
        if not self._samples_ms:
            return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sorted_samples = sorted(self._samples_ms)
        n = len(sorted_samples)
        mean = statistics.fmean(sorted_samples)
        return LatencySummary(
            count=n,
            mean_ms=mean,
            p50_ms=sorted_samples[n // 2],
            p95_ms=sorted_samples[min(int(n * 0.95), n - 1)],
            p99_ms=sorted_samples[min(int(n * 0.99), n - 1)],
            fps=1000.0 / mean if mean > 0 else 0.0,
        )


@dataclass
class BenchmarkRecord:
    """One row of `benchmark_report.json`."""

    component: str
    device: str
    model: str
    summary: LatencySummary


@dataclass
class BenchmarkReport:
    """Full benchmark artefact written to disk after a run."""

    runtime_version: str
    records: list[BenchmarkRecord] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": self.runtime_version,
            "records": [
                {
                    "component": r.component,
                    "device": r.device,
                    "model": r.model,
                    "summary": asdict(r.summary),
                }
                for r in self.records
            ],
            "notes": self.notes,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Benchmark report written to %s", path)
