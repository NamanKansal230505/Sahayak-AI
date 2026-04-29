"""Tests for the benchmark utilities."""

from __future__ import annotations

import json
from pathlib import Path

from sahaayak.utils.benchmark import BenchmarkRecord, BenchmarkReport, LatencyMeter


def test_latency_meter_records_samples() -> None:
    m = LatencyMeter()
    for _ in range(5):
        with m.measure():
            sum(range(100))
    s = m.summary()
    assert s.count == 5
    assert s.mean_ms >= 0.0
    assert s.fps >= 0.0


def test_latency_meter_summary_when_empty() -> None:
    s = LatencyMeter().summary()
    assert s.count == 0
    assert s.mean_ms == 0.0


def test_benchmark_report_writes_json(tmp_path: Path) -> None:
    m = LatencyMeter()
    m.record(10.0)
    rep = BenchmarkReport(
        runtime_version="2024.4.0",
        records=[
            BenchmarkRecord(
                component="iris", device="CPU", model="iris.xml", summary=m.summary()
            )
        ],
    )
    out = tmp_path / "bench.json"
    rep.write(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["runtime_version"] == "2024.4.0"
    assert data["records"][0]["component"] == "iris"
