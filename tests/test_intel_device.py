"""Tests for sahaayak.utils.intel_device.

These tests do not require an actual OpenVINO install — when OpenVINO is
absent, detect_devices() must return a graceful 'unavailable' report.
"""

from __future__ import annotations

from sahaayak.utils.intel_device import (
    DeviceInfo,
    DeviceReport,
    detect_devices,
    format_report,
    get_best_device,
)


def test_detect_devices_returns_report() -> None:
    report = detect_devices()
    assert isinstance(report, DeviceReport)
    assert isinstance(report.runtime_version, str)


def test_format_report_renders_header() -> None:
    report = detect_devices()
    text = format_report(report)
    assert "SahaayakAI" in text
    assert "OpenVINO runtime" in text


def test_get_best_device_falls_back_to_cpu() -> None:
    # Even with no devices reported, we must return a usable device id.
    chosen = get_best_device("vision")
    assert chosen in {"NPU", "GPU", "GPU.0", "GPU.1", "CPU"} or chosen.startswith("GPU")


def test_best_for_prefers_npu_over_gpu_over_cpu() -> None:
    report = DeviceReport(
        runtime_version="2024.4.0-test",
        devices=[
            DeviceInfo(name="CPU", full_name="cpu"),
            DeviceInfo(name="GPU.0", full_name="gpu"),
            DeviceInfo(name="NPU", full_name="npu"),
        ],
    )
    assert report.best_for("vision") == "NPU"
    assert report.best_for("llm") == "NPU"


def test_best_for_skips_missing_npu() -> None:
    report = DeviceReport(
        runtime_version="2024.4.0-test",
        devices=[
            DeviceInfo(name="CPU", full_name="cpu"),
            DeviceInfo(name="GPU.0", full_name="gpu"),
        ],
    )
    assert report.best_for("vision") == "GPU.0"


def test_best_for_returns_none_when_empty() -> None:
    report = DeviceReport(runtime_version="x", devices=[])
    assert report.best_for("vision") is None
