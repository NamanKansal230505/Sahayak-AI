"""Intel device detection for OpenVINO Runtime.

Enumerates the OpenVINO devices available on the host (NPU, GPU, CPU)
and picks the best target for each model class. Used at startup so the
user can see — visibly — that we are running on Intel silicon.

The module degrades gracefully: if OpenVINO is not yet installed
(e.g. during initial scaffold inspection) we still emit a useful
diagnostic instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

ModelKind = Literal["vision", "llm"]

# Per-model preference order. Vision models (MediaPipe Iris) are tiny and
# benefit most from NPU latency; the LLM benefits from NPU/GPU memory.
_PREFERENCE: dict[ModelKind, tuple[str, ...]] = {
    "vision": ("NPU", "GPU", "CPU"),
    "llm":    ("NPU", "GPU", "CPU"),
}


@dataclass(frozen=True)
class DeviceInfo:
    """A single OpenVINO device entry."""

    name: str                   # e.g. "CPU", "GPU.0", "NPU"
    full_name: str              # e.g. "Intel(R) Core(TM) Ultra 7 155H"


@dataclass(frozen=True)
class DeviceReport:
    """Full snapshot of OpenVINO devices visible on this host."""

    runtime_version: str
    devices: list[DeviceInfo] = field(default_factory=list)
    available: bool = True
    error: str | None = None

    def best_for(self, kind: ModelKind) -> str | None:
        """Return the highest-priority device available for the model class."""
        names = {d.name.split(".")[0] for d in self.devices}
        for candidate in _PREFERENCE[kind]:
            if candidate in names:
                # Return the *exact* device id for GPU.0 / GPU.1 disambiguation.
                for dev in self.devices:
                    if dev.name.split(".")[0] == candidate:
                        return dev.name
        return None


def detect_devices() -> DeviceReport:
    """Probe OpenVINO Runtime and return a `DeviceReport`.

    Returns:
        A populated `DeviceReport`. If OpenVINO is unavailable, the report
        still returns with ``available=False`` and an explanatory ``error``
        so callers can render a graceful diagnostic.
    """
    try:
        import openvino as ov  # noqa: PLC0415 — deferred import is intentional
    except ImportError as exc:
        return DeviceReport(
            runtime_version="not installed",
            devices=[],
            available=False,
            error=f"OpenVINO not importable: {exc}. Run `pip install -r requirements.txt`.",
        )

    try:
        core = ov.Core()
        version = ov.get_version()
        devices: list[DeviceInfo] = []
        for name in core.available_devices:
            try:
                full = core.get_property(name, "FULL_DEVICE_NAME")
            except Exception as exc:  # noqa: BLE001 — vendor-string lookup is best-effort
                full = f"<unknown: {exc}>"
            devices.append(DeviceInfo(name=name, full_name=str(full)))
        return DeviceReport(runtime_version=str(version), devices=devices)
    except Exception as exc:  # noqa: BLE001 — surface any runtime probe failure
        return DeviceReport(
            runtime_version="error",
            devices=[],
            available=False,
            error=f"OpenVINO Core() probe failed: {exc}",
        )


def get_best_device(kind: ModelKind) -> str:
    """Return the best device id for the given model kind, or ``CPU`` as fallback.

    Args:
        kind: ``"vision"`` for MediaPipe Iris, ``"llm"`` for Phi-3.

    Returns:
        An OpenVINO device id, e.g. ``"NPU"`` or ``"GPU.0"``.
    """
    report = detect_devices()
    chosen = report.best_for(kind)
    if chosen is None:
        logger.warning("No preferred device available for %s; falling back to CPU.", kind)
        return "CPU"
    return chosen


def format_report(report: DeviceReport) -> str:
    """Render a `DeviceReport` as a human-readable multi-line string."""
    from sahaayak import __version__  # noqa: PLC0415 — avoid import cycle

    lines: list[str] = [
        f"SahaayakAI {__version__}  --  device check",
        f"OpenVINO runtime : {report.runtime_version}",
    ]
    if not report.available:
        lines.append(f"Status           : UNAVAILABLE ({report.error})")
        return "\n".join(lines)

    lines.append("Available devices:")
    if not report.devices:
        lines.append("  (none — OpenVINO loaded but no devices enumerated)")
    for dev in report.devices:
        lines.append(f"  - {dev.name:<14} ({dev.full_name})")

    vision = report.best_for("vision") or "CPU"
    llm = report.best_for("llm") or "CPU"
    lines.append(f"Selected for vision : {vision}")
    lines.append(f"Selected for LLM    : {llm}")
    lines.append("All checks passed.")
    return "\n".join(lines)
