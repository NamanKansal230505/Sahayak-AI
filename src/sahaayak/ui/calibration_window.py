"""Fullscreen 9-point calibration UI built on PyQt6.

Visually presents one red dot at a time on a black canvas. The dot turns
green once the dwell window completes. After the 9th point the routine
saves the resulting `CalibrationProfile` and emits ``finished``.

This module imports PyQt6 lazily so the rest of SahaayakAI can be tested
without a display. UI code is intentionally short — the math lives in
`core.calibrator`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sahaayak.core.calibrator import CalibrationProfile, Calibrator
from sahaayak.utils.i18n import i18n
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.core.eye_tracker import EyeTrackResult

logger = get_logger(__name__)


def run_calibration_ui(
    sample_source: Callable[[], EyeTrackResult | None],
    config: dict[str, Any] | None = None,
) -> CalibrationProfile | None:
    """Launch the fullscreen calibration UI. Returns the fitted profile."""
    try:
        from PyQt6.QtCore import QPoint, Qt, QTimer  # noqa: PLC0415
        from PyQt6.QtGui import QColor, QPainter  # noqa: PLC0415
        from PyQt6.QtWidgets import QApplication, QWidget  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover — depends on display
        logger.error("PyQt6 not installed: %s", exc)
        return None

    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen().size()
    screen_size = (screen.width(), screen.height())
    cal = Calibrator(screen_size, config)
    targets = cal.targets

    class _Window(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("SahaayakAI Calibration")
            self.setStyleSheet("background:black;")
            self.showFullScreen()
            self._idx = 0
            self._dot_state = "red"
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(50)
            self._dwell_ms = int((config or {}).get("calibration", {}).get("dwell_per_point_ms", 1500))
            self._elapsed = 0
            self.profile: CalibrationProfile | None = None

        def _tick(self) -> None:
            if self._idx >= len(targets):
                try:
                    self.profile = cal.fit()
                    self.profile.save()
                except RuntimeError as exc:
                    logger.error("Calibration failed: %s", exc)
                self._timer.stop()
                self.close()
                return
            target = targets[self._idx]
            sample = sample_source()
            if sample is not None:
                cal.collect_for_target(target, lambda s=sample: s, clock=lambda: 0.0)
            self._elapsed += 50
            if self._elapsed >= self._dwell_ms:
                self._dot_state = "green"
                self.update()
                # short pause then advance
                QTimer.singleShot(200, self._advance)

        def _advance(self) -> None:
            self._idx += 1
            self._elapsed = 0
            self._dot_state = "red"
            self.update()

        def paintEvent(self, _event: object) -> None:  # noqa: N802 — Qt API
            painter = QPainter(self)
            if self._idx >= len(targets):
                return
            x, y = targets[self._idx]
            color = QColor("#22c55e") if self._dot_state == "green" else QColor("#ef4444")
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPoint(x, y), 18, 18)
            painter.setPen(QColor("white"))
            painter.drawText(40, 40, i18n("calibration.intro"))

        def keyPressEvent(self, event: object) -> None:  # noqa: N802
            if event.key() == Qt.Key.Key_Escape:  # type: ignore[attr-defined]
                self._timer.stop()
                cal.reset()
                self.close()

    window = _Window()
    app.exec()
    return window.profile
