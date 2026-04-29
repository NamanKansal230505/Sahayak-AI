"""Transparent always-on-top gaze-cursor overlay.

A frameless, click-through PyQt6 widget that draws a small reticle at the
current smoothed gaze coordinate. Click-through is implemented via the
`WindowTransparentForInput` flag so the overlay never steals OS-level
cursor events from the actual application beneath.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.core.gaze_estimator import GazePoint

logger = get_logger(__name__)


class GazeOverlay:
    """A floating reticle that follows the current gaze point.

    The widget is constructed lazily on `start()` so importing the module
    in headless tests does not require a display.
    """

    def __init__(self, screen_size: tuple[int, int]) -> None:
        self._w, self._h = screen_size
        self._widget: object | None = None
        self._x: float = self._w / 2
        self._y: float = self._h / 2
        self._confidence: float = 0.0

    def start(self) -> None:
        """Construct and show the overlay widget."""
        try:
            from PyQt6.QtCore import Qt, QTimer  # noqa: PLC0415
            from PyQt6.QtGui import QColor, QPainter, QPen  # noqa: PLC0415
            from PyQt6.QtWidgets import QApplication, QWidget  # noqa: PLC0415
        except ImportError:  # pragma: no cover
            logger.error("PyQt6 unavailable; overlay disabled.")
            return

        app = QApplication.instance() or QApplication([])
        outer = self

        class _Widget(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.Tool
                )
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                self.setGeometry(0, 0, outer._w, outer._h)
                self._timer = QTimer(self)
                self._timer.timeout.connect(self.update)
                self._timer.start(33)  # ~30 FPS repaint

            def paintEvent(self, _event: object) -> None:  # noqa: N802
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                # Reticle colour reflects confidence: green (good) -> red (bad).
                conf = max(0.0, min(1.0, outer._confidence))
                color = QColor(
                    int(255 * (1 - conf)),
                    int(255 * conf),
                    80,
                    220,
                )
                pen = QPen(color, 3)
                painter.setPen(pen)
                cx, cy = int(outer._x), int(outer._y)
                painter.drawEllipse(cx - 14, cy - 14, 28, 28)
                painter.drawLine(cx - 22, cy, cx - 8, cy)
                painter.drawLine(cx + 8, cy, cx + 22, cy)
                painter.drawLine(cx, cy - 22, cx, cy - 8)
                painter.drawLine(cx, cy + 8, cx, cy + 22)

        widget = _Widget()
        widget.show()
        self._widget = widget
        _ = app  # keep reference

    def update_position(self, gaze: GazePoint) -> None:
        """Push the latest gaze point into the overlay."""
        self._x = gaze.x
        self._y = gaze.y
        self._confidence = gaze.confidence

    def stop(self) -> None:
        """Hide and release the overlay widget."""
        widget = self._widget
        if widget is not None and hasattr(widget, "close"):
            widget.close()  # type: ignore[union-attr]
            self._widget = None
