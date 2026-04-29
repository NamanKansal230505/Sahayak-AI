"""On-screen predictive eye keyboard.

Two layouts ship: ``qwerty`` (English) and ``devanagari`` (Hindi). Keys
are 80x80 px minimum, high-contrast, and respond to either gaze-dwell or
short-blink confirmation. The top row holds 3 LLM-generated next-word
predictions; clicking one inserts the word + space.

Typing path: the keyboard widget computes the typed string and asks the
shared `CursorController.type_text` to send it to whichever app currently
holds focus. The keyboard widget is itself click-through-friendly so the
target text field keeps focus.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.control.cursor_controller import CursorController

logger = get_logger(__name__)


QWERTY_ROWS: tuple[tuple[str, ...], ...] = (
    ("q", "w", "e", "r", "t", "y", "u", "i", "o", "p"),
    ("a", "s", "d", "f", "g", "h", "j", "k", "l"),
    ("z", "x", "c", "v", "b", "n", "m"),
    ("space", "back", "enter", "lang"),
)

DEVANAGARI_ROWS: tuple[tuple[str, ...], ...] = (
    ("क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ"),
    ("त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ"),
    ("म", "य", "र", "ल", "व", "श", "स", "ह"),
    ("space", "back", "enter", "lang"),
)


@dataclass
class KeyboardState:
    """Mutable state shared between widget and predictor."""

    layout: str = "qwerty"
    buffer: str = ""
    predictions: list[str] = field(default_factory=list)
    visible: bool = False


class EyeKeyboard:
    """High-level keyboard controller (UI agnostic core).

    Args:
        cursor: The cursor controller used to send keystrokes to the OS.
        predict: Optional callable that takes the current buffer and returns
            up to N suggested next words. Wired to `LLMEngine.predict_next`.
        config: Loaded SahaayakAI config.
    """

    def __init__(
        self,
        cursor: CursorController,
        predict: Callable[[str], list[str]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        kb_cfg = (config or {}).get("keyboard", {})
        self._cursor = cursor
        self._predict = predict
        self._dwell_ms = int(kb_cfg.get("dwell_to_type_ms", 600))
        self._key_min = int(kb_cfg.get("key_min_size_px", 80))
        self._n_pred = int(kb_cfg.get("prediction_count", 3))
        layout = kb_cfg.get("layout", "qwerty")
        self.state = KeyboardState(layout=layout)
        self._widget: object | None = None

    # ------------------------------------------------------------------ rows
    def rows(self) -> tuple[tuple[str, ...], ...]:
        """Return the active layout rows."""
        return DEVANAGARI_ROWS if self.state.layout == "devanagari" else QWERTY_ROWS

    # -------------------------------------------------------------- key path
    def press_key(self, key: str) -> None:
        """Handle a single key activation (from dwell, click, or test)."""
        if key == "space":
            self.state.buffer += " "
            self._cursor.type_text(" ")
        elif key == "back":
            self.state.buffer = self.state.buffer[:-1]
            self._cursor.press("backspace")
        elif key == "enter":
            self.state.buffer += "\n"
            self._cursor.press("enter")
        elif key == "lang":
            self.state.layout = "devanagari" if self.state.layout == "qwerty" else "qwerty"
            logger.info("Keyboard layout switched to %s", self.state.layout)
        else:
            self.state.buffer += key
            self._cursor.type_text(key)
        self._refresh_predictions()

    def insert_prediction(self, word: str) -> None:
        """Insert a predicted word and a trailing space."""
        # If the user is mid-word, replace the partial prefix.
        prefix_len = len(self.state.buffer) - self.state.buffer.rfind(" ") - 1
        for _ in range(prefix_len):
            self._cursor.press("backspace")
        self._cursor.type_text(word + " ")
        # Update internal buffer
        if " " in self.state.buffer:
            head = self.state.buffer.rsplit(" ", 1)[0]
            self.state.buffer = head + " " + word + " "
        else:
            self.state.buffer = word + " "
        self._refresh_predictions()

    def _refresh_predictions(self) -> None:
        if self._predict is None:
            self.state.predictions = []
            return
        try:
            words = self._predict(self.state.buffer)[: self._n_pred]
            self.state.predictions = list(words)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prediction call failed: %s", exc)
            self.state.predictions = []

    # ---------------------------------------------------------- visibility
    def toggle_visible(self) -> None:
        self.state.visible = not self.state.visible
        if self.state.visible:
            self._show_widget()
        else:
            self._hide_widget()

    def _show_widget(self) -> None:
        try:
            from PyQt6.QtCore import Qt  # noqa: PLC0415
            from PyQt6.QtWidgets import (  # noqa: PLC0415
                QApplication,
                QGridLayout,
                QPushButton,
                QWidget,
            )
        except ImportError:  # pragma: no cover
            logger.error("PyQt6 unavailable; eye keyboard hidden.")
            return

        app = QApplication.instance() or QApplication([])
        outer = self

        class _KbWidget(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                )
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
                self.setStyleSheet(
                    "background:#0b1220;color:#fff;"
                    "QPushButton{background:#1f2937;color:#fff;border-radius:8px;font-size:24pt;}"
                    "QPushButton:hover{background:#374151;}"
                )
                self._grid = QGridLayout(self)
                self._render()

            def _render(self) -> None:
                while self._grid.count():
                    child = self._grid.takeAt(0)
                    if child and child.widget():
                        child.widget().deleteLater()
                # Predictions row
                for col, word in enumerate(outer.state.predictions[: outer._n_pred]):
                    btn = QPushButton(word)
                    btn.setMinimumSize(outer._key_min * 2, outer._key_min)
                    btn.clicked.connect(lambda _, w=word: (outer.insert_prediction(w), self._render()))
                    self._grid.addWidget(btn, 0, col * 2, 1, 2)
                # Letter rows
                for r, row in enumerate(outer.rows(), start=1):
                    for c, key in enumerate(row):
                        btn = QPushButton({"space": "Space", "back": "<-", "enter": "Enter", "lang": "अ/A"}.get(key, key))
                        btn.setMinimumSize(outer._key_min, outer._key_min)
                        btn.clicked.connect(lambda _, k=key: (outer.press_key(k), self._render()))
                        self._grid.addWidget(btn, r, c)

        widget = _KbWidget()
        widget.show()
        self._widget = widget
        _ = app

    def _hide_widget(self) -> None:
        widget = self._widget
        if widget is not None and hasattr(widget, "close"):
            widget.close()  # type: ignore[union-attr]
            self._widget = None
