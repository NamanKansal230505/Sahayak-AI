"""Docked side panel that renders LLM outputs.

The panel is a frameless PyQt6 widget docked to the right edge of the
primary screen. It always shows the AI disclaimer at the bottom and the
"long-blink to send" affordance whenever a draft reply is selected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sahaayak.utils.i18n import i18n
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.copilot.email_simplifier import SimplifiedEmail


logger = get_logger(__name__)


class CopilotPanel:
    """Side-panel renderer for the LLM output."""

    def __init__(self, screen_size: tuple[int, int]) -> None:
        self._w, self._h = screen_size
        self._widget: object | None = None
        self._on_send: Callable[[str], None] | None = None
        self._draft_index: int | None = None

    def set_send_handler(self, handler: Callable[[str], None]) -> None:
        """Set the callback to invoke when the user confirms send."""
        self._on_send = handler

    def show_email(self, email: SimplifiedEmail) -> None:
        """Display a `SimplifiedEmail` in the panel."""
        try:
            from PyQt6.QtCore import Qt  # noqa: PLC0415
            from PyQt6.QtWidgets import (  # noqa: PLC0415
                QApplication,
                QButtonGroup,
                QLabel,
                QListWidget,
                QListWidgetItem,
                QPushButton,
                QVBoxLayout,
                QWidget,
            )
        except ImportError:  # pragma: no cover
            logger.error("PyQt6 unavailable; copilot panel disabled.")
            return

        app = QApplication.instance() or QApplication([])
        panel_w = min(420, self._w // 3)
        outer = self

        class _Panel(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowFlags(
                    Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                )
                self.setStyleSheet(
                    "background:#0b1220;color:#e5e7eb;font-size:14pt;"
                )
                self.setGeometry(outer._w - panel_w, 0, panel_w, outer._h)
                v = QVBoxLayout(self)
                v.addWidget(QLabel(f"<b>{i18n('copilot.tldr')}</b>"))
                v.addWidget(QLabel(email.tldr))
                v.addWidget(QLabel(f"<b>{i18n('copilot.action_items')}</b>"))
                for ai in email.action_items:
                    deadline = f" — {ai.deadline}" if ai.deadline else ""
                    v.addWidget(QLabel(f"• {ai.task}{deadline}"))
                v.addWidget(QLabel(f"<b>{i18n('copilot.suggested_replies')}</b>"))
                lst = QListWidget()
                for d in email.suggested_reply_drafts:
                    QListWidgetItem(d, lst)
                v.addWidget(lst)
                send = QPushButton("Long-blink to send")
                send.setStyleSheet("background:#16a34a;color:white;padding:12px;font-size:14pt;")

                def _on_send_clicked() -> None:
                    item = lst.currentItem()
                    if item is None or outer._on_send is None:
                        return
                    outer._on_send(item.text())

                send.clicked.connect(_on_send_clicked)
                v.addWidget(send)
                v.addWidget(QLabel(f"<i>{i18n('copilot.disclaimer')}</i>"))
                _ = QButtonGroup  # silence unused-import lint

        widget = _Panel()
        widget.show()
        self._widget = widget
        _ = app

    def toggle_visible(self) -> None:
        """Show or hide the existing panel."""
        widget = self._widget
        if widget is None:
            return
        if hasattr(widget, "isVisible") and widget.isVisible():  # type: ignore[union-attr]
            widget.hide()  # type: ignore[union-attr]
        else:
            widget.show()  # type: ignore[union-attr]

    def close(self) -> None:
        widget = self._widget
        if widget is not None and hasattr(widget, "close"):
            widget.close()  # type: ignore[union-attr]
            self._widget = None

    def render_dict(self, payload: dict[str, Any]) -> None:
        """Generic render for non-email payloads (mind-map, meeting summary)."""
        # Convenience wrapper used by demos / tests.
        logger.info("Rendering generic payload with keys: %s", list(payload.keys()))
