"""20-20-20 mandatory eye-rest reminder.

Per spec:

* Fires every 20 minutes by default.
* Cannot be permanently disabled — only snoozed up to 1 hour.
* Disappears automatically after `rest_reminder_duration_seconds`.

The scheduler is a small background `threading.Timer`. The popup itself
falls back to stderr when PyQt6 is unavailable.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from sahaayak.utils.i18n import i18n
from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)


class RestReminder:
    """Periodic eye-rest reminder.

    Args:
        config: Loaded SahaayakAI config dict.
        notifier: Optional callable that displays the reminder. Defaults
            to a Qt popup if available, else a stderr log line.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        notifier: Callable[[], None] | None = None,
    ) -> None:
        cfg = (config or {}).get("safety", {})
        self._interval = float(cfg.get("rest_reminder_minutes", 20)) * 60.0
        self._duration = float(cfg.get("rest_reminder_duration_seconds", 20))
        self._max_snooze = float(cfg.get("rest_snooze_max_minutes", 60)) * 60.0
        self._notifier = notifier or _default_notifier
        self._timer: threading.Timer | None = None
        self._snoozed_until: float = 0.0
        self._stopped: bool = False

    def start(self) -> None:
        """Begin the recurring reminder."""
        self._stopped = False
        self._schedule(self._interval)

    def stop(self) -> None:
        """Stop the timer (e.g. on app exit)."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def snooze(self, minutes: float) -> None:
        """Snooze for up to `rest_snooze_max_minutes`. Cannot disable forever."""
        seconds = max(0.0, min(float(minutes) * 60.0, self._max_snooze))
        self._snoozed_until = time.monotonic() + seconds
        logger.info("Rest reminder snoozed for %.0fs.", seconds)

    def _schedule(self, delay: float) -> None:
        if self._stopped:
            return
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        if self._stopped:
            return
        if time.monotonic() < self._snoozed_until:
            self._schedule(self._snoozed_until - time.monotonic())
            return
        try:
            self._notifier()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rest reminder notifier raised: %s", exc)
        # Reschedule unconditionally — cannot be permanently disabled.
        self._schedule(self._interval)


def _default_notifier() -> None:
    """Default notifier: Qt popup if available, else stderr log."""
    try:
        from PyQt6.QtCore import QTimer  # noqa: PLC0415
        from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: PLC0415
    except ImportError:
        logger.warning("[REST] %s", i18n("rest.message"))
        return
    app = QApplication.instance() or QApplication([])
    box = QMessageBox()
    box.setWindowTitle(i18n("rest.title"))
    box.setText(i18n("rest.message"))
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    QTimer.singleShot(20_000, box.close)
    box.exec()
    _ = app
