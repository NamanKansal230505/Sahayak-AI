"""Detects rapid task-switching and surfaces a gentle, gaze-dismissable popup.

Privacy invariant enforced HERE: we hash the active-window title before it
enters our internal sliding window — so even in-RAM we hold a 8-byte digest
rather than a string that might be inspected by a swap-file or core dump.
We never log app names or window titles.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)


def _digest(title: str) -> bytes:
    return hashlib.sha256(title.encode("utf-8", errors="replace")).digest()[:8]


def _active_window_title() -> str:
    try:
        import pygetwindow  # noqa: PLC0415

        win = pygetwindow.getActiveWindow()
        return getattr(win, "title", "") or ""
    except Exception:  # noqa: BLE001 - defensive: any platform error -> empty
        return ""


class FocusNudger:
    """Counts window switches in a sliding window and fires a callback.

    Args:
        on_nudge: Callable invoked when the threshold is crossed.
        threshold: Number of distinct switches in the window.
        window_seconds: Sliding-window duration.
        title_provider: Override window-title source (used by tests).
    """

    def __init__(
        self,
        on_nudge: Callable[[], None],
        threshold: int = 5,
        window_seconds: float = 60.0,
        title_provider: Callable[[], str] | None = None,
    ) -> None:
        self._on_nudge = on_nudge
        self._threshold = int(threshold)
        self._window = float(window_seconds)
        self._title_provider = title_provider or _active_window_title
        self._hashes: deque[tuple[float, bytes]] = deque()
        self._last: bytes | None = None
        self._cooldown_until: float = 0.0

    def step(self, now: float | None = None) -> bool:
        """Tick the nudger. Returns ``True`` if a nudge fired this tick."""
        now = float(now) if now is not None else time.monotonic()
        title = self._title_provider()
        if not title:
            return False
        h = _digest(title)
        if h != self._last:
            self._hashes.append((now, h))
            self._last = h
        # Drop entries outside the window
        while self._hashes and now - self._hashes[0][0] > self._window:
            self._hashes.popleft()
        if (
            len(self._hashes) >= self._threshold
            and now > self._cooldown_until
        ):
            self._cooldown_until = now + self._window
            try:
                self._on_nudge()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Nudge callback raised: %s", exc)
            return True
        return False

    @classmethod
    def from_config(cls, on_nudge: Callable[[], None], config: dict[str, Any]) -> FocusNudger:
        cfg = config.get("focus_nudger", {})
        return cls(
            on_nudge=on_nudge,
            threshold=int(cfg.get("window_switch_threshold", 5)),
            window_seconds=float(cfg.get("window_switch_window_seconds", 60)),
        )
