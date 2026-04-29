"""Latches the cursor controller closed when gaze confidence drops.

Per spec: if confidence < 0.6 for >500ms, freeze cursor control. The gate
is a small state machine consulted by `cursor_controller` on every tick.
"""

from __future__ import annotations

import time
from threading import RLock


class ConfidenceGate:
    """Block downstream actions when gaze confidence is poor.

    Args:
        min_confidence: Threshold below which we start the latch timer.
        grace_ms: How long confidence must remain low before we block.
    """

    def __init__(self, min_confidence: float = 0.6, grace_ms: int = 500) -> None:
        self._min = float(min_confidence)
        self._grace_s = grace_ms / 1000.0
        self._lock = RLock()
        self._below_since: float | None = None
        self._blocked: bool = False
        self._manual_block: bool = False

    def update(self, confidence: float, ts: float | None = None) -> bool:
        """Push a new confidence sample. Returns the *blocked* state.

        Args:
            confidence: Latest confidence reading (0..1).
            ts: Optional timestamp; defaults to ``time.monotonic``.

        Returns:
            ``True`` if the gate is currently blocking actions.
        """
        ts = float(ts) if ts is not None else time.monotonic()
        with self._lock:
            if confidence < self._min:
                if self._below_since is None:
                    self._below_since = ts
                if ts - self._below_since >= self._grace_s:
                    self._blocked = True
            else:
                self._below_since = None
                self._blocked = False
            return self._blocked or self._manual_block

    def is_blocked(self) -> bool:
        """Return the current blocked state without updating it."""
        with self._lock:
            return self._blocked or self._manual_block

    def force_block(self, blocked: bool) -> None:
        """Manually pin the gate open or shut (used by the kill switch)."""
        with self._lock:
            self._manual_block = bool(blocked)
