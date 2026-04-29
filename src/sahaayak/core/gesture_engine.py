"""Blink, dwell, wink, and gaze-zone gesture state machine.

Inputs per tick: an `EyeTrackResult` and a smoothed `GazePoint`. Output: a
sequence of `Gesture` events (most ticks emit zero or one).

State machine outline:

    EAR-based blink detector
        SHORT_BLINK  (80-250 ms, both eyes closed-open)
        LONG_BLINK   (>500 ms)
        DOUBLE_BLINK (two short blinks within 600 ms)
        WINK_LEFT    (right eye stays open, left EAR drops only)
        WINK_RIGHT   (mirror)
    Dwell detector
        DWELL        (gaze still inside an N-px disc for `dwell_click_ms`)
    Gaze-zone detector
        GAZE_ZONE_TOP / GAZE_ZONE_BOTTOM (gaze in top/bottom 8% of screen)
        GAZE_OFF_SCREEN (gaze invalid for >2 s)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.core.eye_tracker import EyeTrackResult
    from sahaayak.core.gaze_estimator import GazePoint

logger = get_logger(__name__)


class GestureKind(str, Enum):
    """Recognised gestures."""

    SHORT_BLINK = "short_blink"
    LONG_BLINK = "long_blink"
    DOUBLE_BLINK = "double_blink"
    DWELL = "dwell"
    WINK_LEFT = "wink_left"
    WINK_RIGHT = "wink_right"
    GAZE_ZONE_TOP = "gaze_zone_top"
    GAZE_ZONE_BOTTOM = "gaze_zone_bottom"
    GAZE_OFF_SCREEN = "gaze_off_screen"


@dataclass(frozen=True)
class Gesture:
    """A detected gesture event."""

    kind: GestureKind
    timestamp: float
    payload: dict[str, Any] | None = None


class GestureEngine:
    """State machine that turns gaze + eyelid signals into gesture events."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("gestures", {})
        self._short_min, self._short_max = cfg.get("short_blink_ms", [80, 250])
        self._long_ms = int(cfg.get("long_blink_ms", 500))
        self._double_window_ms = int(cfg.get("double_blink_window_ms", 600))
        self._dwell_ms = int(cfg.get("dwell_click_ms", 800))
        self._off_screen_ms = int(cfg.get("gaze_off_screen_pause_ms", 2000))
        self._ear_thresh = float(cfg.get("ear_blink_threshold", 0.21))
        self._wink_asym = float(cfg.get("ear_wink_asymmetry", 0.15))
        self._wink_cooldown_s = float(cfg.get("wink_cooldown_ms", 1200)) / 1000.0
        self._last_wink_ts: float = 0.0

        self._closed_since: float | None = None
        self._left_closed_since: float | None = None
        self._right_closed_since: float | None = None
        self._last_short_blink_ts: float | None = None
        self._dwell_anchor: tuple[float, float] | None = None
        self._dwell_anchor_ts: float = 0.0
        self._off_screen_since: float | None = None
        self._recent: deque[Gesture] = deque(maxlen=16)

    @property
    def recent(self) -> list[Gesture]:
        """Last N gestures, newest last (debug aid)."""
        return list(self._recent)

    def step(
        self,
        result: EyeTrackResult,
        gaze: GazePoint,
        screen_size: tuple[int, int],
    ) -> list[Gesture]:
        """Advance the state machine by one tick. Returns 0+ gestures."""
        # `or` would treat 0.0 as falsy and clobber the test/replay timeline.
        now = result.timestamp if result.timestamp is not None else time.monotonic()
        gestures: list[Gesture] = []

        # --- gaze-zone & off-screen detection -------------------------------
        sw, sh = screen_size
        if gaze.confidence < 0.3:
            if self._off_screen_since is None:
                self._off_screen_since = now
            elif (now - self._off_screen_since) * 1000 >= self._off_screen_ms:
                gestures.append(Gesture(GestureKind.GAZE_OFF_SCREEN, now))
                self._off_screen_since = now  # debounce
        else:
            self._off_screen_since = None
            top_band = sh * 0.08
            bot_band = sh * 0.92
            if gaze.y <= top_band:
                gestures.append(Gesture(GestureKind.GAZE_ZONE_TOP, now))
            elif gaze.y >= bot_band:
                gestures.append(Gesture(GestureKind.GAZE_ZONE_BOTTOM, now))

        # --- blink / wink detection -----------------------------------------
        left_closed = result.left_ear < self._ear_thresh
        right_closed = result.right_ear < self._ear_thresh
        both_closed = left_closed and right_closed

        # Track closed-since timestamps per eye
        if left_closed and self._left_closed_since is None:
            self._left_closed_since = now
        elif not left_closed:
            self._left_closed_since = None
        if right_closed and self._right_closed_since is None:
            self._right_closed_since = now
        elif not right_closed:
            self._right_closed_since = None

        if both_closed:
            if self._closed_since is None:
                self._closed_since = now
        else:
            if self._closed_since is not None:
                duration_ms = (now - self._closed_since) * 1000
                self._closed_since = None
                if duration_ms >= self._long_ms:
                    gestures.append(Gesture(GestureKind.LONG_BLINK, now))
                elif self._short_min <= duration_ms <= self._short_max:
                    last = self._last_short_blink_ts
                    if last is not None and (now - last) * 1000 <= self._double_window_ms:
                        gestures.append(Gesture(GestureKind.DOUBLE_BLINK, now))
                        self._last_short_blink_ts = None
                    else:
                        gestures.append(Gesture(GestureKind.SHORT_BLINK, now))
                        self._last_short_blink_ts = now
            else:
                # Detect winks: one eye closed asymmetrically.
                # Cooldown prevents the same wink from firing multiple times
                # while the user holds the eye shut.
                if (now - self._last_wink_ts) > self._wink_cooldown_s:
                    if left_closed and not right_closed and (
                        result.right_ear - result.left_ear > self._wink_asym
                    ):
                        gestures.append(Gesture(GestureKind.WINK_LEFT, now))
                        self._left_closed_since = None
                        self._last_wink_ts = now
                    elif right_closed and not left_closed and (
                        result.left_ear - result.right_ear > self._wink_asym
                    ):
                        gestures.append(Gesture(GestureKind.WINK_RIGHT, now))
                        self._right_closed_since = None
                        self._last_wink_ts = now

        # --- dwell detection ------------------------------------------------
        if gaze.confidence >= 0.6:
            if self._dwell_anchor is None:
                self._dwell_anchor = (gaze.x, gaze.y)
                self._dwell_anchor_ts = now
            else:
                dx = gaze.x - self._dwell_anchor[0]
                dy = gaze.y - self._dwell_anchor[1]
                if (dx * dx + dy * dy) ** 0.5 > 30.0:
                    self._dwell_anchor = (gaze.x, gaze.y)
                    self._dwell_anchor_ts = now
                elif (now - self._dwell_anchor_ts) * 1000 >= self._dwell_ms:
                    gestures.append(
                        Gesture(
                            GestureKind.DWELL,
                            now,
                            payload={"x": self._dwell_anchor[0], "y": self._dwell_anchor[1]},
                        )
                    )
                    self._dwell_anchor = None
        else:
            self._dwell_anchor = None

        for g in gestures:
            self._recent.append(g)
        return gestures
