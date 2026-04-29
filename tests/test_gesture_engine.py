"""Tests for the gesture state machine."""

from __future__ import annotations

from dataclasses import dataclass

from sahaayak.core.gesture_engine import GestureEngine, GestureKind


@dataclass
class _Result:
    left_iris: tuple[float, float] = (0.0, 0.0)
    right_iris: tuple[float, float] = (0.0, 0.0)
    left_eye_landmarks: object = None
    right_eye_landmarks: object = None
    confidence: float = 0.9
    timestamp: float = 0.0
    left_ear: float = 0.3
    right_ear: float = 0.3


@dataclass
class _Gaze:
    x: float = 400.0
    y: float = 300.0
    confidence: float = 0.9
    timestamp: float = 0.0


def test_short_blink_emitted_on_brief_closure() -> None:
    eng = GestureEngine()
    screen = (800, 600)
    # Eyes open
    eng.step(_Result(timestamp=0.0), _Gaze(timestamp=0.0), screen)
    # Both closed
    eng.step(_Result(left_ear=0.1, right_ear=0.1, timestamp=0.05), _Gaze(timestamp=0.05), screen)
    # Open again 150 ms after closure start -> SHORT_BLINK
    out = eng.step(_Result(timestamp=0.20), _Gaze(timestamp=0.20), screen)
    kinds = {g.kind for g in out}
    assert GestureKind.SHORT_BLINK in kinds


def test_long_blink_emitted_after_500ms() -> None:
    eng = GestureEngine()
    screen = (800, 600)
    eng.step(_Result(left_ear=0.1, right_ear=0.1, timestamp=0.0), _Gaze(timestamp=0.0), screen)
    out = eng.step(_Result(timestamp=0.7), _Gaze(timestamp=0.7), screen)
    kinds = {g.kind for g in out}
    assert GestureKind.LONG_BLINK in kinds


def test_dwell_emitted_after_threshold() -> None:
    eng = GestureEngine(config={"gestures": {"dwell_click_ms": 200}})
    screen = (800, 600)
    eng.step(_Result(timestamp=0.0), _Gaze(timestamp=0.0), screen)
    eng.step(_Result(timestamp=0.1), _Gaze(timestamp=0.1), screen)
    out = eng.step(_Result(timestamp=0.3), _Gaze(timestamp=0.3), screen)
    kinds = {g.kind for g in out}
    assert GestureKind.DWELL in kinds


def test_gaze_zone_top_emitted() -> None:
    eng = GestureEngine()
    out = eng.step(_Result(timestamp=0.0), _Gaze(x=400, y=10, timestamp=0.0), (800, 600))
    assert any(g.kind is GestureKind.GAZE_ZONE_TOP for g in out)


def test_off_screen_emitted_on_low_confidence() -> None:
    eng = GestureEngine()
    eng.step(_Result(confidence=0.1, timestamp=0.0), _Gaze(confidence=0.1, timestamp=0.0), (800, 600))
    out = eng.step(_Result(confidence=0.1, timestamp=2.5), _Gaze(confidence=0.1, timestamp=2.5), (800, 600))
    assert any(g.kind is GestureKind.GAZE_OFF_SCREEN for g in out)
