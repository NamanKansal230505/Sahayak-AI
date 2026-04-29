"""Tests for the focus nudger."""

from __future__ import annotations

from sahaayak.copilot.focus_nudger import FocusNudger


def test_nudge_fires_after_threshold_switches() -> None:
    titles = iter(["a", "b", "c", "d", "e", "f"])
    fired: list[bool] = []
    n = FocusNudger(
        on_nudge=lambda: fired.append(True),
        threshold=5,
        window_seconds=10,
        title_provider=lambda: next(titles, ""),
    )
    for t in range(5):
        n.step(now=float(t))
    n.step(now=5.0)
    assert fired, "Expected nudge after 5 distinct switches in 10s"


def test_nudge_does_not_fire_when_idle() -> None:
    fired: list[bool] = []
    n = FocusNudger(
        on_nudge=lambda: fired.append(True),
        threshold=3,
        window_seconds=10,
        title_provider=lambda: "same-window",
    )
    for t in range(10):
        n.step(now=float(t))
    assert not fired
