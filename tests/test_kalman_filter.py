"""Tests for the gaze smoothing filters."""

from __future__ import annotations

from sahaayak.core.kalman_filter import GazeKalmanFilter, OneEuroFilter


def test_kalman_first_sample_is_passthrough() -> None:
    f = GazeKalmanFilter()
    x, y = f.update(100.0, 200.0, ts=0.0)
    assert (x, y) == (100.0, 200.0)


def test_kalman_smooths_jitter() -> None:
    f = GazeKalmanFilter(process_noise=0.001, measurement_noise=0.5)
    f.update(0.0, 0.0, ts=0.0)
    # Inject noisy samples around 100,100; smoothed output should pull toward it.
    out_x = out_y = 0.0
    for i, x in enumerate([95, 105, 92, 108, 100]):
        out_x, out_y = f.update(float(x), float(x), ts=0.05 * (i + 1))
    assert 80.0 < out_x < 120.0
    assert 80.0 < out_y < 120.0


def test_kalman_reset_clears_state() -> None:
    f = GazeKalmanFilter()
    f.update(50.0, 50.0, ts=0.0)
    f.reset()
    x, y = f.update(10.0, 10.0, ts=0.0)
    assert (x, y) == (10.0, 10.0)


def test_one_euro_first_sample_is_passthrough() -> None:
    f = OneEuroFilter()
    assert f(42.0, ts=0.0) == 42.0


def test_one_euro_smooths_constant_signal() -> None:
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    f(0.0, ts=0.0)
    out = f(100.0, ts=0.05)
    # min_cutoff=0.5 is heavy smoothing; output stays well below the new sample.
    assert 0.0 < out < 100.0
