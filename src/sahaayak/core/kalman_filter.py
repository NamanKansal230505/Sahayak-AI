"""Smoothing filters for jittery gaze coordinates.

Two implementations live here:

* `GazeKalmanFilter` — a 2D constant-velocity Kalman filter (state =
  [x, y, vx, vy]). Tracks slow head motion well; can overshoot on fast
  saccades.
* `OneEuroFilter` — adaptive low-pass; the fallback when Kalman overshoots,
  per the spec. Cheap, stateless except for the previous sample.

Pure NumPy — no `filterpy` dependency at the smoothing path so the module
can be used in tests without optional deps. `filterpy` is still listed in
`requirements.txt` for advanced users who want to swap in higher-order
filters.
"""

from __future__ import annotations

import math
import time

import numpy as np


class GazeKalmanFilter:
    """Constant-velocity Kalman filter for 2D gaze coordinates.

    Args:
        process_noise: Variance of the process noise (acceleration).
        measurement_noise: Variance of the measurement noise.
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.05) -> None:
        self._q = float(process_noise)
        self._r = float(measurement_noise)
        self._x = np.zeros((4, 1), dtype=np.float64)  # [x, y, vx, vy]
        self._p = np.eye(4, dtype=np.float64) * 1.0
        self._h = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        self._initialised = False
        self._last_ts: float | None = None

    def reset(self) -> None:
        """Forget all state — call after calibration or large jumps."""
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._p = np.eye(4, dtype=np.float64) * 1.0
        self._initialised = False
        self._last_ts = None

    def update(self, x: float, y: float, ts: float | None = None) -> tuple[float, float]:
        """Step the filter with a new measurement and return the smoothed point."""
        ts = float(ts) if ts is not None else time.monotonic()
        if not self._initialised:
            self._x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
            self._initialised = True
            self._last_ts = ts
            return x, y

        dt = max(ts - (self._last_ts or ts), 1e-3)
        self._last_ts = ts
        f_mat = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q_mat = np.eye(4, dtype=np.float64) * self._q
        r_mat = np.eye(2, dtype=np.float64) * self._r

        # Predict
        self._x = f_mat @ self._x
        self._p = f_mat @ self._p @ f_mat.T + q_mat

        # Update
        z = np.array([[x], [y]], dtype=np.float64)
        y_resid = z - self._h @ self._x
        s = self._h @ self._p @ self._h.T + r_mat
        k = self._p @ self._h.T @ np.linalg.inv(s)
        self._x = self._x + k @ y_resid
        self._p = (np.eye(4) - k @ self._h) @ self._p

        return float(self._x[0, 0]), float(self._x[1, 0])


class OneEuroFilter:
    """Single-axis 1-Euro filter — adaptive low-pass for noisy signals."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Forget previous sample."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, ts: float | None = None) -> float:
        """Step the filter and return the smoothed value."""
        ts = float(ts) if ts is not None else time.monotonic()
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = ts
            return x
        dt = max(ts - self._t_prev, 1e-3)
        self._t_prev = ts
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat
