"""Iris-coordinate to screen-coordinate mapping with smoothing.

Two stages:

1. **Mapping**: combines both iris centres into a single normalised pupil
   vector and maps to screen coordinates using a 3x3 homography learned
   during calibration. Without calibration, falls back to a linear
   centre-and-scale heuristic so the dot still tracks roughly.
2. **Smoothing**: applies the Kalman filter from `core.kalman_filter` (or
   the 1-Euro fallback) configured in `config/default.yaml`.

Privacy invariant: the iris embedding (eyelid landmark vector) is
never persisted. Only the smoothed (x, y, confidence) tuple is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sahaayak.core.kalman_filter import GazeKalmanFilter, OneEuroFilter
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.core.eye_tracker import EyeTrackResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class GazePoint:
    """Smoothed gaze coordinate in screen pixels."""

    x: float
    y: float
    confidence: float
    timestamp: float


class GazeEstimator:
    """Maps eye-tracker output to a smoothed on-screen gaze point.

    Args:
        screen_size: (width, height) in pixels. Used for clipping and as the
            target space of the homography.
        config: Loaded SahaayakAI config.
        homography: Optional 3x3 ndarray from `Calibrator.fit`. When
            ``None`` we fall back to a centre-and-scale heuristic.
    """

    def __init__(
        self,
        screen_size: tuple[int, int],
        config: dict[str, Any] | None = None,
        homography: np.ndarray | None = None,
    ) -> None:
        self._screen_w, self._screen_h = screen_size
        self._config = config or {}
        gaze_cfg = self._config.get("gaze", {})
        self._smoothing = gaze_cfg.get("smoothing", "kalman")
        self._homography = homography
        if self._smoothing == "one_euro":
            self._filter_x = OneEuroFilter(
                min_cutoff=float(gaze_cfg.get("one_euro_min_cutoff", 1.0)),
                beta=float(gaze_cfg.get("one_euro_beta", 0.007)),
            )
            self._filter_y = OneEuroFilter(
                min_cutoff=float(gaze_cfg.get("one_euro_min_cutoff", 1.0)),
                beta=float(gaze_cfg.get("one_euro_beta", 0.007)),
            )
            self._kalman: GazeKalmanFilter | None = None
        else:
            self._kalman = GazeKalmanFilter(
                process_noise=float(gaze_cfg.get("kalman_process_noise", 0.01)),
                measurement_noise=float(gaze_cfg.get("kalman_measurement_noise", 0.05)),
            )
            self._filter_x = self._filter_y = None

    def set_homography(self, homography: np.ndarray | None) -> None:
        """Install (or remove) a calibration homography."""
        self._homography = homography
        logger.info("Calibration homography %s.", "installed" if homography is not None else "cleared")

    def estimate(self, result: EyeTrackResult) -> GazePoint:
        """Project an `EyeTrackResult` to a smoothed `GazePoint`.

        Args:
            result: Per-frame eye-tracker output.

        Returns:
            A smoothed gaze point in screen pixels (clipped to bounds).
        """
        # Combine both irises. Average is robust when one eye is occluded
        # (e.g., kohl/kajal user closes one eye).
        ix = (result.left_iris[0] + result.right_iris[0]) / 2.0
        iy = (result.left_iris[1] + result.right_iris[1]) / 2.0

        if self._homography is not None:
            mapped = self._apply_homography(ix, iy)
        else:
            mapped = self._linear_fallback(ix, iy)

        sx, sy = self._smooth(mapped[0], mapped[1], result.timestamp)
        sx = float(np.clip(sx, 0, self._screen_w - 1))
        sy = float(np.clip(sy, 0, self._screen_h - 1))
        return GazePoint(x=sx, y=sy, confidence=result.confidence, timestamp=result.timestamp)

    def _apply_homography(self, x: float, y: float) -> tuple[float, float]:
        vec = np.array([x, y, 1.0], dtype=np.float64)
        out = self._homography @ vec  # type: ignore[operator]
        if abs(out[2]) < 1e-9:
            return x, y
        return float(out[0] / out[2]), float(out[1] / out[2])

    def _linear_fallback(self, ix: float, iy: float) -> tuple[float, float]:
        cam_w = self._config.get("camera", {}).get("width", 640)
        cam_h = self._config.get("camera", {}).get("height", 480)
        return (
            float(ix / cam_w * self._screen_w),
            float(iy / cam_h * self._screen_h),
        )

    def _smooth(self, x: float, y: float, ts: float) -> tuple[float, float]:
        if self._kalman is not None:
            return self._kalman.update(x, y, ts)
        if self._filter_x is None or self._filter_y is None:
            return x, y
        return self._filter_x(x, ts), self._filter_y(y, ts)

    def reset(self) -> None:
        """Reset internal smoothing state — call after calibration."""
        if self._kalman is not None:
            self._kalman.reset()
        if self._filter_x is not None:
            self._filter_x.reset()
        if self._filter_y is not None:
            self._filter_y.reset()
