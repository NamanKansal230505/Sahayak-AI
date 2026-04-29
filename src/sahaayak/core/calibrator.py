"""9-point gaze calibration.

The user looks at each of 9 fixed screen points for ``dwell_per_point_ms``
ms. For each point we collect a window of iris-centre samples, take the
median (robust to blinks), and fit a 3x3 homography from iris-space to
screen-space.

Output: a per-user `CalibrationProfile` written to
``config/calibration_profile.yaml`` (gitignored). Re-calibration triggers
when gaze confidence drops below threshold for >5s — wired in main.py.

Privacy invariant: only the homography matrix and the screen size are
persisted. No iris embedding, no eyelid landmark vector, no frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from sahaayak.utils.config import USER_PROFILE_PATH
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.core.eye_tracker import EyeTracker, EyeTrackResult

logger = get_logger(__name__)


def nine_point_grid(width: int, height: int, margin_pct: float = 0.08) -> list[tuple[int, int]]:
    """Return the canonical 9-point calibration grid in screen pixels."""
    mx = int(width * margin_pct)
    my = int(height * margin_pct)
    cx = (mx, width // 2, width - mx)
    cy = (my, height // 2, height - my)
    return [(x, y) for y in cy for x in cx]


@dataclass
class CalibrationProfile:
    """Persisted per-user calibration."""

    homography: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    screen_size: tuple[int, int] = (1920, 1080)
    monocular: bool = False
    samples_per_point: int = 0

    def save(self, path: Path | None = None) -> Path:
        """Write the profile to YAML; returns the path written."""
        target = path or USER_PROFILE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "calibration": {
                "homography": self.homography.tolist(),
                "screen_size": list(self.screen_size),
                "monocular": self.monocular,
                "samples_per_point": self.samples_per_point,
            }
        }
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        logger.info("Calibration profile saved to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> CalibrationProfile | None:
        """Load a saved profile, or return ``None`` if absent/invalid."""
        target = path or USER_PROFILE_PATH
        if not target.exists():
            return None
        try:
            with target.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cal = data.get("calibration", {})
            return cls(
                homography=np.asarray(cal["homography"], dtype=np.float64),
                screen_size=tuple(cal.get("screen_size", (1920, 1080))),
                monocular=bool(cal.get("monocular", False)),
                samples_per_point=int(cal.get("samples_per_point", 0)),
            )
        except (KeyError, ValueError, yaml.YAMLError) as exc:
            logger.warning("Could not load calibration profile %s: %s", target, exc)
            return None


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve a 3x3 homography mapping `src` (Nx2) to `dst` (Nx2) via DLT.

    Args:
        src: Source points in iris-space (Nx2). N must be >= 4.
        dst: Destination points in screen-space (Nx2).

    Returns:
        3x3 homography matrix.

    Raises:
        ValueError: if fewer than 4 point pairs are given or solve fails.
    """
    if src.shape[0] < 4 or dst.shape[0] != src.shape[0]:
        raise ValueError("Need at least 4 matched point pairs for a homography.")
    n = src.shape[0]
    a_rows = []
    for i in range(n):
        x, y = src[i]
        u, v = dst[i]
        a_rows.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        a_rows.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    a = np.asarray(a_rows, dtype=np.float64)
    _u, _s, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def _iris_vector(result: EyeTrackResult) -> tuple[float, float]:
    return (
        (result.left_iris[0] + result.right_iris[0]) / 2.0,
        (result.left_iris[1] + result.right_iris[1]) / 2.0,
    )


class Calibrator:
    """Drives the calibration data-collection routine.

    The UI layer (`ui.calibration_window`) calls `collect_for_target` for
    each of the 9 grid points. When done, it calls `fit` to produce a
    `CalibrationProfile`.
    """

    def __init__(
        self,
        screen_size: tuple[int, int],
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = (config or {}).get("calibration", {})
        self._screen_size = screen_size
        self._dwell_ms = int(cfg.get("dwell_per_point_ms", 1500))
        self._min_samples = 5
        self._monocular_auto = bool(cfg.get("monocular_auto_detect", True))
        self._samples: list[tuple[tuple[float, float], tuple[int, int]]] = []
        self._monocular = False

    @property
    def targets(self) -> list[tuple[int, int]]:
        return nine_point_grid(*self._screen_size)

    @property
    def monocular(self) -> bool:
        return self._monocular

    def collect_for_target(
        self,
        target: tuple[int, int],
        sample_source: Callable[[], EyeTrackResult | None],
        clock: Callable[[], float] | None = None,
    ) -> int:
        """Pull samples for the given target until the dwell window expires.

        Args:
            target: Screen-space (x, y) the user is looking at.
            sample_source: Zero-arg callable yielding the latest tracker result.
            clock: Optional clock; defaults to ``time.monotonic``. Tests
                inject a fake clock to make the loop deterministic.

        Returns:
            Number of accepted samples for this target.
        """
        import time as _time  # noqa: PLC0415

        clock = clock or _time.monotonic
        deadline = clock() + self._dwell_ms / 1000.0
        accepted = 0
        while clock() < deadline:
            result = sample_source()
            if result is None or result.confidence < 0.4:
                continue
            iris_xy = _iris_vector(result)
            self._samples.append((iris_xy, target))
            accepted += 1
            # Detect monocular use: one eye consistently at (0,0) means
            # the iris model returned default for an occluded eye.
            if (
                self._monocular_auto
                and (result.left_iris == (0.0, 0.0) or result.right_iris == (0.0, 0.0))
            ):
                self._monocular = True
        return accepted

    def fit(self) -> CalibrationProfile:
        """Fit a homography over all collected samples."""
        if len(self._samples) < 4:
            raise RuntimeError("Not enough calibration samples — collected fewer than 4.")
        # Aggregate per-target via median for robustness.
        by_target: dict[tuple[int, int], list[tuple[float, float]]] = {}
        for iris_xy, target in self._samples:
            by_target.setdefault(target, []).append(iris_xy)
        srcs, dsts = [], []
        for target, iris_list in by_target.items():
            arr = np.asarray(iris_list, dtype=np.float64)
            srcs.append(np.median(arr, axis=0))
            dsts.append(np.asarray(target, dtype=np.float64))
        src = np.vstack(srcs)
        dst = np.vstack(dsts)
        h = _solve_homography(src, dst)
        per_point = max(1, len(self._samples) // max(1, len(by_target)))
        return CalibrationProfile(
            homography=h,
            screen_size=self._screen_size,
            monocular=self._monocular,
            samples_per_point=per_point,
        )

    def reset(self) -> None:
        """Discard collected samples (e.g., user cancelled mid-routine)."""
        self._samples.clear()
        self._monocular = False


def run_headless_calibration(
    tracker: EyeTracker,
    screen_size: tuple[int, int],
    config: dict[str, Any] | None = None,
    max_seconds: float = 60.0,
) -> CalibrationProfile:
    """Headless 9-point calibration loop — useful in tests and for CLI use.

    The PyQt6 UI version lives in `ui.calibration_window`.
    """
    import time as _time  # noqa: PLC0415

    cap = tracker.stream()
    cal = Calibrator(screen_size, config)
    deadline = _time.monotonic() + max_seconds
    for target in cal.targets:
        if _time.monotonic() > deadline:
            raise TimeoutError("Headless calibration ran past its budget.")
        cal.collect_for_target(target, lambda: next(cap)[1])
    return cal.fit()
