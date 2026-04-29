"""Tests for the 9-point calibrator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sahaayak.core.calibrator import (
    CalibrationProfile,
    Calibrator,
    _solve_homography,
    nine_point_grid,
)


def test_nine_point_grid_yields_nine_points() -> None:
    pts = nine_point_grid(1920, 1080)
    assert len(pts) == 9
    # Corners and centre present
    assert (1920 // 2, 1080 // 2) in pts


def test_solve_homography_identity_recovers_input() -> None:
    src = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float64)
    dst = src.copy()
    h = _solve_homography(src, dst)
    # Apply to a new point
    p = np.array([5.0, 5.0, 1.0])
    out = h @ p
    out = out / out[2]
    assert abs(out[0] - 5.0) < 1e-6
    assert abs(out[1] - 5.0) < 1e-6


def test_calibrator_collects_and_fits() -> None:
    cal = Calibrator(screen_size=(800, 600), config={"calibration": {"dwell_per_point_ms": 0}})
    # Push synthetic samples whose iris coords map cleanly to screen-space.
    for tx, ty in cal.targets:
        for _ in range(3):
            cal._samples.append(((tx / 2, ty / 2), (tx, ty)))  # noqa: SLF001
    profile = cal.fit()
    assert isinstance(profile, CalibrationProfile)
    assert profile.homography.shape == (3, 3)


def test_calibration_profile_save_and_load(tmp_path: Path) -> None:
    h = np.eye(3, dtype=np.float64)
    profile = CalibrationProfile(homography=h, screen_size=(1024, 768))
    target = tmp_path / "profile.yaml"
    profile.save(target)
    loaded = CalibrationProfile.load(target)
    assert loaded is not None
    assert loaded.screen_size == (1024, 768)
    assert np.allclose(loaded.homography, h)


def test_calibration_profile_load_missing_returns_none(tmp_path: Path) -> None:
    assert CalibrationProfile.load(tmp_path / "nope.yaml") is None
