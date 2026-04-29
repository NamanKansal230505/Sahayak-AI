"""Tests for the MediaPipe perception backend.

We avoid loading the actual MediaPipe model in unit tests (it pulls TFLite
weights into RAM and slows the suite). Instead we exercise the helper
functions and the dataclass-shape contract.
"""

from __future__ import annotations

import inspect

import numpy as np

from sahaayak.core import mediapipe_tracker as mpt


def test_module_does_not_persist_frames() -> None:
    """Privacy invariant: never write a webcam frame to disk."""
    src = inspect.getsource(mpt)
    forbidden = (".write_bytes(", "imwrite(", "Image.save", "imageio.imwrite")
    for needle in forbidden:
        assert needle not in src, f"mediapipe_tracker.py must not call {needle}"


def test_ear_six_point_open_eye() -> None:
    # Synthetic 6-point eye in pixel coords ~ canonical EAR layout.
    pts = np.array(
        [[0, 5], [3, 0], [7, 0], [10, 5], [7, 10], [3, 10]], dtype=np.float32
    )
    ear = mpt._ear_six_point(pts)
    assert ear > 0.4


def test_ear_six_point_closed_eye() -> None:
    pts = np.array(
        [[0, 5], [3, 4.95], [7, 4.95], [10, 5], [7, 5.05], [3, 5.05]], dtype=np.float32
    )
    ear = mpt._ear_six_point(pts)
    assert ear < 0.05


def test_ear_six_point_handles_short_input() -> None:
    assert mpt._ear_six_point(np.zeros((3, 2))) == 0.0


def test_landmark_index_constants_exist() -> None:
    # Sanity: the canonical MediaPipe FaceMesh refined-landmark indices.
    assert mpt.RIGHT_IRIS_CENTRE == 473
    assert mpt.LEFT_IRIS_CENTRE == 468
    assert len(mpt.LEFT_EYE_6) == 6
    assert len(mpt.RIGHT_EYE_6) == 6
