"""Smoke tests for the eye-tracker module.

The full OpenVINO inference path requires real model files; we validate
the privacy invariant (no disk writes / no embedding persistence) by
grepping the source.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from sahaayak.core import eye_tracker as et


def test_module_does_not_persist_frames() -> None:
    src = Path(et.__file__).read_text(encoding="utf-8")
    forbidden = ("cv2.imwrite", ".write_bytes(", "PIL.Image.save", "imageio.imwrite")
    for needle in forbidden:
        assert needle not in src, f"eye_tracker.py must not call {needle}"


def test_eye_aspect_ratio_open_eye_above_threshold() -> None:
    # Synthetic 6-point open eye: roughly 2:1 horizontal:vertical = EAR ~0.5
    landmarks = np.array(
        [[0.0, 0.5], [0.3, 0.0], [0.7, 0.0], [1.0, 0.5], [0.7, 1.0], [0.3, 1.0]]
    )
    ear = et._eye_aspect_ratio(landmarks)
    assert ear > 0.4


def test_eye_aspect_ratio_closed_eye_low() -> None:
    landmarks = np.array(
        [[0.0, 0.5], [0.3, 0.49], [0.7, 0.49], [1.0, 0.5], [0.7, 0.51], [0.3, 0.51]]
    )
    ear = et._eye_aspect_ratio(landmarks)
    assert ear < 0.05


def test_eye_aspect_ratio_handles_short_input() -> None:
    assert et._eye_aspect_ratio(np.zeros((3, 2))) == 0.0


def test_class_signature_accepts_optional_models_dir() -> None:
    sig = inspect.signature(et.EyeTracker.__init__)
    assert "models_dir" in sig.parameters
    assert "device" in sig.parameters
