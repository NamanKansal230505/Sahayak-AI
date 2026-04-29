"""MediaPipe FaceMesh perception backend.

Drop-in alternative to `EyeTracker` (the OpenVINO + Haar pipeline). Emits
the same `EyeTrackResult` dataclass so downstream modules — `GazeEstimator`,
`GestureEngine`, `ActionDispatcher` — work unchanged.

**Why an alternative backend?** During real-device testing on a Core Ultra
the hand-rolled OpenCV-Haar + raw OpenVINO BlazeFace path produced
unstable face crops and frequent corner-spike cursor jumps. MediaPipe's
FaceMesh Solution wraps a face detector + 468-point landmark regressor +
iris-refinement model into one CPU-friendly pipeline that "just works".

This means MediaPipe runs its TFLite kernels on the CPU rather than going
through OpenVINO Runtime, **for the vision path only**. The OpenVINO
acceleration story for SahaayakAI lives in the Phi-3 LLM (`LLMEngine`),
which is by far the more compute-bound model. See
`docs/INTEL_STACK_RATIONALE.md` for the honest write-up of this trade-off.

Privacy invariants are unchanged:
    * Webcam frames live in RAM only.
    * Iris embeddings are not persisted; only transient (x, y).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sahaayak.core.eye_tracker import EyeTrackResult
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

# MediaPipe FaceMesh refined-landmark indices (478 total when refine_landmarks=True).
# Iris centres come from the 468-477 cluster; eyelids from the canonical eye contour.
RIGHT_IRIS_CENTRE = 473
LEFT_IRIS_CENTRE = 468
# 6-point eye-contour subset for the classical Soukupová-Čech EAR formula.
# Source: MediaPipe FaceMesh canonical map.
LEFT_EYE_6 = (33, 160, 158, 133, 153, 144)   # user's-left = image-left after mirror
RIGHT_EYE_6 = (362, 385, 387, 263, 373, 380)


def _ear_six_point(landmarks: np.ndarray) -> float:
    """Classic 6-point eye-aspect-ratio: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||).

    With MediaPipe pixel landmarks this returns ~0.30 open / <0.18 closed,
    making the engine's existing thresholds in `config/default.yaml` valid.
    """
    if landmarks.shape[0] < 6:
        return 0.0
    p1, p2, p3, p4, p5, p6 = landmarks[:6]
    horiz = float(np.linalg.norm(p1 - p4))
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


class MediaPipeTracker:
    """FaceMesh-backed eye tracker with the same interface as `EyeTracker`.

    Args:
        config: Loaded SahaayakAI config dict.
        device: Ignored (MediaPipe runs on CPU). Accepted for API parity.
        models_dir: Ignored. Accepted for API parity.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | None = None,
        models_dir: Path | None = None,
    ) -> None:
        self._config = config or {}
        self._device = device or "CPU (MediaPipe TFLite)"
        try:
            import mediapipe as mp  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "mediapipe not installed. Run `pip install -r requirements.txt`."
            ) from exc
        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._cv: Any = None
        logger.info("MediaPipeTracker initialised (vision on %s)", self._device)

    @property
    def device(self) -> str:
        """API-compat with EyeTracker.device — the CPU MediaPipe backend label."""
        return self._device

    def _require_cv2(self) -> Any:
        if self._cv is not None:
            return self._cv
        try:
            import cv2  # noqa: PLC0415
            self._cv = cv2
            return cv2
        except ImportError as exc:
            raise ImportError("opencv-python not installed.") from exc

    def process(self, frame: np.ndarray) -> EyeTrackResult | None:
        """Run a single BGR frame through MediaPipe FaceMesh."""
        cv = self._require_cv2()
        h, w = frame.shape[:2]
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = self._mesh.process(rgb)
        del rgb
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark

        def px(idx: int) -> tuple[float, float]:
            return (lm[idx].x * w, lm[idx].y * h)

        left_iris = px(LEFT_IRIS_CENTRE)
        right_iris = px(RIGHT_IRIS_CENTRE)
        left_pts = np.asarray([px(i) for i in LEFT_EYE_6], dtype=np.float32)
        right_pts = np.asarray([px(i) for i in RIGHT_EYE_6], dtype=np.float32)
        left_ear = _ear_six_point(left_pts)
        right_ear = _ear_six_point(right_pts)
        # Confidence: clamped mean EAR / open-eye reference (0.30 open).
        confidence = float(np.clip((left_ear + right_ear) / 0.6, 0.0, 1.0))

        return EyeTrackResult(
            left_iris=left_iris,
            right_iris=right_iris,
            left_eye_landmarks=left_pts,
            right_eye_landmarks=right_pts,
            confidence=confidence,
            timestamp=time.monotonic(),
            left_ear=left_ear,
            right_ear=right_ear,
        )

    def stream(self, source: int = 0) -> Iterator[tuple[np.ndarray, EyeTrackResult | None]]:
        """Yield (mirrored frame, result) tuples from a webcam.

        Mirroring matches the user's spatial intuition (move head left -> cursor left).
        Frames must NOT be persisted by the caller (privacy invariant).
        """
        cv = self._require_cv2()
        cam_cfg = self._config.get("camera", {})
        cap = cv.VideoCapture(int(cam_cfg.get("index", source)))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, int(cam_cfg.get("width", 640)))
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(cam_cfg.get("height", 480)))
        cap.set(cv.CAP_PROP_FPS, int(cam_cfg.get("fps", 30)))
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv.flip(frame, 1)
                yield frame, self.process(frame)
        finally:
            cap.release()

    def close(self) -> None:
        """Release the FaceMesh graph."""
        try:
            self._mesh.close()
        except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
            logger.debug("FaceMesh.close() raised: %s", exc)
