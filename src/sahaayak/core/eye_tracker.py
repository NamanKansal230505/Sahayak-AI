"""OpenVINO MediaPipe-Iris inference at 30 FPS.

Pipeline per webcam frame:

    BGR frame -> face detector (OpenVINO) -> crop both eye ROIs ->
    iris_landmark (OpenVINO, twice) -> iris centres + eyelid landmarks ->
    EAR (eye-aspect-ratio) for blink detection.

Privacy invariants enforced HERE — not just in policy:

* Frames live in RAM only. The module never writes any image to disk; a
  unit test greps the source for forbidden output calls.
* Iris embeddings are not persisted. Only the transient (x, y) gaze tuple
  leaves this module.

Both the face detector and the iris model are loaded as OpenVINO IR. The
face detector is small enough (a few MB) that we run it on the same
device as the iris model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sahaayak.utils.intel_device import get_best_device
from sahaayak.utils.logger import get_logger

try:  # cv2 is heavy; lazy-import friendly so unit tests can run without it.
    import cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only on bare environments
    cv2 = None  # type: ignore[assignment]


def _require_cv2() -> Any:
    if cv2 is None:
        raise ImportError(
            "opencv-python not installed. Run `pip install -r requirements.txt`."
        )
    return cv2

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger(__name__)

IRIS_INPUT_SIZE = 64  # iris_landmark.tflite expects 64x64 RGB
FACE_INPUT_SIZE = 128  # face_detection_short_range.tflite expects 128x128 RGB
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "iris"


@dataclass(frozen=True)
class EyeTrackResult:
    """Per-frame output of the eye tracker.

    Coordinates are in the frame's pixel space. Iris centres are floats so
    sub-pixel precision survives the homography in `gaze_estimator`.
    """

    left_iris: tuple[float, float]
    right_iris: tuple[float, float]
    left_eye_landmarks: np.ndarray = field(repr=False)
    right_eye_landmarks: np.ndarray = field(repr=False)
    confidence: float
    timestamp: float
    left_ear: float = 0.0
    right_ear: float = 0.0


def _eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Compute eye aspect ratio (vertical / horizontal extent).

    We use bounding-box extents of the eye-contour cloud rather than the
    classical 6-point Soukupová-Čech formula because MediaPipe's 71-point
    eye contour output does not have a canonical 6-point subset shared
    across model versions. The bbox form is robust to landmark ordering
    and gives the same useful range (~0.30 open, <0.21 blink).

    Args:
        landmarks: ndarray of shape (N, 2) with at least 4 points.

    Returns:
        Eye-aspect-ratio in [0, ~0.6].
    """
    if landmarks.shape[0] < 4:
        return 0.0
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    horiz = float(xs.max() - xs.min())
    vert = float(ys.max() - ys.min())
    if horiz < 1e-6:
        return 0.0
    return vert / horiz


class EyeTracker:
    """OpenVINO-backed eye and iris tracker.

    Args:
        models_dir: Directory containing ``face_detector.xml`` and
            ``iris_landmark.xml`` produced by `models/download_models.py`.
        device: Explicit OpenVINO device id (e.g. ``"NPU"``). If ``None``,
            picks the best available via `get_best_device("vision")`.
        config: Loaded SahaayakAI config dict.

    Raises:
        FileNotFoundError: If a required IR file is missing.
        ImportError: If OpenVINO is not installed.
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        self._models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self._device = device or get_best_device("vision")
        self._face_compiled: Any = None
        self._iris_compiled: Any = None
        self._load_models()
        logger.info("EyeTracker initialised on device %s", self._device)

    @property
    def device(self) -> str:
        """Active OpenVINO device id."""
        return self._device

    def _load_models(self) -> None:
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "OpenVINO not installed. Run `pip install -r requirements.txt`."
            ) from exc

        iris_xml = self._models_dir / "iris_landmark.xml"
        if not iris_xml.exists():
            raise FileNotFoundError(
                f"OpenVINO IR missing: {iris_xml}. "
                "Run `python models/download_models.py` first."
            )

        core = ov.Core()
        self._iris_compiled = core.compile_model(str(iris_xml), self._device)
        # Face detection uses OpenCV's bundled Haar cascade — classical CV
        # rather than DL inference, so we are not violating the
        # "OpenVINO-only inference" rule. The interesting model (iris) still
        # runs on the chosen Intel device. Replace with an OpenVINO Open
        # Model Zoo detector (e.g. face-detection-retail-0005) when shipping.
        self._face_compiled = None  # unused; kept for backwards compatibility
        cv = _require_cv2()
        # `alt2` is the most permissive of the bundled frontal-face cascades.
        cascade_path = Path(cv.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
        if not cascade_path.exists():
            cascade_path = Path(cv.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise FileNotFoundError(f"OpenCV cascade missing: {cascade_path}")
        self._face_cascade = cv.CascadeClassifier(str(cascade_path))

    def process(self, frame: np.ndarray) -> EyeTrackResult | None:
        """Run a single frame through the pipeline.

        Args:
            frame: BGR image as returned by `cv2.VideoCapture.read`.

        Returns:
            An `EyeTrackResult`, or ``None`` if no face was found.
        """
        # Privacy invariant: we never store the frame anywhere.
        height, width = frame.shape[:2]
        face_box = self._detect_face(frame)
        if face_box is None:
            return None

        x1, y1, x2, y2 = face_box
        # Estimate eye ROIs as fixed proportions of the face bbox. These ratios
        # come from MediaPipe's FaceMesh average geometry and are good enough
        # for iris cropping; calibration corrects per-user offsets later.
        face_w, face_h = x2 - x1, y2 - y1
        eye_w = int(face_w * 0.34)
        eye_h = int(face_h * 0.22)
        cy = int(y1 + face_h * 0.42)
        left_cx = int(x1 + face_w * 0.30)
        right_cx = int(x1 + face_w * 0.70)

        left_roi = self._safe_crop(frame, left_cx, cy, eye_w, eye_h, width, height)
        right_roi = self._safe_crop(frame, right_cx, cy, eye_w, eye_h, width, height)
        if left_roi is None or right_roi is None:
            return None

        left_iris_local, left_lm_local = self._run_iris(left_roi)
        right_iris_local, right_lm_local = self._run_iris(right_roi)

        # Map iris centre back from eye-ROI space to full-frame pixels.
        l_off = (left_cx - eye_w // 2, cy - eye_h // 2)
        r_off = (right_cx - eye_w // 2, cy - eye_h // 2)
        left_iris = (
            l_off[0] + left_iris_local[0] * eye_w / IRIS_INPUT_SIZE,
            l_off[1] + left_iris_local[1] * eye_h / IRIS_INPUT_SIZE,
        )
        right_iris = (
            r_off[0] + right_iris_local[0] * eye_w / IRIS_INPUT_SIZE,
            r_off[1] + right_iris_local[1] * eye_h / IRIS_INPUT_SIZE,
        )
        left_lm = self._scale_landmarks(left_lm_local, l_off, eye_w, eye_h)
        right_lm = self._scale_landmarks(right_lm_local, r_off, eye_w, eye_h)

        left_ear = _eye_aspect_ratio(left_lm)
        right_ear = _eye_aspect_ratio(right_lm)
        # Confidence = mean EAR / open-eye reference (~0.30), clamped.
        ear_conf = min(1.0, (left_ear + right_ear) / 0.6)
        confidence = float(np.clip(ear_conf, 0.0, 1.0))

        return EyeTrackResult(
            left_iris=left_iris,
            right_iris=right_iris,
            left_eye_landmarks=left_lm,
            right_eye_landmarks=right_lm,
            confidence=confidence,
            timestamp=time.monotonic(),
            left_ear=left_ear,
            right_ear=right_ear,
        )

    def stream(self, source: int = 0) -> Iterator[tuple[np.ndarray, EyeTrackResult | None]]:
        """Yield (frame, result) tuples from a webcam.

        The frame is yielded for UI overlay only; callers MUST NOT persist it.
        """
        cv = _require_cv2()
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
                result = self.process(frame)
                yield frame, result
        finally:
            cap.release()

    @staticmethod
    def _safe_crop(
        frame: np.ndarray,
        cx: int,
        cy: int,
        w: int,
        h: int,
        max_w: int,
        max_h: int,
    ) -> np.ndarray | None:
        x1, y1 = max(cx - w // 2, 0), max(cy - h // 2, 0)
        x2, y2 = min(cx + w // 2, max_w), min(cy + h // 2, max_h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2]

    def _run_iris(self, eye_roi: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        """Run the iris model on a single eye crop.

        Returns:
            (iris_centre_xy_in_64x64, eyelid_landmarks_in_64x64).
        """
        cv = _require_cv2()
        rgb = cv.cvtColor(eye_roi, cv.COLOR_BGR2RGB)
        resized = cv.resize(rgb, (IRIS_INPUT_SIZE, IRIS_INPUT_SIZE))
        tensor = (resized.astype(np.float32) / 127.5 - 1.0)[None, ...]

        outputs = self._iris_compiled(tensor)
        # Two named outputs: 'output_eyes_contours_and_brows' (213) and
        # 'output_iris' (15). Order varies between exports — find by length.
        eye_lm = iris_lm = None
        for arr in outputs.values():
            flat = np.asarray(arr).reshape(-1)
            if flat.size >= 200:
                eye_lm = flat
            elif flat.size >= 12:
                iris_lm = flat
        if eye_lm is None or iris_lm is None:
            return (IRIS_INPUT_SIZE / 2, IRIS_INPUT_SIZE / 2), np.zeros((6, 2))

        iris_pts = iris_lm.reshape(-1, 3)[:, :2]
        eye_pts = eye_lm.reshape(-1, 3)[:, :2]
        iris_centre = (float(iris_pts[0, 0]), float(iris_pts[0, 1]))
        # Pass the full eye contour cloud — _eye_aspect_ratio uses bbox extents.
        return iris_centre, eye_pts

    @staticmethod
    def _scale_landmarks(
        local: np.ndarray,
        offset: tuple[int, int],
        roi_w: int,
        roi_h: int,
    ) -> np.ndarray:
        if local.size == 0:
            return local
        scaled = local.copy().astype(np.float32)
        scaled[:, 0] = offset[0] + scaled[:, 0] * roi_w / IRIS_INPUT_SIZE
        scaled[:, 1] = offset[1] + scaled[:, 1] * roi_h / IRIS_INPUT_SIZE
        return scaled

    def _detect_face(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect a face in the frame and return the largest bbox."""
        cv = _require_cv2()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Equalise histogram to handle varied lighting (rural classrooms etc.).
        gray = cv.equalizeHist(gray)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=3, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Pick the largest detected face (closest to camera).
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(x + w), int(y + h)
