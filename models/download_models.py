"""Download and convert the models SahaayakAI depends on.

Run this once after `pip install -r requirements.txt`. Network is required
ONLY for this step. All subsequent runtime calls are local.

Fetched assets (none committed to git):

1. **MediaPipe Iris** + **BlazeFace short-range** (TFLite from Google).
   Converted to OpenVINO IR (FP16) using the `ovc` CLI shipped with the
   `openvino` wheel. Two .xml/.bin pairs land in ``models/iris/``.

2. **Phi-3-mini-4k-instruct** (INT4 weight-only) via `optimum-cli export
   openvino --weight-format int4 --model microsoft/Phi-3-mini-4k-instruct`.
   Lands in ``models/phi-3-mini-int4-ov/``.

Usage:
    python models/download_models.py [--skip-llm] [--skip-iris]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
IRIS_DIR = MODELS_DIR / "iris"
LLM_DIR = MODELS_DIR / "phi-3-mini-int4-ov"

IRIS_TFLITE_URL = "https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite"
FACE_TFLITE_URL = "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite"

LLM_HF_ID = "microsoft/Phi-3-mini-4k-instruct"


def _http_download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return
    print(f"[download] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310 - HTTPS to known asset host
        shutil.copyfileobj(resp, fh)


def _ovc_convert(tflite_path: Path, output_dir: Path, model_name: str) -> None:
    """Run the OpenVINO Model Converter on a TFLite file (FP16)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"{model_name}.xml"
    if xml_path.exists():
        print(f"[skip] {xml_path.name} already converted")
        return
    print(f"[convert] {tflite_path.name} -> {xml_path.name}")
    cmd = [
        "ovc", str(tflite_path),
        "--output_model", str(output_dir / model_name),
        "--compress_to_fp16=True",
    ]
    subprocess.run(cmd, check=True)  # noqa: S603 - args composed from constants only


def download_iris() -> None:
    """Fetch MediaPipe Iris and BlazeFace, convert to OpenVINO IR."""
    IRIS_DIR.mkdir(parents=True, exist_ok=True)
    iris_tflite = IRIS_DIR / "iris_landmark.tflite"
    face_tflite = IRIS_DIR / "face_detector.tflite"
    _http_download(IRIS_TFLITE_URL, iris_tflite)
    _http_download(FACE_TFLITE_URL, face_tflite)
    _ovc_convert(iris_tflite, IRIS_DIR, "iris_landmark")
    _ovc_convert(face_tflite, IRIS_DIR, "face_detector")


def download_llm() -> None:
    """Pull Phi-3-mini and quantise to INT4 OpenVINO IR via `optimum-cli`."""
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    if (LLM_DIR / "openvino_model.xml").exists():
        print("[skip] Phi-3 INT4 IR already present")
        return
    print(f"[export] {LLM_HF_ID} -> {LLM_DIR}")
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", LLM_HF_ID,
        "--weight-format", "int4",
        "--task", "text-generation-with-past",
        str(LLM_DIR),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download SahaayakAI model assets.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Phi-3 download.")
    parser.add_argument("--skip-iris", action="store_true", help="Skip iris download.")
    args = parser.parse_args(argv)

    try:
        if not args.skip_iris:
            download_iris()
        if not args.skip_llm:
            download_llm()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Hint: ensure `ovc` (OpenVINO) and `optimum-cli` (pip install "
            "optimum[openvino,nncf]) are on PATH.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
