"""One-shot Whisper transcription helper.

Runs in a fresh Python subprocess so we get a clean DLL state — avoids the
"dynamic link library initialization routine failed" Windows quirk that
hits when faster-whisper / ctranslate2 are loaded into a process that
already has MediaPipe + PyQt6 + torch sitting in memory.

Usage:
    python scripts/whisper_transcribe.py <wav_path> [<lang>]
        -> prints the transcribed text to stdout (or empty string)
        <lang> defaults to "en". Pass "auto" for Whisper auto-detection
        (warning: on short or quiet clips it often picks wrong language).

Exit codes:
    0   success (transcript on stdout)
    1   transcription failed (error on stderr)
    2   bad arguments
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force single-thread inside ctranslate2 + suppress torch framework probe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
# Force UTF-8 stdout so non-ASCII transcripts don't mojibake on cp1252.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print("usage: whisper_transcribe.py <wav_path> [<lang>]", file=sys.stderr)
        return 2
    wav_path = Path(sys.argv[1])
    if not wav_path.exists():
        print(f"file not found: {wav_path}", file=sys.stderr)
        return 1
    lang_arg = sys.argv[2] if len(sys.argv) == 3 else "en"
    forced_language: str | None = None if lang_arg == "auto" else lang_arg

    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError as exc:
        print(f"faster_whisper missing: {exc}", file=sys.stderr)
        return 1

    try:
        model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )
        # Forcing language="en" (or whatever was passed) prevents Whisper
        # from guessing Russian / Welsh / etc. on short or quiet clips,
        # which is the most common dictation failure mode.
        segments, _info = model.transcribe(
            str(wav_path),
            language=forced_language,
            vad_filter=True,           # drops silent leading/trailing audio
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback  # noqa: PLC0415
        traceback.print_exc(file=sys.stderr)
        print(f"\ntranscribe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
