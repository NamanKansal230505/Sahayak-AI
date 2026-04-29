"""Mic input -> transcript -> Phi-3 bullet summary.

We capture mic audio with `sounddevice`, hand chunks to `faster-whisper`
for STT (which itself runs CTranslate2 on CPU; OpenVINO-Whisper drop-in
is a follow-up), and pass the final transcript through `LLMEngine`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sahaayak.copilot.llm_engine import load_prompt
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.copilot.llm_engine import LLMEngine

logger = get_logger(__name__)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class MeetingActionItem:
    owner: str | None
    task: str
    deadline: str | None = None


@dataclass
class MeetingSummary:
    summary: str
    decisions: list[str] = field(default_factory=list)
    action_items: list[MeetingActionItem] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


class MeetingSummarizer:
    """Transcribe + summarise an offline meeting."""

    def __init__(self, engine: LLMEngine, whisper_model: str = "small") -> None:
        self._engine = engine
        self._whisper_model = whisper_model
        self._whisper: object | None = None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a WAV file path. Returns plain-text transcript."""
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faster-whisper not installed. Run `pip install -r requirements.txt`."
            ) from exc
        if self._whisper is None:
            logger.info("Loading Whisper model: %s", self._whisper_model)
            self._whisper = WhisperModel(
                self._whisper_model, device="cpu", compute_type="int8"
            )
        segments, _info = self._whisper.transcribe(audio_path)  # type: ignore[union-attr]
        return " ".join(seg.text.strip() for seg in segments).strip()

    def summarise(self, transcript: str) -> MeetingSummary:
        """Run Phi-3 on a transcript and return a structured summary."""
        prompt = load_prompt("meeting_summary").format(transcript=transcript.strip()[:6000])
        raw = self._engine.generate(prompt)
        return _parse_summary(raw)


def _parse_summary(raw: str) -> MeetingSummary:
    match = JSON_BLOCK_RE.search(raw)
    if not match:
        return MeetingSummary(summary=raw[:240])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return MeetingSummary(summary=raw[:240])
    items: list[MeetingActionItem] = []
    for item in data.get("action_items", []) or []:
        if isinstance(item, dict):
            items.append(
                MeetingActionItem(
                    owner=item.get("owner"),
                    task=str(item.get("task", "")),
                    deadline=item.get("deadline"),
                )
            )
    return MeetingSummary(
        summary=str(data.get("summary", "")),
        decisions=[str(d) for d in (data.get("decisions") or [])],
        action_items=items,
        open_questions=[str(q) for q in (data.get("open_questions") or [])],
    )
