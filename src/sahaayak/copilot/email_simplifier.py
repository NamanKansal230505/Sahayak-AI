"""Long email -> TL;DR + key points + reply drafts.

Returns a `SimplifiedEmail` dataclass. NEVER auto-sends — the UI must show
the drafts to the user, who confirms via long-blink before any keystroke
is sent to the email client.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sahaayak.copilot.llm_engine import load_prompt
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.copilot.llm_engine import LLMEngine

logger = get_logger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ActionItem:
    """One concrete task extracted from the email."""

    task: str
    deadline: str | None = None


@dataclass
class SimplifiedEmail:
    """Output of `EmailSimplifier.simplify`."""

    tldr: str
    key_points: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    tone: str = "neutral"
    suggested_reply_drafts: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tldr": self.tldr,
            "key_points": list(self.key_points),
            "action_items": [{"task": a.task, "deadline": a.deadline} for a in self.action_items],
            "tone": self.tone,
            "suggested_reply_drafts": list(self.suggested_reply_drafts),
        }


class EmailSimplifier:
    """Wraps `LLMEngine` to produce a structured email summary."""

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine

    def simplify(self, email: str) -> SimplifiedEmail:
        """Run the LLM and parse the JSON response.

        Args:
            email: The raw email body.

        Returns:
            A populated `SimplifiedEmail`. Falls back to a minimal one
            containing the TL;DR-as-plain-text if JSON parsing fails.
        """
        prompt = load_prompt("email_simplify").format(email=email.strip())
        raw = self._engine.generate(prompt)
        return parse_response(raw)


def parse_response(raw: str) -> SimplifiedEmail:
    """Parse Phi-3 output. Tolerant of stray prose around the JSON object."""
    match = JSON_BLOCK_RE.search(raw)
    if not match:
        logger.warning("LLM response contained no JSON block; returning raw text.")
        return SimplifiedEmail(tldr=raw[:240], raw_response=raw)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse LLM JSON (%s); returning raw text.", exc)
        return SimplifiedEmail(tldr=raw[:240], raw_response=raw)

    items = []
    for item in data.get("action_items", []) or []:
        if isinstance(item, dict):
            items.append(
                ActionItem(
                    task=str(item.get("task", "")),
                    deadline=item.get("deadline") or None,
                )
            )
        elif isinstance(item, str):
            items.append(ActionItem(task=item))

    return SimplifiedEmail(
        tldr=str(data.get("tldr", "")),
        key_points=[str(p) for p in (data.get("key_points") or [])][:5],
        action_items=items,
        tone=str(data.get("tone", "neutral")),
        suggested_reply_drafts=[str(d) for d in (data.get("suggested_reply_drafts") or [])][:3],
        raw_response=raw,
    )
