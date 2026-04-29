"""PDF / text -> mind-map renderer.

LLM produces a JSON tree; `graphviz` renders the SVG. The text extractor
is intentionally minimal — for richer PDF parsing we will add `pypdf`
in a follow-up milestone (would require asking before adding deps).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sahaayak.copilot.llm_engine import load_prompt
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.copilot.llm_engine import LLMEngine

logger = get_logger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class MindMap:
    """Root + branches structure suitable for rendering."""

    root: str
    branches: list[dict[str, list[str]]] = field(default_factory=list)

    def to_graphviz_source(self) -> str:
        """Return a `graphviz` source string (no rendering)."""
        lines = [
            "digraph SahaayakMindMap {",
            "  rankdir=LR;",
            '  node [shape=box, style=rounded, fontname="Helvetica"];',
            f'  "root" [label="{_escape(self.root)}", style="rounded,bold"];',
        ]
        for i, branch in enumerate(self.branches):
            label = branch.get("label", f"branch{i}") if isinstance(branch, dict) else str(branch)
            bid = f"b{i}"
            lines.append(f'  "{bid}" [label="{_escape(str(label))}"];')
            lines.append(f'  "root" -> "{bid}";')
            children = branch.get("children", []) if isinstance(branch, dict) else []
            for j, child in enumerate(children or []):
                cid = f"{bid}_c{j}"
                lines.append(f'  "{cid}" [label="{_escape(str(child))}"];')
                lines.append(f'  "{bid}" -> "{cid}";')
        lines.append("}")
        return "\n".join(lines)


def _escape(s: str) -> str:
    return s.replace('"', '\\"').replace("\n", " ")


class DocVisualizer:
    """Turn a document (string) into a mind-map SVG."""

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine

    def to_mind_map(self, document: str) -> MindMap:
        """Generate a `MindMap` from arbitrary text."""
        prompt = load_prompt("mind_map").format(document=document.strip()[:4000])
        raw = self._engine.generate(prompt)
        return _parse_mind_map(raw)

    def render_svg(self, mind_map: MindMap, out_path: Path) -> Path:
        """Render the mind map to an SVG file via `graphviz`. Returns the path."""
        try:
            from graphviz import Source  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError("graphviz Python binding not installed.") from exc
        Source(mind_map.to_graphviz_source(), format="svg").render(
            out_path.with_suffix(""), cleanup=True
        )
        return out_path.with_suffix(".svg")


def _parse_mind_map(raw: str) -> MindMap:
    match = JSON_BLOCK_RE.search(raw)
    if not match:
        return MindMap(root="Document", branches=[])
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Mind-map JSON parse failed: %s", exc)
        return MindMap(root="Document", branches=[])
    branches_raw = data.get("branches", []) or []
    branches: list[dict[str, list[str]]] = []
    for b in branches_raw[:6]:
        if isinstance(b, dict):
            branches.append(
                {
                    "label": str(b.get("label", ""))[:80],
                    "children": [str(c)[:80] for c in (b.get("children") or [])][:5],
                }
            )
    return MindMap(root=str(data.get("root", "Document"))[:80], branches=branches)
