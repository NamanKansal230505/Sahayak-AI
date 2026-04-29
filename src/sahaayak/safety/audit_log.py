"""Local-only, user-readable JSON-Lines audit log.

Each entry is one JSON object on its own line:

    {"ts": 1700000000.123, "kind": "click", "data": {...}}

Privacy invariants:

* Path is local — no URLs allowed (we assert ``not path.startswith("http")``).
* The log can be cleared from the UI; we never roll it off-device.
* No frame, no audio chunk, no full text body. Only short metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import RLock
from typing import Any


class AuditLog:
    """Thread-safe append-only JSONL writer.

    Args:
        path: Local file path (POSIX or Windows). HTTP URLs are rejected.
        max_bytes: Soft cap; once exceeded we trim the oldest 50% on next write.
    """

    def __init__(self, path: Path | str, max_bytes: int = 2_000_000) -> None:
        # Validate the original string before pathlib normalises slashes —
        # WindowsPath turns "https://..." into "https:\..." which would let
        # a URL slip past a post-Path startswith check.
        if str(path).lower().startswith(("http://", "https://", "ftp://")):
            raise ValueError("Audit log path must be local — URLs are forbidden.")
        self._path = Path(path)
        self._max_bytes = int(max_bytes)
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, kind: str, data: dict[str, Any] | None = None) -> None:
        """Append a single audit entry."""
        entry = {
            "ts": round(time.time(), 3),
            "kind": kind,
            "data": _scrub(data or {}),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._maybe_trim()
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read entries, newest first if limit is set."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-limit:]
        out: list[dict[str, Any]] = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    def clear(self) -> None:
        """Wipe the log (user-initiated)."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def _maybe_trim(self) -> None:
        if not self._path.exists():
            return
        if self._path.stat().st_size <= self._max_bytes:
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        keep = lines[len(lines) // 2 :]
        self._path.write_text("\n".join(keep) + "\n", encoding="utf-8")


_FORBIDDEN_KEYS = frozenset({"frame", "image", "iris_embedding", "audio", "transcript"})


def _scrub(data: dict[str, Any]) -> dict[str, Any]:
    """Drop any key that could contain bulk PII; cap remaining string lengths."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in _FORBIDDEN_KEYS:
            continue
        if isinstance(v, str) and len(v) > 240:
            out[k] = v[:240] + "..."
        else:
            out[k] = v
    return out
