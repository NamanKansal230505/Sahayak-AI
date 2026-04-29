"""Structured logging for SahaayakAI.

All `src/` code MUST log via `get_logger(__name__)` — no `print()` calls.
Logs are written to stderr only; we never emit telemetry.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = (
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-32s  %(message)s"
)
_DATE_FORMAT: Final[str] = "%H:%M:%S"

_configured: bool = False


def _configure_root(level: int) -> None:
    """Attach a single stderr handler to the root logger (idempotent)."""
    global _configured  # noqa: PLW0603 — module-level latch is intentional
    if _configured:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: str, level: int | str = logging.INFO) -> logging.Logger:
    """Return a configured logger bound to the given module name.

    Args:
        name: Logger name; pass ``__name__`` from the calling module.
        level: Logging level (numeric or name). Applied on first call only.

    Returns:
        A `logging.Logger` writing to stderr in the SahaayakAI format.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if not isinstance(level, int):
        level = logging.INFO
    _configure_root(level)
    return logging.getLogger(name)
