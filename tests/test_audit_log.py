"""Tests for the local-only audit log."""

from __future__ import annotations

from pathlib import Path

import pytest

from sahaayak.safety.audit_log import AuditLog, _scrub


def test_audit_log_rejects_url_paths() -> None:
    with pytest.raises(ValueError):
        AuditLog("https://evil.example.com/audit.json")


def test_audit_log_appends_and_reads(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.json")
    log.append("session_start", {"version": "0.1.0"})
    log.append("gesture.short_blink")
    entries = log.read()
    assert len(entries) == 2
    assert entries[0]["kind"] == "session_start"
    assert entries[1]["kind"] == "gesture.short_blink"


def test_scrub_drops_forbidden_keys() -> None:
    out = _scrub({"frame": b"x", "iris_embedding": [1, 2, 3], "ok": "fine"})
    assert "frame" not in out
    assert "iris_embedding" not in out
    assert out["ok"] == "fine"


def test_scrub_truncates_long_strings() -> None:
    long = "a" * 500
    out = _scrub({"text": long})
    assert len(out["text"]) <= 245
    assert out["text"].endswith("...")


def test_audit_log_clear(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.json")
    log.append("a")
    log.clear()
    assert log.read() == []
