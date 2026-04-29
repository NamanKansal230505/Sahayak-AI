"""Smoke tests for the CLI entry point."""

from __future__ import annotations

from sahaayak.main import main


def test_check_subcommand_runs_and_returns_int(capsys) -> None:
    rc = main(["--check"])
    out = capsys.readouterr().out
    assert isinstance(rc, int)
    assert "SahaayakAI" in out
    assert "OpenVINO runtime" in out


def test_no_args_prints_help(capsys) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage" in out.lower()
