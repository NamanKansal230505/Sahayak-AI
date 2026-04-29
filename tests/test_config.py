"""Tests for the config loader and i18n table."""

from __future__ import annotations

from pathlib import Path

import yaml

from sahaayak.utils.config import load_config
from sahaayak.utils.i18n import get_language, i18n, set_language


def test_load_config_returns_default_keys() -> None:
    cfg = load_config()
    assert "app" in cfg
    assert "inference" in cfg
    assert "gestures" in cfg
    assert cfg["safety"]["kill_switch_hotkey"] == "F12"


def test_load_config_merges_user_profile(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    user = tmp_path / "user.yaml"
    default.write_text(yaml.safe_dump({"app": {"language": "en", "log_level": "INFO"}}), encoding="utf-8")
    user.write_text(yaml.safe_dump({"app": {"language": "hi"}}), encoding="utf-8")
    cfg = load_config(default_path=default, user_path=user)
    assert cfg["app"]["language"] == "hi"
    assert cfg["app"]["log_level"] == "INFO"


def test_i18n_returns_english_for_known_key() -> None:
    set_language("en")
    assert "AI-generated" in i18n("copilot.disclaimer")


def test_i18n_falls_back_to_key_when_missing() -> None:
    assert i18n("nonexistent.key") == "nonexistent.key"


def test_i18n_switches_language() -> None:
    set_language("hi")
    assert get_language() == "hi"
    assert i18n("copilot.tldr") == "संक्षेप में"
    set_language("en")  # reset for other tests
