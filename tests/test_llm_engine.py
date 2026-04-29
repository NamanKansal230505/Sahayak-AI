"""Tests for the LLM engine wrapper.

We do not have Phi-3 IR available in this environment, so we exercise
prompt loading and the timeout-error path. The actual generation path is
covered by `test_email_simplifier.py` against a stubbed LLM in CI.
"""

from __future__ import annotations

import pytest

from sahaayak.copilot.llm_engine import LLMEngine, LLMTimeoutError, load_prompt


def test_load_prompt_returns_template() -> None:
    text = load_prompt("email_simplify")
    assert "{email}" in text
    assert "JSON" in text


def test_load_prompt_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("does-not-exist")


def test_engine_generate_raises_when_model_missing(tmp_path) -> None:
    engine = LLMEngine(model_dir=tmp_path / "nope")
    with pytest.raises((FileNotFoundError, ImportError)):
        engine.generate("hi")


def test_predict_next_returns_seed_for_empty_buffer() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    engine._max_new_tokens = 24  # noqa: SLF001
    engine._timeout = 30.0  # noqa: SLF001
    out = engine.predict_next("")
    assert out == ["I", "The", "Hello"]


def test_timeout_error_subclasses_timeout() -> None:
    assert issubclass(LLMTimeoutError, TimeoutError)
