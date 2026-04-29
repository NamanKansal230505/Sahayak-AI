"""Tests for the eye keyboard core (UI-agnostic)."""

from __future__ import annotations

from sahaayak.control.eye_keyboard import DEVANAGARI_ROWS, QWERTY_ROWS, EyeKeyboard


class _FakeCursor:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        self.events.append(("type", text))
        return True

    def press(self, key: str) -> bool:
        self.events.append(("press", key))
        return True


def test_press_key_appends_to_buffer() -> None:
    kb = EyeKeyboard(_FakeCursor())  # type: ignore[arg-type]
    kb.press_key("h")
    kb.press_key("i")
    assert kb.state.buffer == "hi"


def test_back_removes_from_buffer() -> None:
    cursor = _FakeCursor()
    kb = EyeKeyboard(cursor)  # type: ignore[arg-type]
    for ch in "abc":
        kb.press_key(ch)
    kb.press_key("back")
    assert kb.state.buffer == "ab"
    assert ("press", "backspace") in cursor.events


def test_lang_toggles_layout() -> None:
    kb = EyeKeyboard(_FakeCursor())  # type: ignore[arg-type]
    assert kb.rows() is QWERTY_ROWS
    kb.press_key("lang")
    assert kb.rows() is DEVANAGARI_ROWS


def test_predictions_called_on_each_keypress() -> None:
    seen: list[str] = []

    def predict(buf: str) -> list[str]:
        seen.append(buf)
        return ["foo", "bar"]

    kb = EyeKeyboard(_FakeCursor(), predict=predict)  # type: ignore[arg-type]
    kb.press_key("h")
    kb.press_key("i")
    assert seen == ["h", "hi"]
    assert kb.state.predictions == ["foo", "bar"]


def test_insert_prediction_replaces_partial_word() -> None:
    cursor = _FakeCursor()
    kb = EyeKeyboard(cursor)  # type: ignore[arg-type]
    kb.press_key("h")
    kb.press_key("e")
    kb.insert_prediction("hello")
    assert kb.state.buffer.endswith("hello ")
