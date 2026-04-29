"""Tests for the email-simplifier JSON parser."""

from __future__ import annotations

from sahaayak.copilot.email_simplifier import ActionItem, parse_response


def test_parse_response_extracts_all_fields() -> None:
    raw = """
    Some preamble.
    {
      "tldr": "Boss wants Q3 numbers by Friday.",
      "key_points": ["Q3 deck", "Friday deadline"],
      "action_items": [
        {"task": "Send Q3 numbers", "deadline": "Friday"}
      ],
      "tone": "formal",
      "suggested_reply_drafts": ["Will do.", "On it.", "Acknowledged."]
    }
    Trailing junk.
    """
    out = parse_response(raw)
    assert "Q3" in out.tldr
    assert out.key_points == ["Q3 deck", "Friday deadline"]
    assert out.action_items == [ActionItem(task="Send Q3 numbers", deadline="Friday")]
    assert out.tone == "formal"
    assert len(out.suggested_reply_drafts) == 3


def test_parse_response_falls_back_when_no_json() -> None:
    out = parse_response("Just a friendly note saying hi.")
    assert out.tldr.startswith("Just a friendly note")
    assert out.key_points == []
    assert out.action_items == []


def test_parse_response_handles_string_action_items() -> None:
    raw = '{"tldr": "x", "action_items": ["call back"]}'
    out = parse_response(raw)
    assert out.action_items == [ActionItem(task="call back")]


def test_parse_response_truncates_long_lists() -> None:
    raw = '{"tldr":"x","key_points":["a","b","c","d","e","f","g"],"suggested_reply_drafts":["1","2","3","4"]}'
    out = parse_response(raw)
    assert len(out.key_points) == 5
    assert len(out.suggested_reply_drafts) == 3
