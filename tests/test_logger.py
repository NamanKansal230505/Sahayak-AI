"""Tests for sahaayak.utils.logger."""

from __future__ import annotations

import logging

from sahaayak.utils.logger import get_logger


def test_get_logger_returns_logger_instance() -> None:
    logger = get_logger("sahaayak.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "sahaayak.test"


def test_get_logger_is_idempotent() -> None:
    a = get_logger("sahaayak.test")
    b = get_logger("sahaayak.test")
    assert a is b


def test_get_logger_accepts_string_level() -> None:
    logger = get_logger("sahaayak.test.level", level="DEBUG")
    assert isinstance(logger, logging.Logger)
