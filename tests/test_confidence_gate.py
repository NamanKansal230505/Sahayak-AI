"""Tests for the safety confidence gate."""

from __future__ import annotations

from sahaayak.safety.confidence_gate import ConfidenceGate


def test_gate_starts_open() -> None:
    gate = ConfidenceGate()
    assert not gate.is_blocked()


def test_gate_blocks_after_grace_period() -> None:
    gate = ConfidenceGate(min_confidence=0.6, grace_ms=100)
    gate.update(0.2, ts=0.0)
    assert not gate.is_blocked()  # under grace
    blocked = gate.update(0.2, ts=0.5)
    assert blocked


def test_gate_releases_when_confidence_recovers() -> None:
    gate = ConfidenceGate(min_confidence=0.6, grace_ms=10)
    gate.update(0.1, ts=0.0)
    gate.update(0.1, ts=1.0)
    assert gate.is_blocked()
    gate.update(0.9, ts=2.0)
    assert not gate.is_blocked()


def test_force_block_pins_gate() -> None:
    gate = ConfidenceGate()
    gate.force_block(True)
    gate.update(1.0, ts=0.0)
    assert gate.is_blocked()
    gate.force_block(False)
    assert not gate.is_blocked()
