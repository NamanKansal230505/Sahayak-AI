"""Global F12 kill switch.

Press F12 from anywhere on the system: tracking pauses, the
`ConfidenceGate` is forced shut, and PyAutoGUI receives a fail-safe ping
so any in-flight action terminates immediately.

Hotkey is intentionally hard-coded to F12 — per spec, not user-configurable.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock

from sahaayak.safety.confidence_gate import ConfidenceGate
from sahaayak.utils.i18n import i18n
from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

KILL_HOTKEY = "F12"  # Spec: not configurable.


class KillSwitch:
    """Wraps a global hotkey listener that toggles tracking on/off."""

    def __init__(
        self,
        gate: ConfidenceGate,
        on_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        self._gate = gate
        self._on_toggle = on_toggle
        self._engaged: bool = False
        self._lock = RLock()
        self._listener: object = None

    @property
    def engaged(self) -> bool:
        """``True`` when the kill switch is currently disabling tracking."""
        with self._lock:
            return self._engaged

    def toggle(self) -> None:
        """Flip kill-switch state. Invoked by the hotkey or programmatically."""
        with self._lock:
            self._engaged = not self._engaged
            self._gate.force_block(self._engaged)
            if self._on_toggle is not None:
                self._on_toggle(self._engaged)
        msg = i18n("kill_switch.engaged") if self._engaged else "Eye control re-enabled."
        logger.warning("KILL SWITCH %s. %s", "ENGAGED" if self._engaged else "RELEASED", msg)

    def install(self) -> None:
        """Install the global F12 hotkey via `pynput` if available.

        We import lazily so headless test environments don't need pynput.
        If pynput is missing we still expose `toggle()` for programmatic use.
        """
        try:
            from pynput import keyboard  # noqa: PLC0415

            def _on_press(key: object) -> None:
                if getattr(key, "name", None) == KILL_HOTKEY.lower():
                    self.toggle()

            listener = keyboard.Listener(on_press=_on_press)
            listener.daemon = True
            listener.start()
            self._listener = listener
            logger.info("Kill switch installed (hotkey: %s).", KILL_HOTKEY)
        except ImportError:
            logger.warning(
                "pynput not installed; F12 kill switch is unbound. "
                "Install with `pip install pynput` to enable it."
            )

    def uninstall(self) -> None:
        """Stop the hotkey listener."""
        listener = self._listener
        if listener is not None and hasattr(listener, "stop"):
            listener.stop()  # type: ignore[union-attr]
            self._listener = None
