"""Map detected gestures to OS-level actions.

The dispatcher is the only place where gesture semantics are bound. The
default mapping comes from the project spec; the constructor accepts an
override dict for users who want to remap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sahaayak.core.gesture_engine import GestureKind
from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.control.cursor_controller import CursorController
    from sahaayak.core.gesture_engine import Gesture


logger = get_logger(__name__)


class ActionDispatcher:
    """Translate `Gesture` events into `CursorController` calls + UI toggles.

    Args:
        cursor: The cursor controller backing real OS interaction.
        on_toggle_keyboard: Callback fired when WINK_LEFT toggles the eye keyboard.
        on_toggle_copilot: Callback fired when WINK_RIGHT toggles the copilot panel.
    """

    def __init__(
        self,
        cursor: CursorController,
        on_toggle_keyboard: Callable[[], None] | None = None,
        on_toggle_copilot: Callable[[], None] | None = None,
    ) -> None:
        self._cursor = cursor
        self._on_toggle_keyboard = on_toggle_keyboard
        self._on_toggle_copilot = on_toggle_copilot

    def dispatch(self, gesture: Gesture) -> None:
        """Execute the action bound to this gesture.

        Click bindings (per real-user request during the live demo):
            WINK_LEFT  -> LEFT click
            WINK_RIGHT -> RIGHT click
        Blinks remain as a redundant alternative since some users find
        winking difficult and prefer dwell/blink to click.
        """
        kind = gesture.kind
        try:
            if kind is GestureKind.WINK_LEFT:
                self._cursor.click("left")
            elif kind is GestureKind.WINK_RIGHT:
                self._cursor.click("right")
            elif kind is GestureKind.SHORT_BLINK:
                self._cursor.click("left")
            elif kind is GestureKind.LONG_BLINK:
                self._cursor.click("right")
            elif kind is GestureKind.DOUBLE_BLINK:
                # Double-blink doubles as a panel toggle when the keyboard
                # callback is wired, otherwise becomes a double-click.
                if self._on_toggle_keyboard is not None:
                    self._on_toggle_keyboard()
                else:
                    self._cursor.double_click()
            elif kind is GestureKind.DWELL:
                self._cursor.click("left")
            elif kind is GestureKind.GAZE_ZONE_TOP:
                self._cursor.scroll(3)
            elif kind is GestureKind.GAZE_ZONE_BOTTOM:
                self._cursor.scroll(-3)
            elif kind is GestureKind.GAZE_OFF_SCREEN:
                logger.info("Gaze off-screen for >2s; pausing tracking.")
        except Exception as exc:  # noqa: BLE001 — never let a UI gesture crash the app
            logger.exception("Action dispatch for %s failed: %s", kind, exc)
