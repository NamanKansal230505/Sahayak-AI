"""PyAutoGUI cursor controller with confidence-gated safety bounds.

Every public method consults the `ConfidenceGate` before talking to
PyAutoGUI. We move the cursor in N small linear-interpolated steps to
avoid the "teleport" feel that triggers motion-sickness for some users.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sahaayak.utils.logger import get_logger

if TYPE_CHECKING:
    from sahaayak.safety.confidence_gate import ConfidenceGate

logger = get_logger(__name__)


class CursorController:
    """Move and click the OS cursor based on smoothed gaze coordinates.

    Args:
        gate: The shared `ConfidenceGate` instance.
        smoothing_steps: Number of intermediate moves between source and
            target. Higher = smoother but laggier.
        screen_size: Optional (w, h). If ``None`` we ask PyAutoGUI.
    """

    def __init__(
        self,
        gate: ConfidenceGate,
        smoothing_steps: int = 4,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        self._gate = gate
        self._steps = max(1, int(smoothing_steps))
        self._pyautogui: object | None = None
        self._screen_size = screen_size
        self._last_pos: tuple[int, int] | None = None

    def _backend(self) -> object:
        if self._pyautogui is not None:
            return self._pyautogui
        try:
            import pyautogui  # noqa: PLC0415

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.0
            self._pyautogui = pyautogui
            if self._screen_size is None:
                self._screen_size = pyautogui.size()
            return pyautogui
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyautogui not installed. Run `pip install -r requirements.txt`."
            ) from exc

    def screen_size(self) -> tuple[int, int]:
        """Return the primary screen size (w, h)."""
        if self._screen_size is None:
            self._backend()
        return self._screen_size or (1920, 1080)

    def move(self, x: float, y: float) -> bool:
        """Move the cursor toward (x, y). Returns ``True`` if movement happened."""
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        target = (int(x), int(y))
        start = self._last_pos or pyautogui.position()  # type: ignore[attr-defined]
        for i in range(1, self._steps + 1):
            t = i / self._steps
            ix = int(start[0] + (target[0] - start[0]) * t)
            iy = int(start[1] + (target[1] - start[1]) * t)
            pyautogui.moveTo(ix, iy, _pause=False)  # type: ignore[attr-defined]
        self._last_pos = target
        return True

    def click(self, button: str = "left") -> bool:
        """Issue a single click of the requested button. Returns success."""
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        pyautogui.click(button=button)  # type: ignore[attr-defined]
        return True

    def double_click(self) -> bool:
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        pyautogui.doubleClick()  # type: ignore[attr-defined]
        return True

    def scroll(self, clicks: int) -> bool:
        """Scroll by the given number of wheel clicks (negative = down)."""
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        pyautogui.scroll(int(clicks))  # type: ignore[attr-defined]
        return True

    def press(self, key: str) -> bool:
        """Press and release a single key (e.g. for the eye keyboard)."""
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        pyautogui.press(key)  # type: ignore[attr-defined]
        return True

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type a string. Used by the eye keyboard."""
        if self._gate.is_blocked():
            return False
        pyautogui = self._backend()
        pyautogui.typewrite(text, interval=interval)  # type: ignore[attr-defined]
        return True

    def position(self) -> tuple[int, int]:
        """Return the OS cursor position."""
        pyautogui = self._backend()
        x, y = pyautogui.position()  # type: ignore[attr-defined]
        return int(x), int(y)

    def cooldown(self, seconds: float) -> None:
        """Sleep helper used by the gesture engine to avoid double-fires."""
        time.sleep(max(0.0, seconds))
