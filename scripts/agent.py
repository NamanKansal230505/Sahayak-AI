"""Vision-based screen agent using Claude Sonnet 4.6.

Loop: screenshot -> Claude (vision + tool use) -> PyAutoGUI -> repeat.

Replicates the Self-Operating-Computer pattern (cf. soumyagautam/Eye-Mouse-Tracking
analog at masfaatanveer/Agentic-AI-Computer) but on top of Claude's native
tool-use API with prompt caching on the static prefix (system + tool defs).

Safety surfaces:
    * F12 kill switch via SahaayakAI's existing `KillSwitch` -> aborts the loop
    * PyAutoGUI fail-safe (mouse-into-corner) -> raises FailSafeException
    * Hard `--max-iterations` cap (default 25)
    * `--confirm-all` prompts before every action
    * Claude is instructed to call `request_confirmation` for irreversible
      actions; the script blocks on stdin for the user's approval

Run:
    setx ANTHROPIC_API_KEY "sk-ant-..."     # one-time
    python scripts/agent.py "open notepad and type hello"
    python scripts/agent.py "find the q3 deck in my downloads" --confirm-all
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Force UTF-8 stdout so non-ASCII glyphs render on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import anthropic  # noqa: E402
import pyautogui  # noqa: E402
from PIL import Image  # noqa: E402

from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: E402
from sahaayak.safety.kill_switch import KillSwitch  # noqa: E402

MODEL = "claude-sonnet-4-6"
# Sonnet 4.6 vision caps at 1568px on the long edge; downscale to control cost.
MAX_IMAGE_DIM = 1568
JPEG_QUALITY = 70
DEFAULT_MAX_ITERATIONS = 25

# ──────────────────────────────────────────────────────────────────── prompt ──
SYSTEM_PROMPT = """\
You are SahaayakAI's screen agent. You drive a Windows 11 laptop on behalf of
a real user who has typed a goal in natural language. You see the screen as a
mirrored image (no head-mirroring trick — just a normal screenshot) and you
control the keyboard and mouse via the tools below.

# How to think

For every step:

1. Look at the latest screenshot carefully. Read all visible text. Identify
   the active window, the cursor position, focused field, and any modal
   dialogs. Do not invent UI elements that aren't on screen.
2. Decide the next single concrete action that moves toward the goal. Prefer
   keyboard shortcuts over mouse clicks when both work — they are faster and
   more precise. Examples: `Ctrl+L` to focus the address bar in a browser,
   `Win+R` to open Run, `Win+E` for File Explorer, `Alt+F4` to close, `Win+D`
   to show the desktop.
3. Issue exactly one tool call. The screenshot you receive next reflects the
   result. Do not chain multiple guesses; one action, one observation.
4. If a previous action did not produce the expected result, do not repeat it
   blindly. Look at the new screenshot, reason about why it failed, and try a
   different approach. If you are stuck after 3 attempts at the same subgoal,
   call `failed` with a short explanation.
5. When the goal is fully accomplished, call `done` with a one-sentence
   summary of what you achieved. Do not call `done` until the goal is
   genuinely complete and visible on screen.

# Coordinates

The screenshot you see has been resized for inference. The tools accept
coordinates in *original screen pixel space* — the runtime handles the
rescaling for you, so you just specify the pixel coordinates of the element
you want to interact with as you see them on the supplied image (relative to
the image's top-left). Be precise; off-by-50 px clicks routinely miss small
buttons.

# Safety — non-negotiable

Some actions are irreversible or potentially harmful. For ANY of the following,
call `request_confirmation` first with a clear description, and only proceed
to the actual action AFTER the user approves:

  - Sending an email, message, post, comment, or chat
  - Deleting any file, folder, message, or record
  - Submitting a form that involves payment, purchase, or financial commitment
  - Closing an unsaved document, browser tab, or editor window
  - Changing system settings (network, accounts, permissions, registry)
  - Installing, uninstalling, or updating software
  - Anything you would feel bad about doing if it turned out to be wrong

If the user declines confirmation, call `failed` with the reason and stop.

Do NOT call `request_confirmation` for ordinary navigation: opening apps,
clicking links, typing into fields, scrolling, switching windows. Confirmation
fatigue defeats the purpose.

# Style

Be terse in your text replies. Most turns should be 0–2 sentences plus the
tool call. The user can read the screen and the diff for themselves; do not
narrate every visible element.
"""


# ──────────────────────────────────────────────────────────────────── tools ──
TOOLS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": (
            "Click at a specific pixel coordinate on the screen. Use for "
            "buttons, links, menu items, and form fields that don't have a "
            "keyboard shortcut. Pixel space is the user's full screen "
            "(not the resized inference image)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "X coordinate in screen pixels."},
                "y": {"type": "integer", "description": "Y coordinate in screen pixels."},
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "default": "left",
                    "description": "Mouse button. Right click for context menus.",
                },
                "double": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, double-click instead of single.",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Type a string into whatever field currently has focus. Click the "
            "field first if needed. Newlines are typed as Enter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Exact text to type."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "press_key",
        "description": (
            "Press a single special key. Use for navigation and editor "
            "shortcuts that don't fit type_text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "PyAutoGUI key name. Common: enter, tab, esc, "
                        "backspace, delete, up, down, left, right, home, end, "
                        "pageup, pagedown, win, space."
                    ),
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "hotkey",
        "description": (
            "Press a keyboard shortcut — one or more keys held simultaneously. "
            "Examples: ['ctrl', 'c'] to copy, ['alt', 'tab'] to switch "
            "windows, ['win', 'd'] to show desktop, ['ctrl', 'shift', 't'] to "
            "reopen closed tab."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Ordered list of key names; pressed together.",
                },
            },
            "required": ["keys"],
        },
    },
    {
        "name": "scroll",
        "description": "Scroll the active window. Direction is up or down.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["up", "down"]},
                "amount": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 5,
                    "description": "Wheel-click count. 5 ≈ one screenful on most apps.",
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Pause for the UI to settle (loading, animations, app launches). "
            "Capped at 5 s — for longer waits, take a fresh screenshot and "
            "re-check the state instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                "reason": {"type": "string", "description": "Why you're waiting."},
            },
            "required": ["seconds", "reason"],
        },
    },
    {
        "name": "request_confirmation",
        "description": (
            "Block until the user explicitly approves an irreversible or "
            "high-stakes action. Required for sending messages, deleting "
            "data, payments, closing unsaved work, system changes, "
            "installs/uninstalls. The next observation tells you whether the "
            "user said yes — proceed only on yes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "One sentence describing exactly what is about to "
                        "happen, e.g. 'send this email to priya@example.com'."
                    ),
                },
                "why_irreversible": {
                    "type": "string",
                    "description": "Brief reason confirmation is required.",
                },
            },
            "required": ["action", "why_irreversible"],
        },
    },
    {
        "name": "done",
        "description": "Signal that the goal is complete. The loop ends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-sentence recap of what was achieved.",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "failed",
        "description": (
            "Signal that the goal cannot be achieved. The loop ends. Use this "
            "when blocked after multiple attempts, after a denied "
            "confirmation, or when the goal is impossible from the current state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why the goal failed."},
            },
            "required": ["reason"],
        },
    },
]


# ───────────────────────────────────────────────────────────── screenshotting ──
def _capture_image_block() -> tuple[dict[str, Any], tuple[int, int]]:
    """Take a screenshot, return (image content block, original screen size).

    Returned screen size lets callers note the original pixel space; the model
    is told to issue coordinates in original-pixel space.
    """
    img = pyautogui.screenshot()
    orig_w, orig_h = img.size
    # Downscale to MAX_IMAGE_DIM on the long edge for cost control. The model
    # is informed coordinates are in original-pixel space.
    long_edge = max(orig_w, orig_h)
    if long_edge > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / long_edge
        img = img.resize(
            (int(orig_w * scale), int(orig_h * scale)),
            resample=Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return (
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        },
        (orig_w, orig_h),
    )


# ──────────────────────────────────────────────────────────── tool execution ──
def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool and return a short text describing what happened."""
    if name == "click":
        x, y = int(args["x"]), int(args["y"])
        button = args.get("button", "left")
        if args.get("double"):
            pyautogui.doubleClick(x, y, button=button)
            return f"Double-clicked {button} at ({x}, {y})."
        pyautogui.click(x, y, button=button)
        return f"Clicked {button} at ({x}, {y})."
    if name == "type_text":
        text = str(args["text"])
        pyautogui.typewrite(text, interval=0.015)
        preview = text[:80] + ("…" if len(text) > 80 else "")
        return f"Typed {len(text)} chars: {preview!r}"
    if name == "press_key":
        key = str(args["key"])
        pyautogui.press(key)
        return f"Pressed key: {key}"
    if name == "hotkey":
        keys = [str(k) for k in args["keys"]]
        pyautogui.hotkey(*keys)
        return f"Pressed hotkey: {'+'.join(keys)}"
    if name == "scroll":
        amount = int(args.get("amount", 5))
        clicks = -amount if args["direction"] == "down" else amount
        pyautogui.scroll(clicks)
        return f"Scrolled {args['direction']} by {amount}."
    if name == "wait":
        seconds = max(0.1, min(float(args.get("seconds", 1.0)), 5.0))
        time.sleep(seconds)
        return f"Waited {seconds:.1f}s ({args.get('reason', '')})."
    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────── confirmation ──
def _ask_confirmation(prompt: str) -> bool:
    """Block on stdin until the user types y or n. Default to NO."""
    print()
    print("=" * 72)
    print(f"CONFIRM: {prompt}")
    print("=" * 72)
    try:
        ans = input("Allow this action? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


# ───────────────────────────────────────────────────────────────────── loop ──
def run_agent(
    goal: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    confirm_all: bool = False,
    api_key: str | None = None,
) -> int:
    """Drive Claude through the agent loop. Returns process exit code."""
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05  # tiny inter-call gap so the OS catches up

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    gate = ConfidenceGate()
    abort_evt = Event()

    def _on_kill(engaged: bool) -> None:
        if engaged:
            abort_evt.set()

    kill = KillSwitch(gate, on_toggle=_on_kill)
    kill.install()

    # Initial user message: goal + first screenshot.
    first_image, screen_size = _capture_image_block()
    print(f"Screen: {screen_size[0]}x{screen_size[1]}")
    print(f"Goal: {goal}")
    print(f"Max iterations: {max_iterations}")
    print(f"Confirm-all mode: {confirm_all}")
    print(
        "Safety: F12 = abort. Mouse-into-corner = PyAutoGUI fail-safe abort. "
        "Ctrl+C = also abort."
    )
    print("-" * 72)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                first_image,
                {
                    "type": "text",
                    "text": (
                        f"GOAL: {goal}\n\nScreen size: {screen_size[0]}x{screen_size[1]} "
                        f"(your tool calls use these original coordinates).\n\n"
                        f"Look at the screenshot and take the next single action."
                    ),
                },
            ],
        }
    ]

    exit_code = 0
    try:
        for iteration in range(1, max_iterations + 1):
            if abort_evt.is_set():
                print("Kill switch engaged — aborting.")
                exit_code = 130
                break

            # Trim image blocks from messages older than the most recent 2
            # user turns to control prompt growth — Claude still has the tool
            # call/result text history for memory.
            messages = _strip_old_images(messages, keep_last_n_user_turns=2)

            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                # cache_control on the last system block caches
                # tools + system together (tools render before system).
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=TOOLS,
                messages=messages,
            )

            usage = response.usage
            print(
                f"\n[turn {iteration:02d}/{max_iterations}] "
                f"in={usage.input_tokens} cached_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                f"cached_write={getattr(usage, 'cache_creation_input_tokens', 0)} "
                f"out={usage.output_tokens} stop={response.stop_reason}"
            )
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  agent> {block.text.strip()}")

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                print("Agent ended turn without using a tool. Done.")
                break

            tool_results: list[dict[str, Any]] = []
            should_break = False
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if abort_evt.is_set():
                    should_break = True
                    break

                # Terminal tools end the loop directly (no result needed).
                if block.name == "done":
                    print(f"  ✅ DONE: {block.input.get('summary', '')}")
                    should_break = True
                    break
                if block.name == "failed":
                    print(f"  ❌ FAILED: {block.input.get('reason', '')}")
                    exit_code = 1
                    should_break = True
                    break

                # Confirmation gate.
                needs_confirm = block.name == "request_confirmation" or confirm_all
                if needs_confirm:
                    if block.name == "request_confirmation":
                        prompt = (
                            f"{block.input.get('action', '?')}\n"
                            f"Why: {block.input.get('why_irreversible', '?')}"
                        )
                    else:
                        prompt = f"{block.name}({_short_json(block.input)})"
                    approved = _ask_confirmation(prompt)
                    result_text = (
                        "User APPROVED. Proceed with the action."
                        if approved
                        else "User DECLINED. Choose a different approach or call `failed`."
                    )
                    if block.name == "request_confirmation":
                        # request_confirmation itself is the question — return
                        # the answer; do not also execute anything here.
                        tool_results.append(_text_only_result(block.id, result_text))
                        continue
                    if not approved:
                        tool_results.append(_text_only_result(block.id, result_text, is_error=True))
                        continue

                # Execute the tool.
                try:
                    msg = _execute_tool(block.name, block.input)
                except pyautogui.FailSafeException:
                    print("  PyAutoGUI fail-safe triggered (mouse in corner). Aborting.")
                    exit_code = 130
                    should_break = True
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = f"Tool {block.name} raised: {exc!r}"
                    print(f"  ⚠ {msg}")
                    tool_results.append(_text_only_result(block.id, msg, is_error=True))
                    continue

                print(f"  -> {msg}")
                # Capture a fresh screenshot so the next turn sees the result.
                screenshot_block, screen_size = _capture_image_block()
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": [
                            {"type": "text", "text": msg},
                            screenshot_block,
                        ],
                    }
                )

            if should_break:
                break
            if not tool_results:
                print("No tool calls produced. Bailing.")
                break
            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"\nReached max iterations ({max_iterations}) without `done`. Stopping.")
            exit_code = 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit_code = 130
    except pyautogui.FailSafeException:
        print("\nPyAutoGUI fail-safe triggered. Aborting.")
        exit_code = 130
    finally:
        kill.uninstall()
    return exit_code


# ────────────────────────────────────────────────────────────────── helpers ──
def _text_only_result(tool_use_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
    }
    if is_error:
        out["is_error"] = True
    return out


def _strip_old_images(
    messages: list[dict[str, Any]],
    keep_last_n_user_turns: int = 2,
) -> list[dict[str, Any]]:
    """Replace image blocks in older user messages with text placeholders.

    Keeps prompt growth bounded as the loop runs. Claude still has the tool
    call / tool result text for memory of what happened.
    """
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_indices) <= keep_last_n_user_turns:
        return messages
    keep_after = user_indices[-keep_last_n_user_turns]
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i >= keep_after or msg["role"] != "user":
            out.append(msg)
            continue
        # Walk content blocks; replace images with a text placeholder.
        if isinstance(msg.get("content"), list):
            new_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "image":
                    new_content.append(
                        {"type": "text", "text": "[earlier screenshot omitted]"}
                    )
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        block = dict(block)
                        block["content"] = [
                            {"type": "text", "text": "[earlier screenshot omitted]"}
                            if isinstance(b, dict) and b.get("type") == "image"
                            else b
                            for b in inner
                        ]
                    new_content.append(block)
                else:
                    new_content.append(block)
            out.append({**msg, "content": new_content})
        else:
            out.append(msg)
    return out


def _short_json(d: dict[str, Any]) -> str:
    import json as _json  # noqa: PLC0415
    return _json.dumps(d)[:120]


# ───────────────────────────────────────────────────────────────────── main ──
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a vision-based screen agent driven by Claude Sonnet 4.6.",
    )
    parser.add_argument("goal", help="Natural-language goal for the agent.")
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Hard cap on agent loop iterations (default {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--confirm-all", action="store_true",
        help="Prompt for confirmation before every action (recommended for first runs).",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set.\n"
            "Set it in your shell:\n"
            '  PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."\n'
            '  Bash:        export ANTHROPIC_API_KEY="sk-ant-..."',
            file=sys.stderr,
        )
        return 2

    return run_agent(
        goal=args.goal,
        max_iterations=args.max_iterations,
        confirm_all=args.confirm_all,
        api_key=api_key,
    )


if __name__ == "__main__":
    sys.exit(main())
