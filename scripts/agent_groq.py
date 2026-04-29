"""Vision-based screen agent using Groq (Llama 4 Scout / Maverick).

Same pattern as scripts/agent.py but powered by Groq's OpenAI-compatible API
instead of Anthropic. Llama 4 Scout supports both vision and tool calling on
Groq's free tier (~30 req/min).

Loop: screenshot -> Groq (vision + tool use) -> PyAutoGUI -> repeat.

Setup:
    pip install groq pyautogui pillow pynput
    setx GROQ_API_KEY "gsk_..."     # one-time on Windows
    python scripts/agent_groq.py "open notepad and type hello" --confirm-all

Safety surfaces:
    * F12 kill switch via SahaayakAI's existing `KillSwitch` -> aborts the loop
    * PyAutoGUI fail-safe (mouse-into-corner) -> raises FailSafeException
    * Hard `--max-iterations` cap (default 25)
    * `--confirm-all` prompts before every action
    * Llama is instructed to call `request_confirmation` for irreversible
      actions; the script blocks on stdin for the user's approval

Why a separate file? Mixing Anthropic and OpenAI/Groq SDK calls in the same
module hides which provider a code path actually targets. agent.py uses
Claude; agent_groq.py uses Groq; they share the safety + screenshot pattern
but not the API client.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
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

import pyautogui  # noqa: E402
from groq import Groq  # noqa: E402
from PIL import Image  # noqa: E402

from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: E402
from sahaayak.safety.kill_switch import KillSwitch  # noqa: E402

# Groq's vision + tool-use model. Maverick is bigger/slower but stronger;
# Scout is the daily-driver default — fast, free-tier-friendly, multimodal.
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ALT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# Llama 4 vision works well at lower res; downscale aggressively for speed
# and free-tier rate limits.
MAX_IMAGE_DIM = 1280
JPEG_QUALITY = 70
DEFAULT_MAX_ITERATIONS = 25

# ──────────────────────────────────────────────────────────────────── prompt ──
SYSTEM_PROMPT = """\
You are SahaayakAI's screen agent, powered by Llama 4 on Groq. You drive a
Windows 11 laptop on behalf of a real user who has typed a goal in natural
language. You see the screen as a normal screenshot and you control the
keyboard and mouse via the tools below.

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

# ──────────────────────────────────────────────────────── tool definitions ──
# OpenAI-compatible function-calling schema — Groq accepts the same shape.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": (
                "Click at a specific pixel coordinate on the screen. Use for "
                "buttons, links, menu items, and form fields without a "
                "keyboard shortcut. Pixel space is the user's full screen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate (screen pixels)."},
                    "y": {"type": "integer", "description": "Y coordinate (screen pixels)."},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button. Default left.",
                    },
                    "double": {
                        "type": "boolean",
                        "description": "If true, double-click. Default false.",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type a string into the focused field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Exact text to type."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": (
                "Press a single special key. PyAutoGUI key names: enter, tab, "
                "esc, backspace, delete, up, down, left, right, home, end, "
                "pageup, pagedown, win, space."
            ),
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hotkey",
            "description": (
                "Press a keyboard shortcut. Examples: ['ctrl', 'c'] copy, "
                "['alt', 'tab'] switch windows, ['win', 'd'] show desktop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of key names; pressed together.",
                    },
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the active window up or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "description": "Wheel-click count (default 5).",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Pause for the UI to settle. Capped at 5 seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                    "reason": {"type": "string"},
                },
                "required": ["seconds", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_confirmation",
            "description": (
                "Block until the user approves an irreversible action. "
                "Required for sending messages, deleting data, payments, "
                "closing unsaved work, system changes, installs/uninstalls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One sentence describing what is about to happen.",
                    },
                    "why_irreversible": {"type": "string"},
                },
                "required": ["action", "why_irreversible"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the goal is complete.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "failed",
            "description": "Signal that the goal cannot be achieved.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


# ───────────────────────────────────────────────────────── screenshotting ──
def _capture_image_data_uri() -> tuple[str, tuple[int, int]]:
    """Return (data: URI, original screen size). Always JPEG."""
    img = pyautogui.screenshot()
    orig_w, orig_h = img.size
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
    return f"data:image/jpeg;base64,{b64}", (orig_w, orig_h)


def _user_message_with_screenshot(text: str) -> dict[str, Any]:
    """Build an OpenAI-format user message containing image + text."""
    data_uri, _ = _capture_image_data_uri()
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": text},
        ],
    }


# ──────────────────────────────────────────────────── tool execution ──
def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run a tool and return a short text describing what happened."""
    if name == "click":
        x, y = int(args["x"]), int(args["y"])
        button = args.get("button") or "left"
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
        amount = int(args.get("amount") or 5)
        clicks = -amount if args["direction"] == "down" else amount
        pyautogui.scroll(clicks)
        return f"Scrolled {args['direction']} by {amount}."
    if name == "wait":
        seconds = max(0.1, min(float(args.get("seconds") or 1.0), 5.0))
        time.sleep(seconds)
        return f"Waited {seconds:.1f}s ({args.get('reason', '')})."
    return f"Unknown tool: {name}"


def _ask_confirmation(prompt: str) -> bool:
    print()
    print("=" * 72)
    print(f"CONFIRM: {prompt}")
    print("=" * 72)
    try:
        ans = input("Allow this action? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


# ─────────────────────────────────────────────────────────────── loop ──
def run_agent(
    goal: str,
    model: str = DEFAULT_MODEL,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    confirm_all: bool = False,
    api_key: str | None = None,
) -> int:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    client = Groq(api_key=api_key) if api_key else Groq()
    gate = ConfidenceGate()
    abort_evt = Event()

    def _on_kill(engaged: bool) -> None:
        if engaged:
            abort_evt.set()

    kill = KillSwitch(gate, on_toggle=_on_kill)
    kill.install()

    initial_uri, screen_size = _capture_image_data_uri()
    print(f"Model:  {model}")
    print(f"Screen: {screen_size[0]}x{screen_size[1]}")
    print(f"Goal:   {goal}")
    print(f"Max iterations: {max_iterations}    confirm-all: {confirm_all}")
    print("Safety: F12 = abort. Mouse-into-corner = fail-safe abort. Ctrl+C = abort.")
    print("-" * 72)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": initial_uri}},
                {
                    "type": "text",
                    "text": (
                        f"GOAL: {goal}\n\nScreen size: {screen_size[0]}x{screen_size[1]} "
                        "(your tool calls use these original screen coordinates).\n\n"
                        "Look at the screenshot and take the next single action."
                    ),
                },
            ],
        },
    ]

    exit_code = 0
    try:
        for iteration in range(1, max_iterations + 1):
            if abort_evt.is_set():
                print("Kill switch engaged — aborting.")
                exit_code = 130
                break

            messages = _strip_old_images(messages, keep_last_n_user_turns=2)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=1024,
                    temperature=0.2,
                )
            except Exception as exc:  # noqa: BLE001 — surface API errors clearly
                print(f"Groq API call failed: {exc!r}")
                exit_code = 1
                break

            usage = response.usage
            msg = response.choices[0].message
            print(
                f"\n[turn {iteration:02d}/{max_iterations}] "
                f"in={usage.prompt_tokens} out={usage.completion_tokens} "
                f"total={usage.total_tokens} stop={response.choices[0].finish_reason}"
            )
            if msg.content:
                print(f"  agent> {msg.content.strip()[:200]}")

            # Append assistant message to history (preserve tool_calls for openai shape)
            messages.append(_assistant_msg_to_dict(msg))

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                print("Agent produced no tool call. Bailing.")
                exit_code = 1
                break

            should_break = False
            for call in tool_calls:
                if abort_evt.is_set():
                    should_break = True
                    break

                fn_name = call.function.name
                try:
                    fn_args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    err = f"Could not parse tool args: {exc}"
                    print(f"  ⚠ {err}")
                    messages.append({
                        "role": "tool", "tool_call_id": call.id, "content": err,
                    })
                    continue

                if fn_name == "done":
                    print(f"  ✅ DONE: {fn_args.get('summary', '')}")
                    should_break = True
                    break
                if fn_name == "failed":
                    print(f"  ❌ FAILED: {fn_args.get('reason', '')}")
                    exit_code = 1
                    should_break = True
                    break

                # Confirmation gate
                needs_confirm = fn_name == "request_confirmation" or confirm_all
                if needs_confirm:
                    if fn_name == "request_confirmation":
                        prompt = (
                            f"{fn_args.get('action', '?')}\n"
                            f"Why: {fn_args.get('why_irreversible', '?')}"
                        )
                    else:
                        prompt = f"{fn_name}({json.dumps(fn_args)[:120]})"
                    approved = _ask_confirmation(prompt)
                    result_text = (
                        "User APPROVED. Proceed with the action."
                        if approved
                        else "User DECLINED. Choose a different approach or call failed."
                    )
                    messages.append({
                        "role": "tool", "tool_call_id": call.id, "content": result_text,
                    })
                    if fn_name == "request_confirmation":
                        # Just an answer to the question; loop continues so model
                        # can decide what to do with the approval/decline.
                        continue
                    if not approved:
                        continue

                try:
                    out_msg = _execute_tool(fn_name, fn_args)
                except pyautogui.FailSafeException:
                    print("  PyAutoGUI fail-safe triggered. Aborting.")
                    exit_code = 130
                    should_break = True
                    break
                except Exception as exc:  # noqa: BLE001
                    out_msg = f"Tool {fn_name} raised: {exc!r}"
                    print(f"  ⚠ {out_msg}")
                    messages.append({
                        "role": "tool", "tool_call_id": call.id, "content": out_msg,
                    })
                    continue

                print(f"  -> {out_msg}")
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "content": out_msg,
                })

            if should_break:
                break

            # OpenAI/Groq tool messages can't carry images, so attach the post-
            # action screenshot as a fresh user message before the next turn.
            messages.append(_user_message_with_screenshot(
                "Here is the current screen state. Take the next single action."
            ))

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


# ───────────────────────────────────────────────────────────── helpers ──
def _assistant_msg_to_dict(msg: Any) -> dict[str, Any]:
    """Convert a Groq SDK ChatCompletionMessage into a JSON-serializable dict."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in msg.tool_calls
        ]
    return out


def _strip_old_images(
    messages: list[dict[str, Any]],
    keep_last_n_user_turns: int = 2,
) -> list[dict[str, Any]]:
    """Replace image content blocks in older user messages with text placeholders."""
    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_indices) <= keep_last_n_user_turns:
        return messages
    keep_after = user_indices[-keep_last_n_user_turns]
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i >= keep_after or m["role"] != "user" or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        new_content = []
        for block in m["content"]:
            if isinstance(block, dict) and block.get("type") == "image_url":
                new_content.append({"type": "text", "text": "[earlier screenshot omitted]"})
            else:
                new_content.append(block)
        out.append({**m, "content": new_content})
    return out


# ─────────────────────────────────────────────────────────────── main ──
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vision-based screen agent on Groq (Llama 4 Scout / Maverick).",
    )
    parser.add_argument("goal", help="Natural-language goal for the agent.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=(
            f"Groq model. Default: {DEFAULT_MODEL} (fast, free-tier-friendly). "
            f"Stronger alt: {ALT_MODEL}"
        ),
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help=f"Hard cap on agent loop iterations (default {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--confirm-all", action="store_true",
        help="Prompt for confirmation before every action (recommended for first runs).",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY not set.\n"
            "Get a free key at https://console.groq.com/keys then:\n"
            '  PowerShell:  $env:GROQ_API_KEY = "gsk_..."\n'
            '  Bash:        export GROQ_API_KEY="gsk_..."',
            file=sys.stderr,
        )
        return 2

    return run_agent(
        goal=args.goal,
        model=args.model,
        max_iterations=args.max_iterations,
        confirm_all=args.confirm_all,
        api_key=api_key,
    )


if __name__ == "__main__":
    sys.exit(main())
