"""Vision-based screen agent backed by the `claude` CLI (Claude Code Max).

Yes, this is duct tape. The Anthropic Agent SDK refuses to use your Max
subscription auth — but the `claude -p` command-line interface uses your
already-logged-in session, so we can shell out to it once per agent step.
Each call:

    1. Take screenshot, save as PNG to a temp file
    2. Spawn `claude -p "<prompt>" --output-format json --allowed-tools Read`
       Claude reads the screenshot via its built-in Read tool and decides
       the next single action.
    3. Parse the JSON action from the response
    4. Execute via PyAutoGUI
    5. Repeat

Trade-offs vs scripts/agent_groq.py and scripts/agent.py:

    + Uses your existing Claude Code Max subscription — no new API key
    + Backed by Claude Opus 4.7, the strongest agentic model available
    + System context (CLAUDE.md + tools) is cached server-side for 1 hour,
      so repeated calls in the same session amortise the prefix cost
    - ~3-6 s subprocess overhead per turn (Node startup + auth check)
    - Each `claude -p` invocation counts as one message against your Max
      5-hour rolling quota — a 20-step agent burns ~20 messages
    - Tools are restricted to `Read` to keep the agent from accidentally
      editing files or running arbitrary commands inside Claude Code

If your Max quota matters: prefer scripts/agent_groq.py for casual use and
keep this for the "I want the strongest model" runs.

Run:
    python scripts/agent_claude.py "open notepad" --confirm-all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
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

from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: E402
from sahaayak.safety.kill_switch import KillSwitch  # noqa: E402

DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TURN_TIMEOUT_S = 120

# JSON action object regex — finds {"action": ...} blocks in claude output.
ACTION_RE = re.compile(r'\{[^{}]*"action"\s*:[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)


PROMPT_TEMPLATE = """\
You are a screen-control agent for the SahaayakAI hackathon project. Look at
the screenshot below using the Read tool, then respond with EXACTLY one
JSON action describing the single next step to take. Be terse — every
extra word costs the user subscription quota.

GOAL: {goal}

SCREEN SIZE (original pixels — your coordinates use this space): {screen_w} x {screen_h}

SCREENSHOT FILE: {image_path}
(Use the Read tool on this path to view the current screen state.)

PREVIOUS ACTIONS (oldest -> newest):
{history}

# Allowed actions

Respond with ONE of these JSON objects on the LAST line of your response,
with no markdown fence, no commentary after it. The runtime parses your
LAST `{{"action": ...}}` JSON object — anything after that is ignored.

{{"action": "click", "x": INT, "y": INT, "button": "left|right|middle" (default left), "double": BOOL (default false)}}
{{"action": "type", "text": STR}}
{{"action": "press", "key": STR}}             // pyautogui key name: enter, tab, esc, backspace, win, ...
{{"action": "hotkey", "keys": [STR, STR, ...]}}  // e.g. ["ctrl", "l"], ["win", "r"], ["alt", "tab"]
{{"action": "scroll", "direction": "up|down", "amount": INT}}  // amount = wheel clicks, 1-30
{{"action": "wait", "seconds": NUM, "reason": STR}}            // capped at 5
{{"action": "done", "summary": STR}}            // goal complete
{{"action": "failed", "reason": STR}}           // give up

# Safety

For ANY irreversible action (sending an email/message, deleting data,
payments, closing unsaved work, system / install changes), respond with
{{"action": "failed", "reason": "needs explicit user confirmation: <what you'd do>"}}
and let the human take over. Do NOT silently submit or send.

Prefer keyboard shortcuts over clicks when both work — they are faster and
more precise. Examples: Ctrl+L (browser address bar), Win+R (Run), Win+E
(File Explorer), Alt+F4 (close), Win+D (show desktop).

If a previous action did not produce the expected result, try a different
approach. After 3 failed attempts at the same subgoal, return "failed".
"""


def _capture_screenshot() -> tuple[Path, tuple[int, int]]:
    """Save a fresh PNG screenshot to a temp path. Returns (path, (w, h))."""
    img = pyautogui.screenshot()
    fd, path = tempfile.mkstemp(prefix="sahaayak_screen_", suffix=".png")
    Path(path).write_bytes(b"")  # ensure writable
    import os as _os  # noqa: PLC0415

    _os.close(fd)
    img.save(path, format="PNG")
    return Path(path), img.size


def _run_claude(prompt: str, timeout: int) -> tuple[str, dict[str, Any]]:
    """Invoke `claude -p` and return (assistant_text, full_json_response)."""
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format", "json",
        # Restrict tools to Read so the agent can't accidentally Bash/Edit
        # in your repo while planning the next click.
        "--allowed-tools", "Read",
    ]
    proc = subprocess.run(  # noqa: S603 — args composed from constants
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if proc.returncode != 0:
        snippet = (proc.stderr or proc.stdout or "")[:500]
        raise RuntimeError(f"claude exited {proc.returncode}: {snippet}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse claude output as JSON: {exc}\n"
            f"Raw stdout (first 500): {proc.stdout[:500]}"
        ) from exc
    text = payload.get("result", "") or ""
    return text, payload


def _parse_action(assistant_text: str) -> dict[str, Any]:
    """Find the LAST action JSON object in the assistant's response."""
    matches = ACTION_RE.findall(assistant_text)
    if not matches:
        raise ValueError(
            "No action JSON found in claude response.\n"
            f"Response (first 600 chars): {assistant_text[:600]}"
        )
    last = matches[-1]
    try:
        return json.loads(last)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Found action-shaped text but it was not valid JSON: {last!r}") from exc


def _execute(action: dict[str, Any]) -> str:
    """Execute one parsed action via PyAutoGUI. Returns short result text."""
    name = action.get("action")
    if name == "click":
        x, y = int(action["x"]), int(action["y"])
        button = action.get("button") or "left"
        if action.get("double"):
            pyautogui.doubleClick(x, y, button=button)
            return f"double-click {button} at ({x},{y})"
        pyautogui.click(x, y, button=button)
        return f"click {button} at ({x},{y})"
    if name == "type":
        text = str(action["text"])
        pyautogui.typewrite(text, interval=0.015)
        preview = text[:60] + ("…" if len(text) > 60 else "")
        return f"typed {len(text)} chars: {preview!r}"
    if name == "press":
        key = str(action["key"])
        pyautogui.press(key)
        return f"pressed {key}"
    if name == "hotkey":
        keys = [str(k) for k in action["keys"]]
        pyautogui.hotkey(*keys)
        return f"hotkey {'+'.join(keys)}"
    if name == "scroll":
        amount = int(action.get("amount") or 5)
        clicks = -amount if action["direction"] == "down" else amount
        pyautogui.scroll(clicks)
        return f"scroll {action['direction']} x{amount}"
    if name == "wait":
        seconds = max(0.1, min(float(action.get("seconds") or 1), 5.0))
        time.sleep(seconds)
        return f"wait {seconds:.1f}s ({action.get('reason', '')})"
    raise ValueError(f"Unknown action: {name!r}")


def _ask_confirmation(prompt_text: str) -> bool:
    print()
    print("=" * 72)
    print(f"CONFIRM: {prompt_text}")
    print("=" * 72)
    try:
        ans = input("Allow this action? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


def run_agent(
    goal: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    confirm_all: bool = False,
    turn_timeout: int = DEFAULT_TURN_TIMEOUT_S,
) -> int:
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05

    gate = ConfidenceGate()
    abort_evt = Event()

    def _on_kill(engaged: bool) -> None:
        if engaged:
            abort_evt.set()

    kill = KillSwitch(gate, on_toggle=_on_kill)
    kill.install()

    # Show the user what we're doing.
    _, screen_size = _capture_screenshot()
    print("Backend: claude CLI (Claude Code Max subscription)")
    print(f"Screen:  {screen_size[0]}x{screen_size[1]}")
    print(f"Goal:    {goal}")
    print(f"Max iterations: {max_iterations}    confirm-all: {confirm_all}")
    print(
        "Safety: F12 = abort. Mouse-into-corner = fail-safe abort. Ctrl+C = abort."
    )
    print("Each turn ~5-15s (subprocess + Claude inference).")
    print("-" * 72)

    history: list[str] = []
    last_screenshot_path: Path | None = None
    exit_code = 0

    try:
        for iteration in range(1, max_iterations + 1):
            if abort_evt.is_set():
                print("Kill switch engaged — aborting.")
                exit_code = 130
                break

            # Clean up previous screenshot to avoid /tmp filling up.
            if last_screenshot_path is not None and last_screenshot_path.exists():
                try:
                    last_screenshot_path.unlink()
                except OSError:
                    pass

            shot_path, screen_size = _capture_screenshot()
            last_screenshot_path = shot_path
            history_text = (
                "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(history))
                if history
                else "  (none — this is the first action)"
            )
            prompt = PROMPT_TEMPLATE.format(
                goal=goal,
                screen_w=screen_size[0],
                screen_h=screen_size[1],
                image_path=str(shot_path).replace("\\", "/"),
                history=history_text,
            )

            t0 = time.perf_counter()
            try:
                assistant_text, payload = _run_claude(prompt, timeout=turn_timeout)
            except subprocess.TimeoutExpired:
                print(f"  ⚠ Claude turn exceeded {turn_timeout}s budget — aborting.")
                exit_code = 124
                break
            except RuntimeError as exc:
                print(f"  ⚠ {exc}")
                exit_code = 1
                break
            elapsed = time.perf_counter() - t0

            usage = payload.get("usage", {})
            cost = payload.get("total_cost_usd", 0.0)
            print(
                f"\n[turn {iteration:02d}/{max_iterations}] "
                f"{elapsed:.1f}s  cost~${cost:.4f}  "
                f"in={usage.get('input_tokens', 0)} "
                f"cache_read={usage.get('cache_read_input_tokens', 0)} "
                f"out={usage.get('output_tokens', 0)}"
            )
            preview = (assistant_text or "").strip().splitlines()[-1][:200] if assistant_text else ""
            if preview:
                print(f"  agent> {preview}")

            try:
                action = _parse_action(assistant_text)
            except ValueError as exc:
                print(f"  ⚠ {exc}")
                history.append("(no parseable action — retrying)")
                continue

            name = action.get("action")
            if name == "done":
                print(f"  ✅ DONE: {action.get('summary', '')}")
                break
            if name == "failed":
                reason = action.get("reason", "")
                print(f"  ❌ FAILED: {reason}")
                exit_code = 1
                break

            # Confirmation gate.
            if confirm_all:
                summary = json.dumps({k: v for k, v in action.items() if k != "action"})[:120]
                if not _ask_confirmation(f"{name}({summary})"):
                    history.append(f"{name}: USER DENIED")
                    print("  (denied — asking agent to try a different approach)")
                    continue

            try:
                result = _execute(action)
            except pyautogui.FailSafeException:
                print("  PyAutoGUI fail-safe triggered. Aborting.")
                exit_code = 130
                break
            except (ValueError, KeyError) as exc:
                print(f"  ⚠ Could not execute {action!r}: {exc}")
                history.append(f"{name}: ERROR {exc}")
                continue

            print(f"  -> {result}")
            history.append(result)

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
        # Best-effort cleanup of the last screenshot.
        if last_screenshot_path is not None and last_screenshot_path.exists():
            try:
                last_screenshot_path.unlink()
            except OSError:
                pass

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vision-based screen agent backed by the `claude` CLI (Max subscription).",
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
    parser.add_argument(
        "--turn-timeout", type=int, default=DEFAULT_TURN_TIMEOUT_S,
        help=f"Per-turn `claude -p` timeout in seconds (default {DEFAULT_TURN_TIMEOUT_S}).",
    )
    args = parser.parse_args(argv)

    # Verify claude CLI is on PATH before starting.
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True, timeout=10)  # noqa: S603, S607
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(
            f"ERROR: `claude` CLI not available or not authenticated: {exc}\n"
            "Install Claude Code and run `claude` once interactively to log in.",
            file=sys.stderr,
        )
        return 2

    return run_agent(
        goal=args.goal,
        max_iterations=args.max_iterations,
        confirm_all=args.confirm_all,
        turn_timeout=args.turn_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
