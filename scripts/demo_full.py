"""Full-scale video-recordable demo of SahaayakAI.

Runs the four headline capabilities in sequence with clear banner output,
suitable for a single screen-recording. Each phase pauses for ENTER so
you can narrate.

Phases:
    1. Intel device detection (--check)
    2. Benchmark summary (read from benchmark_report.json)
    3. Live cursor control + wink-to-click (MediaPipe FaceMesh)
    4. Phi-3-mini INT4 email simplifier on demo/sample_email.txt

Usage:
    python scripts/demo_full.py
    python scripts/demo_full.py --no-pause          # auto-advance
    python scripts/demo_full.py --skip-llm          # for quick rehearsals
    python scripts/demo_full.py --cursor-seconds 60 # longer cursor phase
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from textwrap import indent

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Force UTF-8 stdout so non-ASCII renders cleanly on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(title: str, subtitle: str = "") -> None:
    line = "=" * 72
    print()
    print(f"{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    if subtitle:
        print(f"{DIM}  {subtitle}{RESET}")
    print(f"{CYAN}{line}{RESET}")


def pause(prompt: str = "Press ENTER to continue") -> None:
    if PAUSE_ENABLED:
        try:
            input(f"\n{YELLOW}>>> {prompt}...{RESET}")
        except EOFError:
            pass
    else:
        time.sleep(1.5)


PAUSE_ENABLED = True


# --------------------------------------------------------------------- phase 1
def phase_check() -> None:
    banner("PHASE 1 — Intel device detection",
           "OpenVINO Runtime enumerates NPU, GPU, CPU on the host.")
    from sahaayak.utils.intel_device import detect_devices, format_report

    report = detect_devices()
    print()
    print(format_report(report))


# --------------------------------------------------------------------- phase 2
def phase_benchmark() -> None:
    banner("PHASE 2 — Real per-device latency",
           "Iris pipeline benchmarked on every available Intel device.")
    bench_path = REPO / "benchmark_report.json"
    if not bench_path.exists():
        print(f"{DIM}(benchmark_report.json not found — run scripts/bench_iris.py first){RESET}")
        return
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    print(f"\n  OpenVINO runtime : {data.get('runtime_version', '?')}")
    print(f"  {'Device':<14} {'Mean':>8} {'p95':>8} {'p99':>8} {'FPS':>8}")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for rec in data.get("records", []):
        s = rec.get("summary", {})
        print(f"  {GREEN}{rec['device']:<14}{RESET} "
              f"{s.get('mean_ms', 0):>6.2f}ms "
              f"{s.get('p95_ms', 0):>6.2f}ms "
              f"{s.get('p99_ms', 0):>6.2f}ms "
              f"{s.get('fps', 0):>6.1f}")
    if data.get("notes"):
        print(f"\n  {DIM}{data['notes']}{RESET}")


# --------------------------------------------------------------------- phase 3
def phase_cursor(seconds: float) -> None:
    banner("PHASE 3 — Live gaze cursor + wink-to-click",
           f"MediaPipe FaceMesh @ ~28 FPS. {seconds:.0f}s of live control.")
    print(f"\n  {YELLOW}- Move your head to steer the cursor{RESET}")
    print(f"  {YELLOW}- Wink LEFT  eye to LEFT click{RESET}")
    print(f"  {YELLOW}- Wink RIGHT eye to RIGHT click{RESET}")
    print(f"  {YELLOW}- F12 = kill switch  /  mouse-into-corner = FAIL-SAFE{RESET}")
    if PAUSE_ENABLED:
        try:
            input(f"\n{YELLOW}>>> Press ENTER when you're ready to demo...{RESET}")
        except EOFError:
            pass
    cmd = [
        sys.executable, str(REPO / "scripts" / "demo_cursor_mp.py"),
        "--seconds", str(int(seconds)),
        "--gain", "1.5",
    ]
    subprocess.run(cmd, check=False)  # noqa: S603 - args composed from constants


# --------------------------------------------------------------------- phase 4
def phase_llm() -> None:
    banner("PHASE 4 — Phi-3-mini INT4 email co-pilot",
           "Local LLM via OpenVINO GenAI on the Intel NPU. Zero cloud calls.")
    from sahaayak.copilot.email_simplifier import EmailSimplifier
    from sahaayak.copilot.llm_engine import LLMEngine
    from sahaayak.utils.config import load_config

    body = (REPO / "demo" / "sample_email.txt").read_text(encoding="utf-8")
    print(f"\n{DIM}Input email ({len(body)} chars):{RESET}")
    print(indent(body[:600] + ("..." if len(body) > 600 else ""), "  | "))

    print(f"\n{DIM}Loading Phi-3-mini INT4 (cached compile after first run)...{RESET}")
    engine = LLMEngine(config=load_config())
    simp = EmailSimplifier(engine)
    t0 = time.perf_counter()
    out = simp.simplify(body)
    elapsed = time.perf_counter() - t0
    print(f"{GREEN}Generated in {elapsed:.1f}s on {engine.device}.{RESET}\n")

    print(f"{BOLD}TL;DR{RESET}")
    print(indent(out.tldr or "(none)", "  "))

    print(f"\n{BOLD}Key points{RESET}")
    for kp in out.key_points:
        print(f"  - {kp}")

    print(f"\n{BOLD}Action items{RESET}")
    for ai in out.action_items:
        if not ai.task:
            continue
        deadline = f" ({ai.deadline})" if ai.deadline else ""
        print(f"  - {ai.task}{deadline}")

    print(f"\n{BOLD}Tone{RESET}: {out.tone}")

    print(f"\n{BOLD}Suggested reply drafts{RESET}  {DIM}(AI-generated, please review){RESET}")
    for i, draft in enumerate(out.suggested_reply_drafts, 1):
        print(f"  [{i}] {draft}")


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description="SahaayakAI full-scale video demo.")
    parser.add_argument("--no-pause", action="store_true",
                        help="Auto-advance through phases (for unattended runs).")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip the Phi-3 phase (faster rehearsals).")
    parser.add_argument("--skip-cursor", action="store_true",
                        help="Skip the live cursor phase.")
    parser.add_argument("--cursor-seconds", type=float, default=45.0,
                        help="Length of the live cursor demo phase (default 45s).")
    args = parser.parse_args()

    global PAUSE_ENABLED  # noqa: PLW0603
    PAUSE_ENABLED = not args.no_pause

    print()
    print(f"{BOLD}{CYAN}  ╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}  ║   SahaayakAI — offline laptop co-pilot for accessibility        ║{RESET}")
    print(f"{BOLD}{CYAN}  ║   Intel AI for Future Workforce hackathon submission demo       ║{RESET}")
    print(f"{BOLD}{CYAN}  ╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}  100% offline. No webcam frame, audio chunk, or text leaves this laptop.{RESET}")
    pause("Press ENTER to begin")

    phase_check()
    pause()

    phase_benchmark()
    pause()

    if not args.skip_cursor:
        phase_cursor(args.cursor_seconds)
        pause()

    if not args.skip_llm:
        phase_llm()

    banner("DEMO COMPLETE",
           "Repo: sahaayak_ai/   Docs: docs/{ARCHITECTURE,RESPONSIBLE_AI,INTEL_STACK_RATIONALE}.md")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
