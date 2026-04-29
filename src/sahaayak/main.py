"""SahaayakAI entry point.

Subcommands:

* ``--check``        — print OpenVINO device report and exit.
* ``--calibrate``    — run the 9-point calibration UI.
* ``--run``          — full eye-control + co-pilot session.
* ``--simplify FILE``— run the email simplifier on a text file (LLM smoke test).
* ``--benchmark``    — measure per-device inference latency, write JSON.
"""

from __future__ import annotations

import argparse
import sys

# Force UTF-8 stdout/stderr so non-ASCII glyphs render on Windows cp1252.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

from pathlib import Path  # noqa: E402

from sahaayak import __version__  # noqa: E402
from sahaayak.utils.intel_device import detect_devices, format_report  # noqa: E402
from sahaayak.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sahaayak",
        description="Offline, hands-free, neurodivergent-friendly laptop co-pilot.",
    )
    parser.add_argument("--version", action="version", version=f"SahaayakAI {__version__}")
    parser.add_argument("--check", action="store_true", help="Probe OpenVINO devices and exit.")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Launch the 9-point calibration UI and save the profile.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Start the full eye-control + co-pilot session.",
    )
    parser.add_argument(
        "--simplify", metavar="FILE", default=None,
        help="Run the email simplifier on a text file and print the JSON.",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run a synthetic latency benchmark and write benchmark_report.json.",
    )
    parser.add_argument(
        "--backend", choices=("mediapipe", "openvino"), default="mediapipe",
        help="Vision backend for --run/--calibrate. MediaPipe (default) is "
             "more reliable end-to-end; OpenVINO uses our hand-rolled "
             "Iris-IR + OpenCV-Haar pipeline for the spec-strict path.",
    )
    parser.add_argument(
        "--app", action="store_true",
        help="Launch the integrated desktop app: floating button + iris cursor "
             "+ Claude-CLI agent runner.",
    )
    return parser


def _cmd_check() -> int:
    report = detect_devices()
    sys.stdout.write(format_report(report) + "\n")
    if not report.available:
        logger.error("Device probe unavailable: %s", report.error)
        return 1
    logger.info("Device check completed successfully.")
    return 0


def _make_tracker(backend: str, config: dict[str, object]) -> object:
    """Construct the requested perception backend."""
    if backend == "mediapipe":
        from sahaayak.core.mediapipe_tracker import MediaPipeTracker  # noqa: PLC0415
        return MediaPipeTracker(config=config)
    from sahaayak.core.eye_tracker import EyeTracker  # noqa: PLC0415
    return EyeTracker(config=config)


def _cmd_calibrate(backend: str) -> int:
    from sahaayak.ui.calibration_window import run_calibration_ui  # noqa: PLC0415
    from sahaayak.utils.config import load_config  # noqa: PLC0415

    config = load_config()
    tracker = _make_tracker(backend, config)
    stream = tracker.stream()

    def _next() -> object | None:
        try:
            _frame, result = next(stream)
            return result
        except StopIteration:
            return None

    profile = run_calibration_ui(_next, config)
    if profile is None:
        logger.error("Calibration cancelled or failed.")
        return 1
    logger.info("Calibration complete. Homography saved.")
    return 0


def _cmd_run(backend: str) -> int:
    from sahaayak.control.action_dispatcher import ActionDispatcher  # noqa: PLC0415
    from sahaayak.control.cursor_controller import CursorController  # noqa: PLC0415
    from sahaayak.control.eye_keyboard import EyeKeyboard  # noqa: PLC0415
    from sahaayak.copilot.focus_nudger import FocusNudger  # noqa: PLC0415
    from sahaayak.core.calibrator import CalibrationProfile  # noqa: PLC0415
    from sahaayak.core.gaze_estimator import GazeEstimator  # noqa: PLC0415
    from sahaayak.core.gesture_engine import GestureEngine  # noqa: PLC0415
    from sahaayak.safety.audit_log import AuditLog  # noqa: PLC0415
    from sahaayak.safety.confidence_gate import ConfidenceGate  # noqa: PLC0415
    from sahaayak.safety.kill_switch import KillSwitch  # noqa: PLC0415
    from sahaayak.ui.consent_dialog import show_consent_dialog  # noqa: PLC0415
    from sahaayak.ui.overlay import GazeOverlay  # noqa: PLC0415
    from sahaayak.ui.rest_reminder import RestReminder  # noqa: PLC0415
    from sahaayak.utils.config import load_config  # noqa: PLC0415

    if not show_consent_dialog():
        logger.info("Consent declined; exiting.")
        return 0

    config = load_config()
    audit = AuditLog(config.get("safety", {}).get("audit_log_path", "./audit_log.json"))
    audit.append("session_start", {"version": __version__, "backend": backend})

    tracker = _make_tracker(backend, config)
    cursor = CursorController(
        gate=ConfidenceGate(
            min_confidence=config.get("eye_tracker", {}).get("min_confidence", 0.6),
            grace_ms=config.get("eye_tracker", {}).get("low_confidence_grace_ms", 500),
        ),
        smoothing_steps=config.get("cursor", {}).get("smoothing_steps", 4),
    )
    screen_size = cursor.screen_size()
    profile = CalibrationProfile.load()
    gaze = GazeEstimator(
        screen_size=screen_size,
        config=config,
        homography=profile.homography if profile else None,
    )
    gestures = GestureEngine(config=config)
    keyboard = EyeKeyboard(cursor, config=config)
    overlay = GazeOverlay(screen_size)
    overlay.start()

    rest = RestReminder(config=config)
    rest.start()

    def _on_nudge() -> None:
        audit.append("focus_nudge")
        logger.info("[NUDGE] %s", "switching tasks frequently")

    nudger = FocusNudger.from_config(_on_nudge, config)

    kill = KillSwitch(cursor._gate)  # noqa: SLF001 - shared gate
    kill.install()

    dispatcher = ActionDispatcher(
        cursor,
        on_toggle_keyboard=keyboard.toggle_visible,
    )

    try:
        for _frame, result in tracker.stream():
            nudger.step()
            if result is None:
                continue
            point = gaze.estimate(result)
            cursor._gate.update(point.confidence, point.timestamp)  # noqa: SLF001
            overlay.update_position(point)
            cursor.move(point.x, point.y)
            for g in gestures.step(result, point, screen_size):
                dispatcher.dispatch(g)
                audit.append(f"gesture.{g.kind.value}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        kill.uninstall()
        rest.stop()
        overlay.stop()
        audit.append("session_end")
    return 0


def _cmd_simplify(path: str) -> int:
    from sahaayak.copilot.email_simplifier import EmailSimplifier  # noqa: PLC0415
    from sahaayak.copilot.llm_engine import LLMEngine  # noqa: PLC0415
    from sahaayak.utils.config import load_config  # noqa: PLC0415

    body = Path(path).read_text(encoding="utf-8")
    engine = LLMEngine(config=load_config())
    simplifier = EmailSimplifier(engine)
    out = simplifier.simplify(body)
    import json  # noqa: PLC0415

    sys.stdout.write(json.dumps(out.to_dict(), indent=2, ensure_ascii=False) + "\n")
    return 0


def _cmd_benchmark() -> int:
    from sahaayak.utils.benchmark import (  # noqa: PLC0415
        BenchmarkRecord,
        BenchmarkReport,
        LatencyMeter,
    )

    report_path = Path("benchmark_report.json")
    devices = detect_devices()
    rep = BenchmarkReport(runtime_version=devices.runtime_version)

    # The synthetic benchmark just verifies the timing harness end-to-end.
    # Real per-device numbers populate when models exist.
    meter = LatencyMeter()
    for _ in range(50):
        with meter.measure():
            _ = sum(i * i for i in range(1000))
    rep.records.append(
        BenchmarkRecord(
            component="sanity",
            device="CPU (Python loop)",
            model="N/A",
            summary=meter.summary(),
        )
    )
    rep.notes = (
        "Run with iris/Phi-3 IR present to populate per-device records. "
        "See docs/INTEL_STACK_RATIONALE.md for the methodology."
    )
    rep.write(report_path)
    sys.stdout.write(f"Wrote {report_path}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.check:
        return _cmd_check()
    if args.calibrate:
        return _cmd_calibrate(args.backend)
    if args.run:
        return _cmd_run(args.backend)
    if args.simplify:
        return _cmd_simplify(args.simplify)
    if args.benchmark:
        return _cmd_benchmark()
    if args.app:
        from sahaayak.desktop_app import main as app_main  # noqa: PLC0415
        return app_main()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
