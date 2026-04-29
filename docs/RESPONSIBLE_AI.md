# Responsible AI — SahaayakAI

SahaayakAI is built for users with motor, cognitive, and visual
differences. The cost of a wrong action is high — a missed click in a
banking app, an auto-sent email, a misclassified gesture. Our principles
are below; each is enforced by a specific code path.

## 1. Offline-first

| Measure | Code |
|---|---|
| No HTTP client at runtime | grep across `src/` for `requests`, `urllib`, `httpx` returns nothing. |
| Inference paths use OpenVINO only | `EyeTracker`, `LLMEngine` import `openvino` / `openvino_genai`; we ban raw PyTorch / ONNX at runtime. |
| Model assets fetched once, locally | `models/download_models.py` is the only network-touching script and is gated behind manual invocation. |

## 2. Privacy-by-design

| Measure | Code |
|---|---|
| Frames never written | `eye_tracker.py` has no `cv2.imwrite` / `Path.write_bytes`. Test `test_eye_tracker_no_disk_writes` greps the source. |
| Iris embeddings never stored | `gaze_estimator.py` discards the eyelid landmark vector after computing `(x, y)`. The calibration profile saves only the 3x3 homography. |
| Audit log is local-only | `AuditLog.__init__` raises `ValueError` if the configured path looks like a URL. Forbidden keys (`frame`, `iris_embedding`, `audio`, `transcript`) are scrubbed in `_scrub`. |
| Window titles hashed | `focus_nudger.py` only stores 8-byte SHA-256 digests of titles, never the title text. |
| Microphone optional | `meeting_summarizer.py` is opt-in; mic input is never recorded silently. |

## 3. Fail-safe

| Measure | Code |
|---|---|
| Confidence gate | `ConfidenceGate` blocks all cursor actions after >500 ms below 0.6 confidence. |
| Cursor smoothing | `CursorController` interpolates over 4 steps so a runaway gaze never teleports. |
| LLM hard timeout | `LLMEngine.generate` runs in a worker thread with a 30 s join timeout; raises `LLMTimeoutError` rather than blocking the UI forever. |

## 4. Kill switch

`F12` is bound globally via `pynput`; toggling it forces the
`ConfidenceGate` shut and shows the `kill_switch.engaged` i18n string. The
hotkey is intentionally **not** configurable — preventing the user (or a
malicious script) from disabling the safety control.

## 5. Mandatory eye-rest

`RestReminder` fires every 20 minutes by default, lasts 20 seconds, and
**reschedules unconditionally** after each fire. The maximum snooze is 60
minutes, hard-coded in `_max_snooze`. There is no API to set
`_stopped = True` other than full app shutdown.

## 6. Consent + DPDP Act 2023

`ConsentDialog` shows a translated DPDP notice on first launch
(`i18n("consent.dpdp_notice")`). Acceptance is recorded once in
`~/.sahaayak/consent.json`; re-launches honour the choice. Decline exits
the application immediately.

## 7. AI disclosure

Every LLM-rendered string in the `CopilotPanel` carries
`i18n("copilot.disclaimer")` ("AI-generated, please review.") at the
bottom of the panel. Drafts require an explicit long-blink (or a button
click while gaze is still on the panel) to send.

## 8. Bias + accessibility testing

| Concern | Mitigation |
|---|---|
| Skin-tone bias in face detection | We rely on MediaPipe BlazeFace, which is independently evaluated across skin tones. We document the FairFace-style synthetic test set under `tests/fixtures/` (Milestone 7+). |
| Spectacle wearers | Iris model handles glasses; calibration absorbs per-user offsets. |
| Kohl / kajal users | Same — calibration corrects baseline darkness. |
| One-eyed users | `Calibrator` auto-detects monocular use (`monocular_auto_detect`) and switches to single-eye mode. |
| Colour blindness | `GazeOverlay` reticle uses a green↔red ramp, but also draws shape change at low confidence. |
| Hindi-first audience | All UI strings flow through `i18n()` from day 1; English/Hindi translations live in `utils/i18n.py`. |

## What we explicitly do *not* do

* Auto-send messages or commit destructive actions on a single blink.
* Persist or transmit any biometric template.
* Telemetry, analytics, crash-reporting SDKs.
* Auto-update mechanisms (would imply network, would imply trust shift).
* Cloud LLMs / cloud STT / cloud anything.
