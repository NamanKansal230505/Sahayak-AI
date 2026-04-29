<div align="center">

# SahaayakAI

### Control your laptop with your eyes. Let an AI handle the rest.

*An accessibility-first Windows desktop assistant for users who cannot — or prefer not to — use a mouse and keyboard.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![OpenVINO 2024.4](https://img.shields.io/badge/OpenVINO-2024.4-0071C5.svg)](https://docs.openvino.ai/)
[![Intel Core Ultra](https://img.shields.io/badge/Intel-Core%20Ultra%20%7C%20NPU%20%7C%20Arc%20iGPU-0071C5.svg)](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen.svg)](#testing)
[![Hackathon](https://img.shields.io/badge/Intel-AI%20For%20Future%20Workforce-0071C5.svg)](#)

</div>

---

## Table of Contents

- [Why SahaayakAI](#why-sahaayakai)
- [What it does](#what-it-does)
- [Demo](#demo)
- [Architecture](#architecture)
- [Intel hardware stack](#intel-hardware-stack)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Safety & privacy](#safety--privacy)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Author](#author)

---

## Why SahaayakAI

> *Sahaayak* (सहायक) means **helper** in Hindi.

Hundreds of millions of people are excluded from comfortable laptop use:

- People living with motor impairment, RSI, post-stroke recovery, ALS, or limb difference.
- Neurodivergent users (ADHD, autism, dyslexia) overwhelmed by long emails, dense PDFs, and constant context-switching.
- Anyone who simply wants to keep their hands free.

Commercial eye-tracking hardware costs **₹40,000+**. Cloud-based AI assistants ship your screen pixels to a server you do not control. SahaayakAI removes both barriers: a normal webcam plus an Intel-powered laptop are enough.

---

## What it does

| Capability | How it works |
| --- | --- |
| 🎯 **Iris cursor** | Real-time gaze tracking via MediaPipe FaceMesh on a background thread. Head-as-joystick mode, plus dwell- and gaze-zone scrolling. |
| 😉 **Click without a mouse** | Wink-left / wink-right detection drives left- and right-click. F12 is a hardware kill switch. |
| 🤖 **Screen agent** | A natural-language agent that can *see* the screen (screenshot → vision-LLM with tools → PyAutoGUI → repeat). Three interchangeable backends: |
| | • **Claude CLI** subprocess — uses your existing Claude Max subscription (preferred) |
| | • **Anthropic API** — best quality, ~$0.30/run |
| | • **Groq Llama 4 Scout** — free tier, ~$0.002/run |
| 🎙️ **Voice goal entry** | A double-blink launches local speech-to-text via `faster-whisper`. You say what you want; the agent does it. |
| 🪟 **Floating launcher** | A small always-on-top 🤖 button. Tap it (or double-blink) to invoke the agent. The status panel streams the agent's reasoning live. |

Everything except the optional Anthropic/Groq backends runs **locally** on your laptop's NPU + iGPU + CPU.

---

## Demo

```
┌─────────────────────────────────────────────────────────────────┐
│   👁  Webcam → MediaPipe Iris → Kalman → cursor (30 FPS)        │
│                                                                 │
│   😉  wink-left → click       😉  wink-right → right-click      │
│   🪟  double-blink → 🎙 speech-to-text → agent goal             │
│   🛑  F12 → release all input control immediately               │
└─────────────────────────────────────────────────────────────────┘
```

Demo recordings and the full screen-agent walkthrough live in [`demo/`](demo/).

---

## Architecture

```
                          ┌──────────────────────────────┐
                          │         Webcam frame         │
                          └──────────────┬───────────────┘
                                         ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Perception      MediaPipe Iris ─▶ Kalman ─▶ gesture engine    │
   │                  (background thread, 30 FPS, NPU/GPU/CPU)       │
   └────────────────┬────────────────────────────────────────────────┘
                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Safety          confidence gate ─▶ kill-switch (F12) ─▶ audit  │
   └────────────────┬────────────────────────────────────────────────┘
                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Control         cursor_controller / on-screen keyboard         │
   └────────────────┬────────────────────────────────────────────────┘
                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  Agent           screenshot ─▶ vision LLM (tools) ─▶ PyAutoGUI  │
   │                  backends: Claude CLI │ Anthropic │ Groq        │
   └─────────────────────────────────────────────────────────────────┘
```

A more detailed component diagram is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Intel hardware stack

Built and demoed on an **Intel Core Ultra 7 258V** laptop (NPU + Arc 140V iGPU + LPE-Cores).

| Layer | Component | Used For |
| --- | --- | --- |
| Hardware | Intel Core Ultra 7 258V | NPU (Intel AI Boost) + Arc 140V iGPU (16 GB) + CPU |
| Runtime | OpenVINO 2024.4 | Sees and exposes all three accelerators |
| Vision | MediaPipe FaceMesh / Iris | Gaze, blink, wink detection |
| Speech | `faster-whisper` (CTranslate2) | Local STT, runs in a subprocess |
| Orchestration | `ov.Core(device="AUTO:NPU,GPU,CPU")` | Heterogeneous device routing |

**Iris pipeline benchmark on this laptop:**

| Device | FPS |
| --- | --- |
| CPU | **244** |
| NPU | 177 |
| GPU | 147 |

CPU wins on this small model because kernel-launch overhead dominates. Full rationale in [`docs/INTEL_STACK_RATIONALE.md`](docs/INTEL_STACK_RATIONALE.md).

---

## Quickstart

### Prerequisites
- Windows 11
- Python **3.11** (3.12 is not supported because of `mediapipe`)
- A working webcam
- *(Optional)* an Anthropic API key, Groq API key, or the `claude` CLI on `PATH`

### Install

```bash
# 1. Clone
git clone https://github.com/<your-username>/sahaayak-ai.git
cd sahaayak-ai

# 2. Virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Verify Intel device detection
python -m sahaayak.main --check
```

Expected output of `--check`:

```
SahaayakAI 0.1.0  --  device check
OpenVINO runtime : 2024.4.0
Available devices:
  - CPU            (Intel(R) Core(TM) Ultra 7 258V)
  - GPU.0          (Intel(R) Arc(TM) 140V)
  - NPU            (Intel(R) AI Boost)
All checks passed.
```

### Set API keys *(only if you want the cloud agent backends)*

```powershell
# PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GROQ_API_KEY      = "gsk_..."
```

The Claude-CLI backend needs no API key — it shells out to a `claude` binary that uses your existing Max subscription.

---

## Usage

### Launch the integrated desktop app *(default)*

```bash
python -m sahaayak.main --app
```

A floating 🤖 button appears. Iris tracking starts in the background. Click the button — or double-blink hands-free — to start the screen agent.

### Standalone modes

| Command | What it does |
| --- | --- |
| `python -m sahaayak.main --check` | Probe OpenVINO devices and exit |
| `python -m sahaayak.main --calibrate` | Run the 9-point cursor calibration |
| `python scripts/demo_cursor_mp.py` | Iris cursor only, no agent |
| `python scripts/agent_claude.py "open notepad and type hello"` | Screen agent via Claude CLI |
| `python scripts/agent_groq.py  "..."` | Screen agent via Groq Llama 4 Scout |
| `python scripts/agent.py       "..."` | Screen agent via Anthropic API |
| `python scripts/whisper_transcribe.py` | Standalone microphone → text |
| `python scripts/bench_iris.py` | Per-device iris-pipeline benchmark |

> 💡 **F12** instantly releases mouse and keyboard control — keep this in mind if the cursor ever feels "stuck on you".

---

## Project layout

```
sahaayak_ai/
├── config/              # Tunables (default.yaml). User calibration is gitignored.
├── docs/                # ARCHITECTURE, INTEL_STACK_RATIONALE, RESPONSIBLE_AI
├── demo/                # Sample inputs and recording guide
├── models/              # download_models.py — IR weights are gitignored
├── scripts/             # Runnable demos, agents, benchmarks
│   ├── agent.py             # Anthropic API agent loop
│   ├── agent_claude.py      # Claude-CLI subprocess agent loop
│   ├── agent_groq.py        # Groq Llama 4 Scout agent loop
│   ├── bench_iris.py        # Per-device iris benchmark
│   ├── demo_cursor_mp.py    # Iris cursor only
│   ├── demo_full.py         # Cursor + agent
│   └── whisper_transcribe.py
├── src/sahaayak/
│   ├── core/            # eye_tracker, mediapipe_tracker, gaze_estimator,
│   │                    # gesture_engine, calibrator, kalman_filter
│   ├── control/         # cursor_controller, eye_keyboard, action_dispatcher
│   ├── copilot/         # llm_engine, email_simplifier, doc_visualizer,
│   │                    # meeting_summarizer, focus_nudger, prompts/
│   ├── ui/              # overlay, copilot_panel, calibration_window,
│   │                    # consent_dialog, rest_reminder
│   ├── safety/          # confidence_gate, kill_switch (F12), audit_log
│   ├── utils/           # intel_device, logger, config, i18n, benchmark
│   ├── desktop_app.py   # Integrated floating-button + agent UI
│   └── main.py          # CLI entry point
└── tests/               # 69+ pytest tests (no real user data in fixtures/)
```

---

## Configuration

All tunables live in [`config/default.yaml`](config/default.yaml):

- Cursor sensitivity, smoothing, dwell time
- Wink/blink thresholds and debounce windows
- Confidence gate thresholds
- Rest-reminder cadence (default: 20-20-20)

Per-user calibration is written to `config/calibration_profile.yaml`, which is **gitignored** so your eye geometry never leaves your machine.

---

## Safety & privacy

SahaayakAI was originally pitched as a 100%-offline application; that constraint has been relaxed for the cloud agent backends, but **everything else still runs locally**. Concretely:

1. **Webcam frames never touch disk.** Frames are processed in-memory and discarded.
2. **Iris geometry is never persisted** — only an aggregate calibration polynomial.
3. **Confidence gate.** The gaze cursor freezes the moment tracking confidence drops, rather than drifting wildly.
4. **Hardware kill switch.** `F12` instantly releases all mouse/keyboard control.
5. **20-20-20 rest reminder.** Every 20 minutes, look at something 20 feet away for 20 seconds.
6. **DPDP Act 2023 consent dialog** on first launch.
7. **Cloud agent backends are opt-in.** No API key → no network calls.

The complete responsible-AI checklist lives in [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md).

---

## Testing

```bash
# Run the full suite
pytest

# Lint
ruff check .

# Format-check (no auto-write)
ruff format --check .
```

The suite uses synthetic fixtures only — no real webcam frames or user audio. 69+ tests pass on the demo machine.

---

## Benchmarks

A reproducible per-device benchmark of the iris pipeline:

```bash
python scripts/bench_iris.py
```

This writes `benchmark_report.json` (gitignored) with FPS and tail latencies for CPU, GPU, and NPU. The numbers above were produced on the demo Intel Core Ultra 7 258V.

---

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| 1 | Scaffold + Intel device detection | ✅ |
| 2 | Eye tracker (MediaPipe Iris on OpenVINO) | ✅ |
| 3 | Calibration + Kalman smoothing | ✅ |
| 4 | Gestures + cursor control + F12 kill switch | ✅ |
| 5 | On-screen eye keyboard (EN + HI) | ✅ |
| 6 | Screen agent (3 backends) + voice goal entry | ✅ |
| 7 | Integrated desktop app | ✅ |
| 8 | Public submission package + demo video | 🚧 |

---

## Acknowledgements

- **Intel** — OpenVINO, OpenVINO GenAI, AI for Future Workforce program
- **Google** — MediaPipe FaceMesh / Iris
- **Anthropic** — Claude (the agent loop's brain)
- **Groq** — Llama 4 Scout free-tier inference
- **OpenAI / SYSTRAN** — Whisper / `faster-whisper`

---

## License

[Apache License 2.0](LICENSE) — free for personal, academic, and commercial use, with attribution.

---

## Author

**Naman Kansal**
Built for the Intel **AI for Future Workforce** hackathon.

> Made with ❤️ on an Intel Core Ultra 7 258V.
