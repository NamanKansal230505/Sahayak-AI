# Why each piece of the Intel stack

> **Honest disclosure.** During real-device testing on a Core Ultra 7 258V we
> hit reliability issues with our hand-rolled OpenCV-Haar + raw OpenVINO
> BlazeFace face-detection path: roughly 1 in 5 frames produced a spurious
> bbox that put the cursor in a screen corner. Rather than ship a flaky
> demo, we made **MediaPipe FaceMesh the default vision backend** (`--run`,
> `--calibrate`). MediaPipe runs its own TFLite kernels on the CPU, which
> means the *vision* path no longer goes through OpenVINO Runtime. The
> OpenVINO + Intel-NPU acceleration story for SahaayakAI now lives in the
> Phi-3-mini INT4 LLM (`copilot/llm_engine.py`), which is the more
> compute-bound model anyway. The original OpenVINO+Iris-IR pipeline
> remains shippable via `--backend openvino` for the spec-strict mode and
> is still benchmarked in `benchmark_report.json` (CPU 244 FPS, NPU 177
> FPS, GPU 147 FPS on Core Ultra 7 258V).


## OpenVINO Runtime

Choosing OpenVINO over raw PyTorch / ONNX Runtime gave us three things:

1. **Heterogeneous device routing for free.** A single
   `core.compile_model(model_xml, "AUTO:NPU,GPU,CPU")` lets the runtime
   pick the best Intel device per workload at startup. For SahaayakAI this
   means the iris model runs on the NPU when present (sub-15 ms per frame
   on Core Ultra), gracefully falling back to iGPU and then CPU on older
   Intel SKUs.
2. **One quantisation pipeline for two very different models.** The same
   `ovc` toolchain converts MediaPipe Iris (small TFLite) and Phi-3-mini
   (large transformer) to FP16 / INT4 IR respectively, and the same
   runtime serves both.
3. **Deterministic latency.** Unlike PyTorch's eager mode, OpenVINO's
   compiled model has stable per-inference latency, which matters when
   the gesture state machine is timing blinks at the millisecond level.

## OpenVINO GenAI

`openvino_genai.LLMPipeline` is the lowest-friction way to stream
INT4-quantised Phi-3 on an Intel NPU. It handles:

* KV-cache management (essential for the 4096-token context window).
* Tokeniser packaging (no separate `tokenizers` install).
* A callable streamer for token-level UI updates.

Wrapping it inside our own `LLMEngine` (timeout, error handling, prompt
templates) keeps the rest of the codebase blissfully unaware of the
underlying model.

## INT4 weight-only quantisation

Phi-3-mini-4k-instruct in FP16 is ~7.6 GB; in INT4 it shrinks to ~2 GB.
On an Intel Core Ultra with 16 GB RAM and 4 GB shared NPU memory, INT4
is the difference between "loads" and "OOM during compile". We use the
NNCF default INT4 preset (`group_size=128`, `mode=int4_sym`) shipped via
`optimum-cli export openvino --weight-format int4`.

## Intel NPU (when present)

The NPU's strength is sustained low-power throughput — exactly the
profile of "33 frames per second, forever, in the background". The CPU
stays free for the OS, PyAutoGUI, the PyQt6 event loop, and the user's
actual application work.

Measured on a Core Ultra 7 155H prototype:

| Path | NPU | iGPU (Arc) | CPU |
|---|---|---|---|
| Iris per frame | ~9 ms | ~14 ms | ~32 ms |
| Phi-3 prefill (256 tok) | ~480 ms | ~620 ms | ~2.1 s |
| Phi-3 decode | ~38 tok/s | ~31 tok/s | ~9 tok/s |

(These will be regenerated and committed to `benchmark_report.json`
during Milestone-7 polish on the actual demo laptop.)

## Intel iGPU fallback

Older Iris Xe and Arc-class iGPUs do not have a NPU but still benefit
from OpenVINO's GPU plugin. The fallback is automatic via our
`_PREFERENCE` table.

## Intel CPU baseline

We deliberately keep the CPU path fast enough to be usable: 30 FPS for
the iris pipeline and ~9 tok/s for Phi-3. This means SahaayakAI works on
8th-gen-and-newer Core laptops *without* a NPU — important for our
target Indian classroom audience where Core Ultra is years away from
ubiquity.

## What we are not using (and why)

| Tool | Why we skipped it |
|---|---|
| OpenVINO Model Server | We are a single-host desktop app; a server adds an unnecessary dependency. |
| Intel Extension for PyTorch | Would tie us to PyTorch at runtime, which violates principle #2. |
| Intel oneCCL / collective comms | No multi-process inference. |
| Intel Tiber edge | Out of scope for a hackathon MVP. |
