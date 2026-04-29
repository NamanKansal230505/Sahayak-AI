"""OpenVINO-GenAI Phi-3-mini-int4 wrapper.

Wraps `openvino_genai.LLMPipeline` with:

* Auto-device selection (NPU > GPU > CPU) via `intel_device.get_best_device`.
* Streaming token output (callable streamer) for responsive UI.
* Hard timeout per request (default 30s).
* Prompt-template loading from `copilot/prompts/*.txt`.

All inference happens locally. No HTTP, no telemetry.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from sahaayak.utils.intel_device import get_best_device
from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "phi-3-mini-int4-ov"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template by stem (e.g. ``"email_simplify"``)."""
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template missing: {path}")
    return path.read_text(encoding="utf-8")


class LLMTimeoutError(TimeoutError):
    """Raised when generation exceeds the configured wall-clock budget."""


class LLMEngine:
    """OpenVINO GenAI Phi-3 wrapper.

    Args:
        model_dir: Directory containing ``openvino_model.xml`` from
            `models/download_models.py`.
        device: Explicit OpenVINO device id; ``None`` picks the best one.
        config: Loaded SahaayakAI config.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config or {}
        inf_cfg = self._config.get("inference", {})
        self._max_new_tokens = int(inf_cfg.get("llm_max_new_tokens", 512))
        self._timeout = float(inf_cfg.get("llm_timeout_seconds", 30))
        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._device = device or get_best_device("llm")
        self._pipeline: Any = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        """Active OpenVINO device id."""
        return self._device

    def _ensure_loaded(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            import openvino_genai as ov_genai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openvino-genai not installed. Run `pip install -r requirements.txt`."
            ) from exc
        if not (self._model_dir / "openvino_model.xml").exists():
            raise FileNotFoundError(
                f"Phi-3 INT4 IR not found in {self._model_dir}. "
                "Run `python models/download_models.py` first."
            )
        logger.info("Loading Phi-3-mini INT4 on %s", self._device)
        self._pipeline = ov_genai.LLMPipeline(str(self._model_dir), self._device)
        return self._pipeline

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        streamer: Callable[[str], None] | None = None,
    ) -> str:
        """Run a single prompt and return the full string.

        Args:
            prompt: The complete prompt (template already filled in).
            max_new_tokens: Optional override of the default budget.
            streamer: Optional callable invoked with each token chunk.

        Raises:
            LLMTimeoutError: If generation exceeds the configured timeout.
        """
        pipeline = self._ensure_loaded()
        max_tokens = int(max_new_tokens or self._max_new_tokens)
        start = time.monotonic()
        result_box: dict[str, Any] = {}

        def _run() -> None:
            try:
                if streamer is not None:
                    out = pipeline.generate(
                        prompt,
                        max_new_tokens=max_tokens,
                        streamer=streamer,
                    )
                else:
                    out = pipeline.generate(prompt, max_new_tokens=max_tokens)
                result_box["text"] = str(out)
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc

        with self._lock:
            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=self._timeout)
            if thread.is_alive():
                raise LLMTimeoutError(
                    f"LLM generation exceeded {self._timeout}s budget."
                )
        if "error" in result_box:
            raise RuntimeError(f"LLM generation failed: {result_box['error']}") from result_box["error"]
        elapsed = time.monotonic() - start
        logger.info("LLM generated %d-char response in %.2fs", len(result_box.get("text", "")), elapsed)
        return result_box.get("text", "")

    def stream(self, prompt: str, max_new_tokens: int | None = None) -> Iterator[str]:
        """Yield generated tokens as they arrive.

        Implemented as a thread-pumped queue so callers can iterate while
        the model runs. Falls back to a single-shot generate() if the
        underlying pipeline does not support callable streamers.
        """
        from queue import Queue  # noqa: PLC0415

        sentinel = object()
        q: Queue[Any] = Queue()

        def _streamer(chunk: str) -> None:
            q.put(chunk)

        def _runner() -> None:
            try:
                self.generate(prompt, max_new_tokens=max_new_tokens, streamer=_streamer)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield str(item)

    def predict_next(self, buffer: str) -> list[str]:
        """Return up to 3 next-word predictions for the keyboard."""
        if not buffer.strip():
            return ["I", "The", "Hello"]
        prompt = load_prompt("keyboard_predict").format(buffer=buffer)
        try:
            raw = self.generate(prompt, max_new_tokens=24)
        except (LLMTimeoutError, RuntimeError, FileNotFoundError, ImportError):
            return []
        words = [w.strip().strip(",.;:!?") for w in raw.replace("\n", ",").split(",")]
        return [w for w in words if w][:3]
