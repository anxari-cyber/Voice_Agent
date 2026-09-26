"""Speech-to-text engines behind one small interface, plus live (streaming) transcription.

Engines:
    ParakeetSTT  NVIDIA Parakeet TDT 0.6B v2 via onnx-asr (DirectML GPU, CPU fallback). Default.
    WhisperSTT   faster-whisper (the original engine), kept as a backup.

StreamingSTT turns any engine into a live transcriber: while the user speaks it re-decodes the
growing audio every `partial_every_ms` in a background thread (partial text), and when speech
ends it decodes once more (final text). Parakeet is fast enough (~75 ms for 3-4 s of audio)
that re-decoding the whole utterance is simpler and more accurate than chunked decoding.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000


class STTEngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...


class ParakeetSTT:
    name = "parakeet"

    def __init__(
        self,
        model_dir: Path | str = "models/parakeet-tdt-0.6b-v2",
        quantization: str | None = "int8",
        providers: list[str] | None = None,
    ) -> None:
        import onnx_asr
        import onnxruntime as ort

        # CPU by default: measured on the RTX 5050, DirectML took 300-390 ms per utterance
        # (every utterance has a new length) while the CPU took 65-130 ms. It also keeps
        # the GPU free for the LLM and TTS.
        providers = providers or ["CPUExecutionProvider"]
        providers = [p for p in providers if p in ort.get_available_providers()]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False  # required by DirectML
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            model_dir,
            quantization=quantization,
            providers=providers,
            sess_options=options,
        )
        self.providers = providers
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up: first run is slow

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < SAMPLE_RATE // 10:  # < 100 ms: nothing to recognise
            return ""
        return self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()


class WhisperSTT:
    name = "whisper"

    def __init__(self, model_name: str = "small.en", device: str = "auto", compute_type: str = "auto"):
        from voice.transcriber import Transcriber

        self.transcriber = Transcriber(model_name, device=device, compute_type=compute_type)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcriber.transcribe(np.asarray(audio, dtype=np.float32))


def create_engine(name: str, **kwargs: object) -> STTEngine:
    engines = {"parakeet": ParakeetSTT, "whisper": WhisperSTT}
    if name not in engines:
        raise ValueError(f"Unknown STT engine '{name}'. Choose one of: {', '.join(engines)}")
    return engines[name](**kwargs)


class StreamingSTT:
    """Collects one utterance and produces partial and final transcripts."""

    def __init__(self, engine: STTEngine, partial_every_ms: int = 300) -> None:
        self.engine = engine
        self.partial_every = partial_every_ms / 1000
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._partial_job: Future[str] | None = None
        self._last_partial_at = 0.0
        self.partial_text = ""
        self.active = False

    def start(self, pre_roll: np.ndarray | None = None) -> None:
        """Begin a new utterance. `pre_roll` is audio from just before VAD fired."""
        with self._lock:
            self._chunks = [] if pre_roll is None or not len(pre_roll) else [pre_roll]
            self.partial_text = ""
            self._last_partial_at = time.perf_counter()
            self.active = True

    def resume(self, gap: np.ndarray | None = None) -> None:
        """Continue the same utterance after a short pause instead of starting a new one.

        `gap` is the audio recorded during the pause, so the words stay in order.
        """
        with self._lock:
            if gap is not None and len(gap):
                self._chunks.append(np.asarray(gap, dtype=np.float32).reshape(-1))
            self._last_partial_at = time.perf_counter()
            self.active = True

    def feed(self, block: np.ndarray) -> None:
        """Add audio; schedules a partial decode in the background when one is due."""
        if not self.active:
            return
        with self._lock:
            self._chunks.append(np.asarray(block, dtype=np.float32).reshape(-1))
            now = time.perf_counter()
            due = now - self._last_partial_at >= self.partial_every
            busy = self._partial_job is not None and not self._partial_job.done()
            if due and not busy:
                self._last_partial_at = now
                self._partial_job = self._executor.submit(self._decode_partial, self._audio())

    def finish(self) -> str:
        """End the utterance and return the final transcript (blocks for one decode)."""
        with self._lock:
            audio = self._audio()
            self.active = False
            job = self._partial_job
        if job is not None:
            job.result()  # let a running partial finish so the GPU is free
        text = self.engine.transcribe(audio)
        self.partial_text = text
        return text

    def cancel(self) -> None:
        with self._lock:
            self.active = False
            self._chunks = []

    @property
    def duration(self) -> float:
        with self._lock:
            return sum(len(c) for c in self._chunks) / SAMPLE_RATE

    def audio(self) -> np.ndarray:
        """Copy of the utterance audio collected so far."""
        with self._lock:
            return self._audio()

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _audio(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)

    def _decode_partial(self, audio: np.ndarray) -> str:
        text = self.engine.transcribe(audio)
        if self.active:
            self.partial_text = text
        return text
