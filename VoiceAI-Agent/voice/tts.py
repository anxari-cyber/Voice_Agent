"""Text-to-speech behind a small interface, with Kokoro-82M (PyTorch, GPU) as the engine.

Why Kokoro is called at a low level instead of through `KPipeline(text)`:
- KPipeline splits text on newlines and at 510 phonemes. Our clause chunker already decides
  where to cut, so we call Kokoro's G2P and model directly: one clause = one model call.
- Kokoro pads every clip with ~300-400 ms of silence at the start and ~400-500 ms at the end
  (measured). The start silence alone would add ~300 ms to every reply, so it is trimmed, and
  the end is replaced by a controlled pause that depends on the clause's last punctuation.

Each clause is synthesised in one call (Kokoro is not autoregressive) and then yielded in
small chunks, so a cancel takes effect within one chunk.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 24_000

# Pause after a clause, by its final punctuation (milliseconds of silence).
PAUSES_MS = {".": 280, "!": 280, "?": 300, ";": 200, ":": 200, ",": 120}
DEFAULT_PAUSE_MS = 60  # no punctuation (clause cut for length): keep it flowing
LEAD_KEEP_MS = 15  # keep a few ms before the first sound so the onset isn't clipped
TAIL_KEEP_MS = 40  # keep the natural decay of the last sound
TRIM_THRESHOLD = 0.02  # fraction of the clip's peak that counts as sound


class TTS(Protocol):
    sample_rate: int

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]: ...

    def cancel(self) -> None: ...


def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cut leading/trailing silence, keeping LEAD_KEEP_MS / TAIL_KEEP_MS around the sound."""
    if audio.size == 0:
        return audio
    envelope = np.abs(audio)
    peak = float(envelope.max())
    if peak <= 1e-6:
        return audio[:0]
    loud = np.flatnonzero(envelope > TRIM_THRESHOLD * peak)
    start = max(0, loud[0] - sample_rate * LEAD_KEEP_MS // 1000)
    end = min(audio.size, loud[-1] + 1 + sample_rate * TAIL_KEEP_MS // 1000)
    return audio[start:end]


def pause_after(text: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    stripped = text.rstrip().rstrip("\"')]}»”’")
    ms = PAUSES_MS.get(stripped[-1:], DEFAULT_PAUSE_MS) if stripped else 0
    return np.zeros(sample_rate * ms // 1000, dtype=np.float32)


@dataclass
class KokoroConfig:
    weights_dir: Path = Path("models/kokoro-torch")
    voice: str = "af_heart"
    device: str = "cuda"
    speed: float = 1.0
    torch_threads: int = 2
    chunk_ms: int = 40


class KokoroTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, config: KokoroConfig | None = None) -> None:
        import torch
        from kokoro import KModel, KPipeline

        self.config = config or KokoroConfig()
        # Kokoro's G2P and any CPU ops must not grab every core: Parakeet, the VAD and the
        # audio threads need the CPU too. This limit is process-wide.
        torch.set_num_threads(self.config.torch_threads)
        weights = Path(self.config.weights_dir)
        device = self.config.device if (self.config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = device
        self._torch = torch
        self.model = KModel(
            repo_id="hexgrad/Kokoro-82M",
            config=str(weights / "config.json"),
            model=str(weights / "kokoro-v1_0.pth"),
        ).to(device).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)
        self._tokens_to_ps = KPipeline.tokens_to_ps
        self.voice_pack = self._pipeline.load_voice(str(weights / "voices" / f"{self.config.voice}.pt")).to(device)
        self._cancel = threading.Event()
        self._model_lock = threading.Lock()  # one synthesis at a time on the GPU

    def warm_up(self) -> None:
        """First call takes ~1.5 s (CUDA kernels, cuDNN). Run a few lengths at startup."""
        for text in ("Okay.", "Sure, I can do that.", "Here is a slightly longer sentence to warm up."):
            self.synthesize(text)

    def phonemes(self, text: str) -> str:
        _, tokens = self._pipeline.g2p(text)
        return self._tokens_to_ps(tokens).strip()

    def synthesize(self, text: str) -> np.ndarray:
        """Whole clause -> trimmed audio (float32, 24 kHz), without the trailing pause."""
        phonemes = self.phonemes(text)
        if not phonemes:
            return np.zeros(0, dtype=np.float32)
        pieces = [phonemes[i : i + 510] for i in range(0, len(phonemes), 510)]  # rare: > 510
        audio = []
        with self._model_lock, self._torch.inference_mode():
            for piece in pieces:
                output = self.model(piece, self.voice_pack[len(piece) - 1], self.config.speed, return_output=True)
                audio.append(output.audio.float().cpu().numpy())
        return trim_silence(np.concatenate(audio))

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield the clause in `chunk_ms` chunks, then its pause. Stops early on cancel()."""
        self._cancel.clear()
        audio = self.synthesize(text)
        if self._cancel.is_set():
            return
        audio = np.concatenate([audio, pause_after(text)]) if audio.size else audio
        step = self.sample_rate * self.config.chunk_ms // 1000
        for start in range(0, audio.size, step):
            if self._cancel.is_set():
                return
            yield audio[start : start + step]

    def cancel(self) -> None:
        self._cancel.set()


def kokoro_from_settings(settings) -> KokoroTTS:
    """Build KokoroTTS from config.settings.Settings (models_dir, voice, device, speed, threads)."""
    return KokoroTTS(KokoroConfig(
        weights_dir=Path(settings.models_dir) / "kokoro-torch",
        voice=settings.tts_voice,
        device=settings.tts_device,
        speed=settings.tts_speed,
        torch_threads=settings.tts_torch_threads,
    ))
