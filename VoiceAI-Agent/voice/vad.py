"""Streaming voice activity detection with the Silero VAD v6 model.

The model file (voice/assets/silero_vad_v6.onnx) is the export shipped with faster-whisper
(MIT), stored here so faster-whisper stays optional. See THIRD_PARTY_NOTICES.md.

Feed it audio of any length (16 kHz mono float32). It splits the audio into the 512-sample
windows Silero expects, keeps the model state between calls, and reports when speech starts
and ends. Runs on CPU: one window takes well under 1 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16_000
WINDOW = 512  # 32 ms, fixed by the model
CONTEXT = 64  # samples of the previous window the model also sees


def bundled_model_path() -> Path:
    return Path(__file__).parent / "assets" / "silero_vad_v6.onnx"


@dataclass(frozen=True)
class VADEvent:
    kind: Literal["speech_start", "speech_end"]
    sample: int  # position in the stream where the change happened

    @property
    def seconds(self) -> float:
        return self.sample / SAMPLE_RATE


class SileroVAD:
    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float = 0.5,
        start_ms: int = 64,
        end_ms: int = 200,
    ) -> None:
        """`start_ms` of speech opens a segment; `end_ms` of silence closes it."""
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path or bundled_model_path()),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.start_windows = max(1, round(start_ms / 32))
        self.end_windows = max(1, round(end_ms / 32))
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._position = 0  # samples consumed so far
        self._speech_run = 0
        self._silence_run = 0
        self.speaking = False
        self.last_probability = 0.0

    def probability(self, window: np.ndarray) -> float:
        """Speech probability for exactly one 512-sample window (updates model state)."""
        model_input = np.concatenate([self._context, window])[np.newaxis, :]
        probs, self._h, self._c = self.session.run(
            None, {"input": model_input, "h": self._h, "c": self._c}
        )
        self._context = window[-CONTEXT:]
        self.last_probability = float(probs.reshape(-1)[0])
        return self.last_probability

    def process(self, audio: np.ndarray) -> list[VADEvent]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._pending = np.concatenate([self._pending, audio])
        events: list[VADEvent] = []
        while len(self._pending) >= WINDOW:
            window, self._pending = self._pending[:WINDOW], self._pending[WINDOW:]
            probability = self.probability(window)
            self._position += WINDOW
            event = self._update(probability)
            if event:
                events.append(event)
        return events

    def _update(self, probability: float) -> VADEvent | None:
        if not self.speaking:
            self._speech_run = self._speech_run + 1 if probability >= self.threshold else 0
            if self._speech_run >= self.start_windows:
                self.speaking = True
                self._silence_run = 0
                start = self._position - self._speech_run * WINDOW
                return VADEvent("speech_start", start)
            return None
        self._silence_run = self._silence_run + 1 if probability < self.neg_threshold else 0
        if self._silence_run >= self.end_windows:
            self.speaking = False
            self._speech_run = 0
            end = self._position - self._silence_run * WINDOW
            return VADEvent("speech_end", end)
        return None
