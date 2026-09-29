"""Stateful streaming resampler (fallback only).

The player first asks WASAPI to convert 24 kHz itself (`auto_convert=True`), which is the
preferred path. This resampler is only used when a device refuses the TTS sample rate. It keeps
its position and the last input sample between chunks, so chunk boundaries don't click, which
is what a naive per-chunk resample would do.

It uses linear interpolation: fine for speech as a fallback, not a hi-fi resampler.
"""

from __future__ import annotations

import numpy as np


class StatefulResampler:
    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.step = source_rate / target_rate  # input samples per output sample
        self.reset()

    def reset(self) -> None:
        self._position = 0.0  # next output position, in input samples, relative to _previous
        self._previous = np.zeros(1, dtype=np.float32)  # last input sample of the last chunk

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.source_rate == self.target_rate or chunk.size == 0:
            return chunk
        # Index 0 is the last sample of the previous chunk, so interpolation spans the seam.
        signal = np.concatenate([self._previous, chunk])
        last = signal.size - 1
        count = int(np.floor((last - self._position) / self.step)) + 1
        if count <= 0:
            self._position -= chunk.size
            self._previous = signal[-1:]
            return np.zeros(0, dtype=np.float32)
        positions = self._position + self.step * np.arange(count)
        out = np.interp(positions, np.arange(signal.size), signal).astype(np.float32)
        self._position = positions[-1] + self.step - last  # carry into the next chunk
        self._previous = signal[-1:]
        return out
