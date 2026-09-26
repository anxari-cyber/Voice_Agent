"""Always-on microphone stream in small blocks (default 20 ms at 16 kHz).

Unlike `Microphone.record_until_silence`, this never stops between turns: the pipeline reads
blocks continuously, including while the agent is speaking, which is what makes barge-in
possible. The sound card callback only copies data into a queue, so it never blocks.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd

from voice.audio_devices import AudioDevice, resolve_device, stream_settings


class MicStream:
    def __init__(
        self,
        device: str | int | None = None,
        sample_rate: int = 16_000,
        block_ms: int = 20,
        max_queue_seconds: float = 5.0,
    ) -> None:
        self.requested_device = device
        self.sample_rate = sample_rate
        self.block_size = sample_rate * block_ms // 1000
        max_blocks = int(max_queue_seconds * 1000 / block_ms)
        self._blocks: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=max_blocks)
        self._stream: sd.InputStream | None = None
        self.device: AudioDevice | None = None
        self.dropped_blocks = 0

    def start(self) -> None:
        self.device = resolve_device("input", self.requested_device)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device.index,
            latency="low",
            extra_settings=stream_settings(self.device),
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float | None = None) -> tuple[float, np.ndarray]:
        """Next block as (perf_counter time it arrived, samples). Raises queue.Empty on timeout."""
        return self._blocks.get(timeout=timeout)

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        try:
            self._blocks.put_nowait((time.perf_counter(), indata[:, 0].copy()))
        except queue.Full:
            self.dropped_blocks += 1  # consumer is too slow; never block the audio thread
