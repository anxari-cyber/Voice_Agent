"""Speaker output that can be silenced instantly (barge-in) and knows what was played.

Audio is queued as float32 chunks. The sound card pulls 20 ms blocks from the queue in a
callback thread. `stop()` empties the queue, so the very next block is silence.
`played_samples` counts only audio that actually reached the sound card, which later lets
the agent remember exactly how much of its answer the user heard.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from voice.audio_devices import AudioDevice, resolve_device, stream_settings


class Player:
    def __init__(
        self,
        sample_rate: int = 24_000,
        device: str | int | None = None,
        block_ms: int = 20,
        on_event: Callable[[str, float], None] | None = None,
    ) -> None:
        """`on_event(name, perf_counter_time)` receives "playback_start" / "playback_stopped"."""
        self.sample_rate = sample_rate
        self.requested_device = device
        self.block_size = sample_rate * block_ms // 1000
        self.on_event = on_event
        self._queue: deque[np.ndarray] = deque()
        self._offset = 0  # read position inside the first queued chunk
        self._lock = threading.Lock()
        self._was_playing = False
        self._stop_requested = False
        self.played_samples = 0
        self._stream: sd.OutputStream | None = None
        self.device: AudioDevice | None = None

    # -- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        self.device = resolve_device("output", self.requested_device)
        self._stream = sd.OutputStream(
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

    # -- control ---------------------------------------------------------------------------
    def play(self, audio: np.ndarray) -> None:
        """Queue audio (float32 mono at `sample_rate`). Returns immediately."""
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            with self._lock:
                self._queue.append(chunk)

    def stop(self) -> None:
        """Drop everything queued. The next audio block the sound card asks for is silence."""
        with self._lock:
            self._queue.clear()
            self._offset = 0
            self._stop_requested = self._was_playing

    def reset_counter(self) -> None:
        with self._lock:
            self.played_samples = 0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._queue)

    @property
    def played_seconds(self) -> float:
        return self.played_samples / self.sample_rate

    def wait_until_done(self, poll: float = 0.01) -> None:
        while self.is_playing:
            time.sleep(poll)

    # -- sound card thread -----------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, _time: object, _status: object) -> None:
        out = outdata[:, 0]
        written = 0
        with self._lock:
            while written < frames and self._queue:
                chunk = self._queue[0]
                take = min(frames - written, len(chunk) - self._offset)
                out[written : written + take] = chunk[self._offset : self._offset + take]
                written += take
                self._offset += take
                if self._offset >= len(chunk):
                    self._queue.popleft()
                    self._offset = 0
            self.played_samples += written
            started = written > 0 and not self._was_playing
            stopped = self._stop_requested
            self._stop_requested = False
            self._was_playing = bool(self._queue) or (written == frames)
        out[written:] = 0.0
        if self.on_event:
            now = time.perf_counter()
            if stopped:
                self.on_event("playback_stopped", now)
            if started:
                self.on_event("playback_start", now)
