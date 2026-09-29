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

from voice.audio_devices import AudioDevice, AudioDeviceError, resolve_device, stream_settings
from voice.resample import StatefulResampler


class Player:
    def __init__(
        self,
        sample_rate: int = 24_000,
        device: str | int | None = None,
        block_ms: int = 20,
        on_event: Callable[[str, float], None] | None = None,
        resample: str = "auto",
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
        self._played = 0.0  # in `sample_rate` units, even when the stream runs at another rate
        self._stream: sd.OutputStream | None = None
        self.device: AudioDevice | None = None
        self.stream_rate = sample_rate
        self._resampler: StatefulResampler | None = None
        # "auto": let WASAPI convert (fallback to ours if refused); "stateful": always ours
        self.resample_mode = resample
        self.resampling = "none"  # "none" | "wasapi-auto-convert" | "stateful-fallback"
        self._streaming = False
        self.underflows = 0  # sound card reported an output underflow
        self.starved_blocks = 0  # queue ran dry in the middle of an utterance (audible gap)
        self._maybe_gap = 0
        self.available = True
        self.last_callback = 0.0
        self.callbacks = 0

    # -- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        """Open the best speaker (override, else the Windows default); see voice/audio_devices."""
        from voice.windows_audio import default_device_name

        self.open(resolve_device("output", self.requested_device, default_device_name("output")))

    def open(self, device: AudioDevice) -> None:
        """Open `device` at the TTS rate (WASAPI converts it); if the device refuses, open it at
        its own rate and resample here. Queued audio and counters survive a reopen.
        Raises AudioDeviceError if it can't be opened at all."""
        self.close()
        resampler, rate, mode = None, self.sample_rate, "none"
        try:
            try:
                if self.resample_mode == "stateful" and int(device.default_samplerate) != self.sample_rate:
                    raise sd.PortAudioError("stateful resampling requested")
                stream = self._open(device, self.sample_rate)
                if device.host_api == "Windows WASAPI" and int(device.default_samplerate) != self.sample_rate:
                    mode = "wasapi-auto-convert"
            except sd.PortAudioError:
                rate = int(device.default_samplerate)
                resampler = StatefulResampler(self.sample_rate, rate)
                stream = self._open(device, rate)
                mode = "stateful-fallback" if self.resample_mode == "auto" else "stateful-forced"
            with self._lock:
                self._resampler, self.stream_rate, self.resampling = resampler, rate, mode
            stream.start()
        except sd.PortAudioError as error:
            raise AudioDeviceError(
                f"Could not open the speaker {device.label}: {error}. Is it connected and switched on?"
            ) from None
        with self._lock:
            self._stream = stream
            self.device = device
            self.available = True
            self.last_callback = time.perf_counter()

    @property
    def output_latency_ms(self) -> float:
        """Sound card output latency reported by PortAudio (queue -> speaker after the callback)."""
        return float(self._stream.latency) * 1000 if self._stream is not None else 0.0

    def _open(self, device: AudioDevice, rate: int) -> sd.OutputStream:
        return sd.OutputStream(
            samplerate=rate,
            blocksize=rate * self.block_size // self.sample_rate,
            channels=1,
            dtype="float32",
            device=device.index,
            latency="low",
            extra_settings=stream_settings(device),
            callback=self._callback,
        )

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except sd.PortAudioError:
                pass  # the device is already gone

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def set_available(self, available: bool) -> None:
        """While no speaker works, audio is dropped and is_playing is False, so nothing waits
        for playback that can't happen."""
        with self._lock:
            self.available = available
            if not available:
                self._queue.clear()
                self._offset = 0

    # -- health --------------------------------------------------------------------------------
    def probe(self, seconds: float = 0.2, startup_s: float = 3.0) -> tuple[bool, str]:
        """Is the sound card really asking for audio? Waits up to `startup_s` for the first
        callback (Bluetooth profile switches take a moment), then checks `seconds` of callbacks."""
        deadline = time.perf_counter() + startup_s
        self.callbacks = 0
        while self.callbacks == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        if self.callbacks == 0:
            return False, f"the sound card isn't taking audio (nothing within {startup_s:.0f} s)"
        self.callbacks = 0
        time.sleep(seconds)
        expected = seconds * self.sample_rate / self.block_size
        if self.callbacks < expected * 0.4:
            return False, f"the sound card stalls ({self.callbacks} of ~{expected:.0f} blocks)"
        return True, "ok"

    def alive(self, max_gap_s: float = 1.5) -> bool:
        return self._stream is not None and time.perf_counter() - self.last_callback < max_gap_s

    # -- control ---------------------------------------------------------------------------
    def play(self, audio: np.ndarray) -> None:
        """Queue audio (float32 mono at `sample_rate`). Returns immediately."""
        if not self.available:
            return  # no working speaker right now (unplugged / reconnecting)
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self._resampler is not None:
            chunk = self._resampler.process(chunk)
        if chunk.size:
            with self._lock:
                self._queue.append(chunk)
                self.starved_blocks += self._maybe_gap  # the queue ran dry, then audio came
                self._maybe_gap = 0

    def stop(self) -> None:
        """Drop everything queued. The next audio block the sound card asks for is silence."""
        with self._lock:
            self._queue.clear()
            self._offset = 0
            self._stop_requested = self._was_playing
            self._streaming = False
            self._maybe_gap = 0
        if self._resampler is not None:
            self._resampler.reset()

    def reset_counter(self) -> None:
        with self._lock:
            self._played = 0.0

    @property
    def played_samples(self) -> int:
        """Samples that reached the sound card, in `sample_rate` (TTS) units."""
        return round(self._played)

    def begin_utterance(self) -> None:
        """Audio is on its way: from the first played block on, an empty queue is a gap."""
        with self._lock:
            self._streaming = True

    def end_utterance(self) -> None:
        with self._lock:
            self._streaming = False
            self._maybe_gap = 0  # running dry at the very end is not a gap

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self.available and bool(self._queue)

    @property
    def played_seconds(self) -> float:
        return self.played_samples / self.sample_rate

    def wait_until_done(self, poll: float = 0.01) -> None:
        while self.is_playing:
            time.sleep(poll)

    # -- sound card thread -----------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, _time: object, status: object) -> None:
        self.last_callback = time.perf_counter()
        self.callbacks += 1
        if status and getattr(status, "output_underflow", False):
            self.underflows += 1
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
            self._played += written * self.sample_rate / self.stream_rate
            if self._streaming and self._was_playing and written < frames and not self._queue:
                self._maybe_gap += 1  # only a gap if more audio follows (see play())
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
