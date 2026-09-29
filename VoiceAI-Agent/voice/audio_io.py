"""Always-on microphone stream in small blocks (default 20 ms at 16 kHz).

Unlike `Microphone.record_until_silence`, this never stops between turns: the pipeline reads
blocks continuously, including while the agent is speaking, which is what makes barge-in
possible. The sound card callback only copies data into a queue, so it never blocks.

The stream can be (re)opened on another device at any time (`open`), checked with a short
`probe` (does audio really arrive?) and watched with `alive` (is the callback still running?).
The block queue survives a reopen, so readers don't notice the switch. voice/device_manager.py
uses this for automatic device selection and hot-plug.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np
import sounddevice as sd

from voice.audio_devices import AudioDevice, AudioDeviceError, resolve_device, stream_settings
from voice.resample import StatefulResampler


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
        self.block_ms = block_ms
        self.block_size = sample_rate * block_ms // 1000
        max_blocks = int(max_queue_seconds * 1000 / block_ms)
        self._blocks: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=max_blocks)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self.device: AudioDevice | None = None
        self.dropped_blocks = 0
        self.native_rate = sample_rate
        self._resampler: StatefulResampler | None = None
        self.warning = ""  # set when the mic can't give good 16 kHz audio
        self.last_callback = 0.0
        self.callbacks = 0
        self._peak = 0.0

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        """Open the best device (override, else the Windows default); see voice/audio_devices."""
        from voice.windows_audio import default_device_name

        self.open(resolve_device("input", self.requested_device, default_device_name("input")))

    def open(self, device: AudioDevice) -> None:
        """Open `device` at 16 kHz; if the driver refuses, open it at its own rate and resample
        every block to 16 kHz here. Raises AudioDeviceError if it can't be opened at all."""
        self.close()
        native = int(device.default_samplerate)
        warning = ""
        if native < self.sample_rate:
            warning = (f"'{device.name.splitlines()[0]}' only records at {native} Hz (Bluetooth "
                       "hands-free mode). Speech recognition will make more mistakes; a wired "
                       "headset or USB mic is much better.")
        resampler = None
        try:
            try:
                stream = self._open(device, self.sample_rate)
            except sd.PortAudioError:
                resampler = StatefulResampler(native, self.sample_rate)
                stream = self._open(device, native)
            with self._lock:
                self._resampler = resampler
                self.native_rate = native if resampler else self.sample_rate
            stream.start()
        except sd.PortAudioError as error:
            raise AudioDeviceError(
                f"Could not open the microphone {device.label}: {error}. Is it connected and "
                "switched on? Run 'voiceai --list-devices' to see devices."
            ) from None
        with self._lock:
            self._stream = stream
            self.device = device
            self.warning = warning
            self.last_callback = time.perf_counter()

    def _open(self, device: AudioDevice, rate: int) -> sd.InputStream:
        return sd.InputStream(
            samplerate=rate,
            blocksize=rate * self.block_ms // 1000,
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

    # -- health --------------------------------------------------------------------------------
    def probe(self, seconds: float = 0.2, startup_s: float = 3.0) -> tuple[bool, str]:
        """Does audio really arrive? Waits up to `startup_s` for the first block (a Bluetooth
        headset needs ~0.7-0.8 s to switch to hands-free mode, measured), then checks `seconds`
        of audio: enough blocks, and not pure digital silence."""
        deadline = time.perf_counter() + startup_s
        with self._lock:
            self.callbacks, self._peak = 0, 0.0
        while self.callbacks == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        if self.callbacks == 0:
            return False, f"no audio arrived within {startup_s:.0f} s"
        with self._lock:
            self.callbacks = 0
        time.sleep(seconds)
        expected = seconds * 1000 / self.block_ms
        if self.callbacks < expected * 0.4:
            return False, f"audio stalls ({self.callbacks} of ~{expected:.0f} blocks)"
        if self._peak == 0.0:
            return False, "silent (only zeros: the device may be asleep or muted)"
        return True, "ok"

    def alive(self, max_gap_s: float = 1.5) -> bool:
        return self._stream is not None and time.perf_counter() - self.last_callback < max_gap_s

    def drain(self) -> None:
        while True:
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                return

    # -- reading ---------------------------------------------------------------------------------
    def read(self, timeout: float | None = None) -> tuple[float, np.ndarray]:
        """Next block as (perf_counter time it arrived, samples). Raises queue.Empty on timeout."""
        return self._blocks.get(timeout=timeout)

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        now = time.perf_counter()
        block = indata[:, 0].copy()
        self.last_callback = now
        self.callbacks += 1
        peak = float(np.abs(block).max()) if block.size else 0.0
        self._peak = max(self._peak, peak)
        if self._resampler is not None:
            block = self._resampler.process(block)
        try:
            self._blocks.put_nowait((now, block))
        except queue.Full:
            self.dropped_blocks += 1  # consumer is too slow; never block the audio thread
