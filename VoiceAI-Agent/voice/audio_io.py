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
        self.block_size = sample_rate * block_ms // 1000
        max_blocks = int(max_queue_seconds * 1000 / block_ms)
        self._blocks: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=max_blocks)
        self._stream: sd.InputStream | None = None
        self.device: AudioDevice | None = None
        self.dropped_blocks = 0
        self.native_rate = sample_rate
        self._resampler: StatefulResampler | None = None
        self.warning = ""  # set when the mic can't give good 16 kHz audio

    def start(self) -> None:
        """Open the mic at 16 kHz; if the driver refuses (e.g. WDM-KS), open it at its own rate
        and resample every block to 16 kHz here."""
        self.device = resolve_device("input", self.requested_device)
        native = int(self.device.default_samplerate)
        if native < self.sample_rate:
            self.warning = (f"'{self.device.name.splitlines()[0]}' only records at {native} Hz "
                            "(Bluetooth hands-free mode). Speech recognition will make more mistakes; "
                            "a wired headset or USB mic is much better.")
        try:
            try:
                self._stream = self._open(self.sample_rate)
            except sd.PortAudioError:
                self.native_rate = native
                self._resampler = StatefulResampler(native, self.sample_rate)
                self._stream = self._open(native)
            self._stream.start()
        except sd.PortAudioError as error:
            self._stream = None
            name = self.device.name.splitlines()[0]
            raise AudioDeviceError(
                f"Could not open the microphone '{name}' ({self.device.host_api}): {error}. "
                "Is it connected and switched on? Run 'voiceai --list-devices' and set "
                "VOICEAI_MIC_DEVICE in .env to pick another one."
            ) from None

    def _open(self, rate: int) -> sd.InputStream:
        return sd.InputStream(
            samplerate=rate,
            blocksize=rate * self.block_size // self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device.index,
            latency="low",
            extra_settings=stream_settings(self.device),
            callback=self._callback,
        )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float | None = None) -> tuple[float, np.ndarray]:
        """Next block as (perf_counter time it arrived, samples). Raises queue.Empty on timeout."""
        return self._blocks.get(timeout=timeout)

    def _callback(self, indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        block = indata[:, 0].copy()
        if self._resampler is not None:
            block = self._resampler.process(block)
        try:
            self._blocks.put_nowait((time.perf_counter(), block))
        except queue.Full:
            self.dropped_blocks += 1  # consumer is too slow; never block the audio thread
