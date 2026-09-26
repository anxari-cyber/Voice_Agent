from __future__ import annotations

import time
from collections import deque

import numpy as np
import sounddevice as sd


class Microphone:
    def __init__(self, device_name: str = "AirPods Pro", sample_rate: int = 16_000) -> None:
        self.device_name = device_name
        self.sample_rate = sample_rate

    def find_device(self) -> int:
        for index, device in enumerate(sd.query_devices()):
            if self.device_name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
                return index
        raise RuntimeError(f"Microphone input not found: {self.device_name}")

    def _device_index(self) -> int:
        device_index = self.find_device()
        return device_index

    def record(self, seconds: float) -> np.ndarray:
        if seconds <= 0:
            raise ValueError("Recording duration must be greater than zero.")

        device_index = self._device_index()
        audio = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
        return audio[:, 0]

    def record_until_silence(
        self,
        max_seconds: float = 15.0,
        silence_seconds: float = 0.7,
        speech_timeout: float = 10.0,
        calibration_seconds: float = 0.4,
        block_duration: float = 0.1,
        threshold: float | None = None,
        speech_start_blocks: int = 2,
        pre_roll_seconds: float = 1.0,
    ) -> np.ndarray:
        """Record after speech starts and stop once the speaker is quiet."""
        if max_seconds <= 0 or silence_seconds <= 0 or speech_timeout <= 0:
            raise ValueError("Recording limits must be greater than zero.")
        if calibration_seconds < 0 or block_duration <= 0:
            raise ValueError("Calibration and block duration must be non-negative and positive.")
        if speech_start_blocks <= 0 or pre_roll_seconds < 0:
            raise ValueError("Speech start blocks must be positive and pre-roll cannot be negative.")

        device_index = self._device_index()
        block_size = max(1, int(self.sample_rate * block_duration))
        calibration_size = max(1, int(self.sample_rate * calibration_seconds))

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            device=device_index,
        ) as stream:
            if calibration_seconds:
                ambient, _ = stream.read(calibration_size)
                ambient_rms = self._rms(ambient)
            else:
                ambient_rms = 0.0

            if threshold is None:
                speech_threshold = max(0.003, ambient_rms + 0.003)
            else:
                speech_threshold = max(0.005, threshold)
            started_at = time.monotonic()
            speech_started_at: float | None = None
            silence_started_at: float | None = None
            chunks: list[np.ndarray] = []
            speech_blocks = 0
            pre_roll = deque(maxlen=max(1, int(pre_roll_seconds / block_duration)))

            while True:
                chunk, _ = stream.read(block_size)
                samples = np.asarray(chunk[:, 0] if chunk.ndim > 1 else chunk, dtype=np.float32)
                now = time.monotonic()
                is_speech = self._rms(samples) >= speech_threshold

                if speech_started_at is None:
                    pre_roll.append(samples)
                    if is_speech:
                        speech_blocks += 1
                        if speech_blocks >= speech_start_blocks:
                            speech_started_at = now
                            chunks.extend(pre_roll)
                    else:
                        speech_blocks = 0
                    if speech_started_at is None and now - started_at >= speech_timeout:
                        return np.empty(0, dtype=np.float32)
                    continue

                chunks.append(samples)
                if is_speech:
                    silence_started_at = None
                elif silence_started_at is None:
                    silence_started_at = now
                elif now - silence_started_at >= silence_seconds:
                    return np.concatenate(chunks)

                if now - speech_started_at >= max_seconds:
                    return np.concatenate(chunks)

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
