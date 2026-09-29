import numpy as np

from config.settings import Settings
from voice.audio_devices import AudioDevice
from voice.microphone import Microphone


def test_default_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_model == "qwen3:4b-instruct-2507-q4_K_M"
    assert settings.stt_engine == "parakeet"
    assert settings.tts_engine == "kokoro"
    assert settings.project_root is None
    assert settings.mic_device is None


def test_microphone_records_from_speech_until_silence(monkeypatch) -> None:
    class FakeStream:
        def __init__(self):
            self.chunks = [
                np.zeros((4, 1), dtype=np.float32),
                np.full((4, 1), 0.2, dtype=np.float32),
                np.full((4, 1), 0.2, dtype=np.float32),
                np.zeros((4, 1), dtype=np.float32),
                np.zeros((4, 1), dtype=np.float32),
                np.zeros((4, 1), dtype=np.float32),
            ]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, frames):
            return self.chunks.pop(0), None

    clock = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    monkeypatch.setattr("voice.microphone.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("voice.microphone.sd.InputStream", lambda **kwargs: FakeStream())

    microphone = Microphone(sample_rate=40)
    microphone._device = AudioDevice(3, "Test Mic", "MME", 1, 0, 16_000.0)

    audio = microphone.record_until_silence(
        max_seconds=5,
        silence_seconds=0.2,
        calibration_seconds=0.1,
        block_duration=0.1,
        threshold=0.1,
    )

    assert audio.size == 20
    assert np.all(audio[:8] == 0.2)
