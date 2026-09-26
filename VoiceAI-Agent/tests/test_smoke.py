import numpy as np

from agent.gemini_client import GeminiClient, GeminiError
from config.settings import Settings
from voice.microphone import Microphone


def test_default_settings() -> None:
    settings = Settings()

    assert settings.model == "gemini-3.8-flash"
    assert isinstance(settings.gemini_api_key, str)
    assert settings.project_root is None or settings.project_root.is_dir()


def test_gemini_client_generates_visible_text(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        text = "  Calculator ready.  "

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr("agent.gemini_client.genai.Client", lambda **kwargs: FakeClient())

    result = GeminiClient("test-key").generate("Create a calculator")

    assert result == "Calculator ready."
    assert captured["model"] == "gemini-3.8-flash"
    assert captured["contents"] == "Create a calculator"
    assert captured["config"].max_output_tokens == 128
    assert captured["config"].thinking_config.thinking_level.value == "LOW"


def test_gemini_client_requires_api_key() -> None:
    try:
        GeminiClient(" ")
    except GeminiError as error:
        assert "GEMINI_API_KEY" in str(error)
    else:
        raise AssertionError("Expected a missing API key error")


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
    monkeypatch.setattr(microphone, "find_device", lambda: 3)

    audio = microphone.record_until_silence(
        max_seconds=5,
        silence_seconds=0.2,
        calibration_seconds=0.1,
        block_duration=0.1,
        threshold=0.1,
    )

    assert audio.size == 20
    assert np.all(audio[:8] == 0.2)
