from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""
    gemini_timeout_ms: int = 30_000
    project_root: Path | None = None
    stt_engine: str = "parakeet"  # parakeet | whisper
    stt_model: str = "small.en"  # used by the whisper engine
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    # Name substring or index; empty means the system default device.
    mic_device: str | None = None
    speaker_device: str | None = None
    latency_log: Path = Path("logs/latency.jsonl")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VOICEAI_", extra="ignore")


def load_settings() -> Settings:
    return Settings()
