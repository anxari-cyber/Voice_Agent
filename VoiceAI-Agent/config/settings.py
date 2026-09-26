from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""
    gemini_timeout_ms: int = 30_000
    project_root: Path | None = None
    stt_model: str = "small.en"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VOICEAI_", extra="ignore")


def load_settings() -> Settings:
    return Settings()
