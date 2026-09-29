from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or .env, prefixed with VOICEAI_."""

    project_root: Path | None = None
    # Where downloaded models live (Parakeet, Kokoro, Smart Turn). Can point to another drive.
    models_dir: Path = Path("models")

    # Audio devices: name substring or index; empty means the system default device.
    mic_device: str | None = None
    speaker_device: str | None = None

    # LLM (Voice Brain), served locally by Ollama.
    llm_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_num_ctx: int = 4096
    llm_max_tokens: int = 200
    system_prompt_file: Path = Path("config/system_prompt.md")

    # Speech-to-text.
    stt_engine: str = "parakeet"  # parakeet | whisper
    stt_model: str = "small.en"  # used by the whisper engine
    stt_device: str = "auto"
    stt_compute_type: str = "auto"

    # Text-to-speech.
    tts_engine: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_device: str = "cuda"

    # Voice activity detection and turn taking.
    vad_threshold: float = 0.5
    vad_start_ms: int = 64
    vad_end_ms: int = 200
    pre_roll_ms: int = 500
    join_window_ms: int = 700
    bargein_min_ms: int = 160
    smart_turn_threshold: float = 0.5

    latency_log: Path = Path("logs/latency.jsonl")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VOICEAI_", extra="ignore")


def load_settings() -> Settings:
    return Settings()
