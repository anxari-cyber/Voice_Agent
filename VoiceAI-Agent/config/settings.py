from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The VoiceAI-Agent folder. Relative paths in settings are resolved against it, so the agent
# works no matter which folder it is started from.
PROJECT_DIR = Path(__file__).resolve().parents[1]


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
    stt_partial_threads: int = 3  # CPU threads for partial decodes; the final uses all cores
    stt_trim_after_s: float = 2.5  # decode window length before confirmed words move out

    # Text-to-speech.
    tts_engine: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_device: str = "cuda"
    tts_speed: float = 1.0
    tts_torch_threads: int = 2  # keep CPU cores free for Parakeet, the VAD and audio threads

    # Voice activity detection and turn taking.
    vad_threshold: float = 0.5
    vad_start_ms: int = 64
    vad_end_ms: int = 200
    pre_roll_ms: int = 500
    join_window_ms: int = 700
    bargein_min_ms: int = 160
    smart_turn_threshold: float = 0.5

    latency_log: Path = Path("logs/latency.jsonl")

    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", env_prefix="VOICEAI_", extra="ignore")

    @field_validator("models_dir", "system_prompt_file", "latency_log")
    @classmethod
    def _relative_to_project(cls, value: Path) -> Path:
        value = Path(value)
        return value if value.is_absolute() else PROJECT_DIR / value


def load_settings() -> Settings:
    return Settings()
