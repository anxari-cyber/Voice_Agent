from config.settings import load_settings


def main() -> None:
    settings = load_settings()
    print("VoiceAI-Agent MVP")
    print(f"Ollama: {settings.ollama_url}")
    print(f"Model: {settings.model}")
    print(f"Project root: {settings.project_root or 'not selected'}")
    print("Status: scaffold ready")


if __name__ == "__main__":
    main()
