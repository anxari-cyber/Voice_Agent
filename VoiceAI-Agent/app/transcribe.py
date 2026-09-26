from __future__ import annotations

import argparse
import time

from agent.gemini_client import GeminiClient, GeminiError
from config.settings import load_settings
from voice.microphone import Microphone
from voice.transcriber import Transcriber


def transcribe_command(
    microphone: Microphone,
    transcriber: Transcriber,
    max_seconds: float = 15.0,
    silence_seconds: float = 0.7,
    speech_timeout: float = 10.0,
    threshold: float | None = None,
) -> str:
    started_at = time.perf_counter()
    audio = microphone.record_until_silence(
        max_seconds=max_seconds,
        silence_seconds=silence_seconds,
        speech_timeout=speech_timeout,
        threshold=threshold,
    )
    print(f"Recording finished in {time.perf_counter() - started_at:.1f}s")
    if audio.size == 0:
        return ""
    transcription_started_at = time.perf_counter()
    result = transcriber.transcribe(audio)
    print(f"Whisper finished in {time.perf_counter() - transcription_started_at:.1f}s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and transcribe an English voice command.")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--silence-seconds", type=float, default=0.7)
    parser.add_argument("--speech-timeout", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    settings = load_settings()
    microphone = Microphone()
    transcriber = Transcriber(
        settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
    )
    print(f"Speak now... (Whisper device: {transcriber.device})")
    print("Listening for speech...")
    transcription = transcribe_command(
        microphone,
        transcriber,
        max_seconds=args.max_seconds,
        silence_seconds=args.silence_seconds,
        speech_timeout=args.speech_timeout,
        threshold=args.threshold,
    )
    print(f"Transcription: {transcription}")
    if not transcription:
        print("No speech detected.")
        return

    try:
        gemini = GeminiClient(
            settings.gemini_api_key,
            settings.model,
            timeout_ms=settings.gemini_timeout_ms,
        )
        response_started_at = time.perf_counter()
        response = gemini.generate(transcription)
        print(f"Gemini finished in {time.perf_counter() - response_started_at:.1f}s")
        print(f"{settings.model}: {response}")
    except GeminiError as error:
        print(f"Gemini is unavailable: {error}")


if __name__ == "__main__":
    main()
