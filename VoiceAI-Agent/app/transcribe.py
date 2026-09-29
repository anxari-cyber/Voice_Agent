from __future__ import annotations

import argparse
import time

from agent.ollama_client import OllamaClient, OllamaError
from config.settings import load_settings
from metrics.latency import LatencyLog
from voice.microphone import Microphone
from voice.stt import STTEngine, create_engine


def transcribe_command(
    microphone: Microphone,
    transcriber: STTEngine,
    max_seconds: float = 15.0,
    silence_seconds: float = 0.7,
    speech_timeout: float = 10.0,
    threshold: float | None = None,
    latency: LatencyLog | None = None,
) -> str:
    started_at = time.perf_counter()
    audio = microphone.record_until_silence(
        max_seconds=max_seconds,
        silence_seconds=silence_seconds,
        speech_timeout=speech_timeout,
        threshold=threshold,
    )
    recorded_at = time.perf_counter()
    print(f"Recording finished in {recorded_at - started_at:.1f}s")
    if audio.size == 0:
        return ""
    if latency:
        # Recording only ends after `silence_seconds` of quiet, so the user actually
        # stopped talking that long ago. Counting from there keeps the wait visible.
        latency.mark("speech_end", at=recorded_at - silence_seconds)
    transcription_started_at = time.perf_counter()
    result = transcriber.transcribe(audio)
    if latency:
        latency.mark("stt_final")
    print(f"{transcriber.name} finished in {time.perf_counter() - transcription_started_at:.2f}s")
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--silence-seconds", type=float, default=0.7)
    parser.add_argument("--speech-timeout", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=None)


def run(args: argparse.Namespace) -> None:
    settings = load_settings()
    latency = LatencyLog(settings.latency_log)
    microphone = Microphone(settings.mic_device)
    print(f"Microphone: {microphone.device.name} ({microphone.device.host_api})")
    transcriber = create_engine(settings.stt_engine)
    print(f"Speak now... (STT: {transcriber.name})")
    print("Listening for speech...")
    latency.next_turn()
    transcription = transcribe_command(
        microphone,
        transcriber,
        max_seconds=args.max_seconds,
        silence_seconds=args.silence_seconds,
        speech_timeout=args.speech_timeout,
        threshold=args.threshold,
        latency=latency,
    )
    print(f"Transcription: {transcription}")
    if not transcription:
        print("No speech detected.")
        return

    # Temporary bridge until the streaming pipeline (roadmap Step 1.5) replaces this file.
    llm = OllamaClient(settings.llm_url, settings.llm_model, num_predict=settings.llm_max_tokens)
    try:
        response = llm.generate(transcription)
        latency.mark("llm_done")
        print(f"{settings.llm_model}: {response}")
    except OllamaError as error:
        print(f"Ollama is unavailable: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and transcribe an English voice command.")
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
