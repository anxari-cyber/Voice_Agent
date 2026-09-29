from __future__ import annotations

import argparse
import asyncio
import threading
import time

from voice.audio_devices import AudioDeviceError, format_devices, list_devices


def run(args: argparse.Namespace) -> None:
    """Load everything once, warm it up, then talk until Ctrl+C."""
    import warnings

    # torch / Kokoro print harmless deprecation notes while loading; keep the console readable.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    from agent.llm import LLMError, OllamaLLM
    from agent.memory import Memory
    from agent.prompt import PromptBuilder
    from config.settings import load_settings
    from metrics.latency import LatencyLog
    from pipeline.chunker import ClauseChunker
    from pipeline.orchestrator import AudioFrontEnd, Orchestrator
    from voice.audio_io import MicStream
    from voice.gpu_warm import GpuWarmer
    from voice.playback import Player
    from voice.stt import streaming_from_settings
    from voice.tts import kokoro_from_settings
    from voice.vad import SileroVAD

    settings = load_settings()
    started = time.perf_counter()
    print("Starting the voice agent (everything runs locally)...")

    mic = MicStream(device=settings.mic_device)
    mic.start()  # fail fast, before loading models, if there is no usable mic
    player = Player(sample_rate=24_000, device=settings.speaker_device)
    player.start()
    print(f"  Mic:     {mic.device.name.splitlines()[0]} ({mic.device.host_api})")
    print(f"  Speaker: {player.device.name.splitlines()[0]} ({player.device.host_api})")
    if mic.warning:
        print(f"  Warning: {mic.warning}")

    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    llm_error: list[str] = []

    def warm_llm() -> None:  # may take ~30 s on a cold start from the HDD: do it in parallel
        try:
            llm.warm_up(builder.warm_up_messages())
        except LLMError as error:
            llm_error.append(str(error))

    llm_thread = threading.Thread(target=warm_llm, daemon=True)
    llm_thread.start()
    print(f"  Loading speech recognition ({settings.stt_engine})...")
    stt = streaming_from_settings(settings)
    print("  Loading the voice (Kokoro)...")
    tts = kokoro_from_settings(settings)
    tts.warm_up()
    warmer = GpuWarmer()
    print(f"  Loading the language model ({settings.llm_model})...")
    llm_thread.join()
    if llm_error:
        mic.close()
        player.close()
        raise SystemExit(f"The language model is not available: {llm_error[0]}\n"
                         "Is Ollama running? Start it from the Start menu, then try again.")

    latency = LatencyLog(settings.latency_log)
    vad = SileroVAD(threshold=settings.vad_threshold, start_ms=settings.vad_start_ms,
                    end_ms=settings.vad_end_ms)
    frontend = AudioFrontEnd(mic, vad, stt, pre_roll_ms=settings.pre_roll_ms,
                             join_window_ms=settings.join_window_ms, latency=latency)
    orchestrator = Orchestrator(frontend, llm, builder, Memory(), tts, player, latency=latency,
                                warmer=warmer, chunker_factory=ClauseChunker)
    print(f"\nReady in {time.perf_counter() - started:.0f} s. Speak whenever you like.")
    print("Press Ctrl+C to quit.")
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.close()
        mic.close()
        player.close()
        latency.close()
        print(f"\nBye. {len(orchestrator.turns)} turns. Audio underflows: {player.underflows}, "
              f"gaps: {player.starved_blocks}. Latency log: {settings.latency_log}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceai", description="Local voice AI agent")
    parser.add_argument(
        "--list-devices", action="store_true", help="Show available microphones and speakers"
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="Start the voice agent and talk to it")
    args = parser.parse_args()

    if args.list_devices:
        print(format_devices(list_devices()))
    elif args.command == "run":
        try:
            run(args)
        except AudioDeviceError as error:
            raise SystemExit(f"\n{error}") from None
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
