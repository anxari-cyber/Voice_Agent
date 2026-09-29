"""Where do the milliseconds go between "text ready" and the first sound? (before Step 1.4)

Usage: python -m bench.first_audio_breakdown [--prompts 5] [--idle 8] [--play]

"text ready" (t0) = the moment the final user transcript exists (the `stt_final` event); the
LLM request is sent right after it. Everything to do with speech (VAD, STT) is before t0.

  A  TTFT               t0 -> first non-empty LLM token
  B  chunker wait       first token -> first clause complete (tokens still needed for clause 1)
  C  TTS clause 1       clause submitted -> first audio chunk handed to the player
                        (worker pickup + G2P + Kokoro model + trim; the LLM keeps running on the GPU)
  D  queue -> callback  chunk queued -> the sound-card callback plays it
  E  output latency     PortAudio stream.latency: callback -> speaker
  heard = A + B + C + D + E

Runs 4 variants: {WASAPI auto_convert, our StatefulResampler} x {no GPU nudge, nudge}, with
`--idle` seconds of GPU idle before each prompt (like the user talking between answers).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agent.llm import OllamaLLM
from agent.memory import Memory
from agent.prompt import PromptBuilder
from bench.gpu_idle_bench import make_nudge
from config.settings import load_settings
from pipeline.chunker import ClauseChunker
from pipeline.tts_worker import TTSWorker
from voice.playback import Player
from voice.tts import SAMPLE_RATE, kokoro_from_settings

PROMPTS = ["Explain what a REST API is.", "Give me three tips to focus better.",
           "How do I undo my last git commit?", "Why is the sky blue?",
           "What should I cook for dinner tonight?", "What does RAM do in a computer?"]
STAGES = ("A TTFT", "B chunker wait", "C TTS clause 1", "D queue->callback", "E output latency", "heard")


class GainSink:
    def __init__(self, player: Player, gain: float) -> None:
        self.player, self.gain = player, gain

    def play(self, audio: np.ndarray) -> None:
        self.player.play(audio * self.gain)


def one_turn(llm, builder, tts, player, gain, prompt, nudge, nudge_ms) -> dict:
    events: dict[str, float] = {}

    def on_event(name: str, at: float) -> None:
        events.setdefault(name, at)

    if nudge:
        nudge(nudge_ms)  # in the real pipeline this runs at speech_start, while the user talks
    player.on_event = on_event
    worker = TTSWorker(tts, GainSink(player, gain), on_event=on_event)
    chunker = ClauseChunker()
    player.reset_counter()
    player.begin_utterance()
    t0 = time.perf_counter()  # text ready
    first_token = submitted = None
    stream = llm.stream(builder.build(Memory(), prompt))
    for delta in stream:
        if delta.content and first_token is None:
            first_token = time.perf_counter()
        for clause in chunker.feed(delta.content):
            if submitted is None:
                submitted = time.perf_counter()
                first_clause = clause
            worker.submit(clause)
        if "playback_start" in events and time.perf_counter() - events["playback_start"] > 0.5:
            break  # enough: we only measure the start of the answer
    stream.cancel()
    deadline = time.perf_counter() + 5
    while "playback_start" not in events and time.perf_counter() < deadline:
        time.sleep(0.005)
    worker.cancel()
    player.stop()
    player.end_utterance()
    worker.close()
    latency = player.output_latency_ms
    ms = {
        "A TTFT": (first_token - t0) * 1000,
        "B chunker wait": (submitted - first_token) * 1000,
        "C TTS clause 1": (events["tts_first_audio"] - submitted) * 1000,
        "D queue->callback": (events["playback_start"] - events["tts_first_audio"]) * 1000,
        "E output latency": latency,
    }
    ms["heard"] = sum(ms.values())
    ms["clause"] = first_clause
    return ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=5)
    parser.add_argument("--idle", type=float, default=8.0)
    parser.add_argument("--nudge-ms", type=float, default=150)
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    llm.warm_up(builder.warm_up_messages())
    tts = kokoro_from_settings(settings)
    tts.warm_up()
    nudge = make_nudge()
    gain = 1.0 if args.play else 0.0

    summary = {}
    for resample in ("auto", "stateful"):
        player = Player(sample_rate=SAMPLE_RATE, device=settings.speaker_device, resample=resample)
        player.start()
        for use_nudge in (False, True):
            label = f"{player.resampling:<20} | {'nudge ' if use_nudge else 'no nudge'}"
            rows = []
            for prompt in PROMPTS[: args.prompts]:
                time.sleep(args.idle)
                rows.append(one_turn(llm, builder, tts, player, gain, prompt,
                                     nudge if use_nudge else None, args.nudge_ms))
            summary[label] = rows
            print(f"\n{label}  (speaker {player.device.name}, {player.device.default_samplerate:.0f} Hz, "
                  f"underflows so far {player.underflows})")
            print("  " + "  ".join(f"{s:>17}" for s in STAGES) + "   first clause")
            for r in rows:
                print("  " + "  ".join(f"{r[s]:>14.0f} ms" for s in STAGES) + f"   {r['clause']!r}")
            print("  " + "  ".join(f"{np.median([r[s] for r in rows]):>11.0f} ms p50" for s in STAGES))
        player.close()

    print("\np50 heard (t0 -> sound at the speaker):")
    for label, rows in summary.items():
        print(f"  {label}: {np.median([r['heard'] for r in rows]):.0f} ms")
    llm.close()


if __name__ == "__main__":
    main()
