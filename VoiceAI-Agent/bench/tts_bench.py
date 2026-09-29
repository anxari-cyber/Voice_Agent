"""Step 1.3 benchmark: Kokoro TTS alone, next to the LLM on the same GPU, and end to end.

Usage: python -m bench.tts_bench [--play] [--samples]
  --play      make the speaker audible (default: the real audio path runs at gain 0)
  --samples   also write WAV samples to logs/tts_samples/ for the 1-5 naturalness rating

Measures:
  1. cold first call vs after warm_up()
  2. text -> first audio chunk (G2P + model + trim), per clause length, and real-time factor
  3. the same while the LLM is generating on the GPU at the same time (and LLM tok/s meanwhile)
  4. end to end: LLM request -> chunker -> TTS -> first chunk queued / first block played
  5. gaps: player blocks where the queue ran dry mid-answer, and sound-card underflows
  6. cancel: chunks that still arrive after TTSWorker.cancel()
Targets: first audio for the first clause < 120 ms; no gaps; cancel within one chunk.
"""

from __future__ import annotations

import argparse
import threading
import time
import wave
from pathlib import Path

import numpy as np
import torch

from agent.llm import OllamaLLM
from agent.memory import Memory
from agent.prompt import PromptBuilder
from config.settings import load_settings
from metrics.latency import LatencyLog
from pipeline.chunker import ClauseChunker
from pipeline.tts_worker import TTSWorker
from voice.playback import Player
from voice.tts import SAMPLE_RATE, kokoro_from_settings

CLAUSES = ["Okay.", "Sure, I can do that.", "Let me check the login test for you,",
           "The login test fails because the token expired yesterday."]
ANSWER_PROMPTS = ["Explain what a REST API is.", "Give me three tips to focus better.",
                  "How do I undo my last git commit?", "Why is the sky blue?",
                  "What should I cook for dinner tonight?"]
LONG_PROMPT = "Count slowly from one to three hundred, writing every number as a word."


class GainSink:
    def __init__(self, player: Player, gain: float) -> None:
        self.player = player
        self.gain = gain

    def play(self, audio: np.ndarray) -> None:
        self.player.play(audio * self.gain)


def first_audio_ms(tts, text: str) -> tuple[float, float, float]:
    """(ms to first chunk, total ms, seconds of audio) for one clause."""
    started = time.perf_counter()
    first = None
    samples = 0
    for chunk in tts.synthesize_stream(text):
        if first is None:
            first = (time.perf_counter() - started) * 1000
        samples += len(chunk)
    return first or 0.0, (time.perf_counter() - started) * 1000, samples / SAMPLE_RATE


def clause_table(tts, label: str, repeats: int = 5) -> dict[str, float]:
    print(f"\n{label}")
    firsts = {}
    for text in CLAUSES:
        runs = [first_audio_ms(tts, text) for _ in range(repeats)]
        first = float(np.median([r[0] for r in runs]))
        total = float(np.median([r[1] for r in runs]))
        seconds = runs[0][2]
        firsts[text] = first
        print(f"  first audio {first:5.0f} ms | whole clause {total:5.0f} ms | {seconds:4.2f} s audio "
              f"(RTF {total / 1000 / seconds:.3f})  {text!r}")
    return firsts


def llm_in_background(llm: OllamaLLM, builder: PromptBuilder, stop: threading.Event, out: dict) -> None:
    tokens, started = 0, time.perf_counter()
    while not stop.is_set():
        stream = llm.stream(builder.build(Memory(), LONG_PROMPT))
        for delta in stream:
            tokens += bool(delta.content)
            if stop.is_set():
                stream.cancel()
                break
    out["tok_s"] = tokens / (time.perf_counter() - started)


def end_to_end(llm, builder, tts, player, gain: float, latency: LatencyLog) -> list[dict]:
    results = []
    for prompt in ANSWER_PROMPTS:
        events: dict[str, float] = {}

        def on_event(name: str, at: float, events=events) -> None:
            events.setdefault(name, at)
            if name in ("tts_first_audio", "playback_start"):
                latency.mark(name, at=at)

        player.on_event = on_event
        worker = TTSWorker(tts, GainSink(player, gain), on_event=on_event)
        chunker = ClauseChunker()
        player.reset_counter()
        starved_before, underflow_before = player.starved_blocks, player.underflows
        latency.next_turn()
        started = latency.mark("stt_final")  # "text ready" is where this bench starts the clock
        player.begin_utterance()
        first_token = None
        clauses = 0
        for delta in llm.stream(builder.build(Memory(), prompt)):
            if delta.content and first_token is None:
                first_token = latency.mark("llm_first_token")
            for clause in chunker.feed(delta.content):
                worker.submit(clause)
                clauses += 1
        for clause in chunker.flush():
            worker.submit(clause)
            clauses += 1
        latency.mark("llm_done")
        worker.wait_idle(30)
        player.wait_until_done()
        player.end_utterance()
        worker.close()
        results.append({
            "prompt": prompt,
            "first_token": (first_token - started) * 1000,
            "first_audio_queued": (events["tts_first_audio"] - started) * 1000,
            "first_audio_played": (events["playback_start"] - started) * 1000,
            "clauses": clauses,
            "first_clause": worker.clauses[0].text if worker.clauses else "",
            "gaps": player.starved_blocks - starved_before,
            "underflows": player.underflows - underflow_before,
            "seconds": worker.total_samples / SAMPLE_RATE,
        })
    return results


def cancel_test(tts, player: Player) -> tuple[int, float]:
    sink_chunks: list[float] = []

    class CountingSink:
        def play(self, audio: np.ndarray) -> None:
            sink_chunks.append(time.perf_counter())

    worker = TTSWorker(tts, CountingSink())
    for text in CLAUSES * 3:
        worker.submit(text)
    while len(sink_chunks) < 8:
        time.sleep(0.001)
    cancelled_at = time.perf_counter()
    worker.cancel()
    time.sleep(0.4)
    late = [t for t in sink_chunks if t > cancelled_at]
    worker.close()
    return len(late), (max(late) - cancelled_at) * 1000 if late else 0.0


def save_samples(tts, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    texts = ["Hi! I'm your local voice assistant. How can I help you today?",
             "The login test fails because the token expired, so I renewed it and the tests pass now.",
             "Sure, give me a second. I'll check the git status and tell you what changed."]
    paths = []
    for i, text in enumerate(texts, 1):
        audio = np.concatenate(list(tts.synthesize_stream(text)))
        path = folder / f"sample_{i}.wav"
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(SAMPLE_RATE)
            file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--samples", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    latency = LatencyLog(settings.latency_log)

    started = time.perf_counter()
    tts = kokoro_from_settings(settings)
    load_ms = (time.perf_counter() - started) * 1000
    cold_first, _, _ = first_audio_ms(tts, "The very first call.")
    started = time.perf_counter()
    tts.warm_up()
    warm_ms = (time.perf_counter() - started) * 1000
    warm_first, _, _ = first_audio_ms(tts, "The very first call.")
    print(f"Kokoro on {tts.device}: load {load_ms:.0f} ms | cold first call {cold_first:.0f} ms | "
          f"warm_up() {warm_ms:.0f} ms | after warm-up {warm_first:.0f} ms | "
          f"torch CPU threads {torch.get_num_threads()} (setting {settings.tts_torch_threads})")

    alone = clause_table(tts, "Kokoro alone (text -> first audio chunk):")

    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    llm.warm_up(builder.warm_up_messages())
    solo: dict = {}
    stop = threading.Event()
    thread = threading.Thread(target=llm_in_background, args=(llm, builder, stop, solo))
    thread.start()
    time.sleep(2.0)
    stop.set()
    thread.join()
    busy: dict = {}
    stop = threading.Event()
    thread = threading.Thread(target=llm_in_background, args=(llm, builder, stop, busy))
    thread.start()
    time.sleep(0.5)
    shared = clause_table(tts, "Kokoro while the LLM generates on the same GPU:")
    stop.set()
    thread.join()
    print(f"  LLM speed: {solo['tok_s']:.0f} tok/s alone -> {busy['tok_s']:.0f} tok/s while Kokoro runs")
    worst = max(shared[t] - alone[t] for t in CLAUSES)
    print(f"  Kokoro first-audio slowdown with the LLM busy: up to +{worst:.0f} ms")

    player = Player(sample_rate=SAMPLE_RATE, device=settings.speaker_device)
    player.start()
    print(f"\nSpeaker: {player.device.name} ({player.device.host_api}, "
          f"{player.device.default_samplerate:.0f} Hz) -> resampling: {player.resampling}; "
          f"gain {'1.0 (audible)' if args.play else '0 (silent)'}")
    results = end_to_end(llm, builder, tts, player, 1.0 if args.play else 0.0, latency)
    print("\nEnd to end: text ready -> LLM -> chunker -> Kokoro -> speaker")
    for r in results:
        print(f"  first token {r['first_token']:4.0f} ms | first audio queued {r['first_audio_queued']:4.0f} ms "
              f"| played {r['first_audio_played']:4.0f} ms | {r['clauses']:2d} clauses, {r['seconds']:4.1f} s "
              f"| gaps {r['gaps']} underflows {r['underflows']} | first clause {r['first_clause']!r}")
    played = np.array([r["first_audio_played"] for r in results])
    print(f"  text ready -> first audio played: p50 {np.percentile(played, 50):.0f} ms, max {played.max():.0f} ms")
    print(f"  gaps between clauses: {sum(r['gaps'] for r in results)} starved blocks, "
          f"{sum(r['underflows'] for r in results)} underflows over {sum(r['seconds'] for r in results):.0f} s of speech")

    late, late_ms = cancel_test(tts, player)
    print(f"\nCancel: {late} chunk(s) arrived after cancel() (last one {late_ms:.1f} ms later); "
          f"chunk size {tts.config.chunk_ms} ms")
    if args.samples:
        for path in save_samples(tts, Path("logs/tts_samples")):
            print(f"  sample written: {path}")
    player.close()
    llm.close()
    latency.close()


if __name__ == "__main__":
    main()
