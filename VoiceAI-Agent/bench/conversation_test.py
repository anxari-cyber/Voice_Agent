"""Step 1.5 scripted conversation: the real pipeline, fed by recorded speech instead of a mic.

Usage: python -m bench.conversation_test [--turns 10] [--minutes 0] [--first-words 6] [--no-nudge]
                                         [--folder synthetic] [--play]

Everything is real (Silero VAD, Parakeet, Qwen3-4B via Ollama, Kokoro, the speaker) except the
microphone: a FileMic plays the recorded commands (bench/data/commands/<folder>) in real time,
waits until the agent has finished answering, then says the next one after a short pause.
The speaker runs at gain 0 unless --play.

Reports per turn and p50/p95: speech end -> first audio heard, plus STT / join wait / LLM first
token / TTS first audio, and audio underflows + gaps (the GIL check, roadmap L8).
--minutes N keeps going (cycling through the commands) for N minutes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from agent.llm import OllamaLLM
from agent.memory import Memory
from agent.prompt import PromptBuilder
from config.settings import load_settings
from metrics.latency import LatencyLog
from pipeline.chunker import ClauseChunker
from pipeline.orchestrator import AudioFrontEnd, Orchestrator
from voice.gpu_warm import GpuWarmer
from voice.playback import Player
from voice.stt import streaming_from_settings
from voice.tts import kokoro_from_settings
from voice.vad import SileroVAD

BLOCK = 320  # 20 ms at 16 kHz


class FileMic:
    """Plays queued audio in real time (20 ms blocks), silence otherwise, like a real mic."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._audio = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, audio: np.ndarray) -> None:
        with self._lock:
            self._audio = np.concatenate([self._audio, audio])

    def read(self, timeout: float | None = None):
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        start = time.perf_counter()
        count = 0
        while not self._stop.is_set():
            count += 1
            target = start + count * BLOCK / 16_000
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            with self._lock:
                block, self._audio = self._audio[:BLOCK], self._audio[BLOCK:]
            if block.size < BLOCK:
                block = np.concatenate([block, np.zeros(BLOCK - block.size, dtype=np.float32)])
            try:
                self._queue.put_nowait((time.perf_counter(), block))
            except queue.Full:
                pass


class GainSink:
    def __init__(self, player: Player, gain: float) -> None:
        self.player, self.gain = player, gain

    def play(self, audio: np.ndarray) -> None:
        self.player.play(audio * self.gain)


class QuietUI:
    def state(self, name): pass
    def user_partial(self, text): pass
    def user_final(self, text): pass
    def agent_start(self): pass
    def agent_delta(self, text): pass
    def turn_done(self, stats): pass


def load_commands(folder: Path) -> list[tuple[str, np.ndarray]]:
    records = {}
    for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["file"]] = record
    commands = []
    for record in sorted(records.values(), key=lambda r: r["file"]):
        with wave.open(str(folder / record["file"]), "rb") as file:
            audio = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        commands.append((record["text"], audio))
    return commands


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--first-words", type=int, default=6, help="max words in the first clause")
    parser.add_argument("--no-nudge", action="store_true")
    parser.add_argument("--folder", default="synthetic")
    parser.add_argument("--gap", type=float, default=1.5, help="seconds of silence before each command")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    settings = load_settings()
    commands = load_commands(Path("bench/data/commands") / args.folder)

    stt = streaming_from_settings(settings)
    tts = kokoro_from_settings(settings)
    tts.warm_up()
    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    llm.warm_up(builder.warm_up_messages())
    player = Player(sample_rate=24_000, device=settings.speaker_device)
    player.start()
    mic = FileMic()
    latency = LatencyLog(settings.latency_log, run_id=f"conv-{args.label or 'default'}-{int(time.time())}")
    frontend = AudioFrontEnd(mic, SileroVAD(threshold=settings.vad_threshold, start_ms=settings.vad_start_ms,
                                            end_ms=settings.vad_end_ms),
                             stt, pre_roll_ms=settings.pre_roll_ms, join_window_ms=settings.join_window_ms,
                             latency=latency)
    deadline = time.perf_counter() + args.minutes * 60 if args.minutes else None
    total = 10**6 if deadline else args.turns
    said = {"count": 0}

    def next_command() -> None:
        if said["count"] >= total or (deadline and time.perf_counter() > deadline):
            return
        _, audio = commands[said["count"] % len(commands)]
        said["count"] += 1
        mic.say(np.concatenate([np.zeros(int(args.gap * 16_000), dtype=np.float32), audio]))

    def on_state(name: str) -> None:
        if name == "listening":
            next_command()

    orchestrator = Orchestrator(
        frontend, llm, builder, Memory(), tts, player, latency=latency,
        warmer=None if args.no_nudge else GpuWarmer(),
        sink=GainSink(player, 1.0 if args.play else 0.0), ui=QuietUI(),
        chunker_factory=lambda: ClauseChunker(first_max_words=args.first_words), on_state=on_state)

    async def run() -> list:
        task = asyncio.create_task(orchestrator.run(max_turns=None if deadline else args.turns))
        while not task.done():
            await asyncio.sleep(0.2)
            if deadline and time.perf_counter() > deadline + 30:
                orchestrator._stop.set()
        return await task

    started = time.perf_counter()
    turns = asyncio.run(run())
    elapsed = time.perf_counter() - started
    orchestrator.close()
    mic.close()
    player.close()
    latency.close()

    stages = [("heard", "speech_end", "playback_start"), ("STT final", "speech_end", "stt_final"),
              ("join wait", "stt_final", "turn_commit"), ("LLM 1st token", "turn_commit", "llm_first_token"),
              ("TTS 1st audio", "llm_first_token", "tts_first_audio"),
              ("queue->play", "tts_first_audio", "playback_start")]
    label = args.label or f"first_words={args.first_words} nudge={'off' if args.no_nudge else 'on'}"
    print(f"\n== {label}: {len(turns)} turns in {elapsed / 60:.1f} min")
    for i, turn in enumerate(turns, 1):
        heard = turn.ms("speech_end", "playback_start")
        print(f"  {i:2d}. heard {heard:5.0f} ms | You: {turn.user_text[:48]!r} -> {turn.answer[:50]!r}"
              + (f" | ERROR {turn.error}" if turn.error else ""))
    print("  stage                p50      p95")
    for name, a, b in stages:
        values = [v for t in turns if (v := t.ms(a, b)) is not None]
        print(f"  {name:<16} {pct(values, 50):6.0f} ms {pct(values, 95):6.0f} ms")
    print(f"  audio: {player.underflows} underflows, {player.starved_blocks} gaps; "
          f"mic blocks dropped: 0 (file mic); GPU nudges: "
          f"{orchestrator.warmer.kicks if orchestrator.warmer else 0}")


if __name__ == "__main__":
    main()
