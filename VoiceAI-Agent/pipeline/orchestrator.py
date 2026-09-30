"""The end-to-end conversation loop (roadmap Step 1.5).

    source (mic) -> AudioFrontEnd thread: VAD -> streaming STT -> join window -> events
                                                                                  |
    Orchestrator (asyncio): LISTENING -> THINKING -> SPEAKING -> LISTENING   <----+
        on TurnCommit: LLM stream -> clause chunker -> TTS worker -> player   (executor thread)

- The audio front-end is the only real-time part. It runs in its own thread, never waits on the
  models (partials run in the STT's own worker), and hands events to asyncio with
  call_soon_threadsafe. Audio callbacks only touch queues.
- A turn commits once the join window (700 ms) passes in silence, as in Step 1.4. Speculative
  starts (answering during the join window) are Step 1.6.
- The GPU is woken at speech_start and speech_end (amendment #11), while the user talks.
- No barge-in yet (Step 2.2): while the agent thinks and speaks the mic is not listened to, so
  its own voice can't start a turn. Listening resumes when the answer has finished playing.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from agent.llm import LLMError
from agent.memory import Memory
from agent.prompt import PromptBuilder
from metrics.latency import LatencyLog
from pipeline.chunker import ClauseChunker
from pipeline.events import (
    FinalText,
    PartialText,
    SpeechEnd,
    SpeechStart,
    TurnCommit,
    TurnReopen,
)
from pipeline.tts_worker import TTSWorker
from pipeline.turn_filter import ignore_reason

SAMPLE_RATE = 16_000


class AudioSource(Protocol):
    def read(self, timeout: float | None = None) -> tuple[float, np.ndarray]: ...


# ---------------------------------------------------------------------------------------------
# Audio front-end
# ---------------------------------------------------------------------------------------------

@dataclass
class _Pending:
    text: str
    speech_end: float
    final_ready: float
    speech_s: float = 0.0


class AudioFrontEnd:
    """Mic blocks in, turn events out. Runs in its own thread."""

    def __init__(self, source: AudioSource, vad, stt, emit: Callable[[object], None] | None = None,
                 pre_roll_ms: int = 500, join_window_ms: int = 700, latency: LatencyLog | None = None,
                 min_speech_ms: int = 350, on_ignored: Callable[[str, str], None] | None = None):
        self.source = source
        self.vad = vad
        self.stt = stt
        self.emit = emit or (lambda event: None)
        self.pre_roll_blocks = max(1, pre_roll_ms // 20)
        self.join_window = join_window_ms / 1000
        self.latency = latency
        self.min_speech_s = min_speech_ms / 1000
        self.on_ignored = on_ignored or (lambda text, reason: None)
        self.ignored = 0
        self.listening = threading.Event()
        self.listening.set()
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        drain = getattr(self.source, "drain", None)
        if drain:  # audio from before we started listening is stale (and has old timestamps)
            drain()
        self._thread = threading.Thread(target=self._run, name="audio-frontend", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def pause_listening(self) -> None:
        self.listening.clear()

    def reset_audio(self) -> None:
        """The mic changed (hot-plug): drop any half-heard utterance, keep the listening state."""
        self._reset.set()

    def resume_listening(self) -> None:
        """Listen again from now on; anything heard while paused (the agent's voice) is dropped."""
        self._reset.set()
        self.listening.set()

    def _mark(self, event: str, at: float | None = None, **extra) -> None:
        if self.latency:
            self.latency.mark(event, at=at, **extra)

    def _run(self) -> None:
        pre_roll: deque[np.ndarray] = deque(maxlen=self.pre_roll_blocks)
        gap: list[np.ndarray] = []
        pending: _Pending | None = None
        # Positions are counted in the VAD's own coordinates (samples it has processed since its
        # last reset), because VAD events report positions that way. Blocks skipped while the
        # agent speaks are not counted, and the counter restarts with the VAD.
        vad_samples = origin = 0
        shown = ""
        speech_s = 0.0  # detected speech in this turn, summed over its pauses
        segment_start = 0.0
        while not self._stop.is_set():
            try:
                arrived, block = self.source.read(timeout=0.1)
            except queue.Empty:
                arrived, block = time.perf_counter(), None
            if self._reset.is_set():  # (re)start listening with a clean state
                self._reset.clear()
                self.vad.reset()
                vad_samples = origin = 0
                self.stt.cancel()
                pre_roll.clear()
                gap, pending, shown = [], None, ""
                speech_s = 0.0
            if not self.listening.is_set() or block is None:
                if block is not None:
                    pre_roll.append(block)
                continue

            vad_samples += len(block)
            events = self.vad.process(block)
            if self.stt.active:
                self.stt.feed(block)
            elif pending:
                gap.append(block)
            else:
                pre_roll.append(block)

            for event in events:
                happened = arrived - (vad_samples - event.sample) / SAMPLE_RATE
                if event.kind == "speech_start":
                    segment_start = happened
                    if pending:  # a pause inside the join window: the same turn goes on
                        self.stt.resume(np.concatenate(gap) if gap else None)
                        pending, gap = None, []
                        self._mark("turn_reopen", at=happened)
                        self.emit(TurnReopen(happened))
                        continue
                    pre = np.concatenate(pre_roll) if pre_roll else None
                    origin = vad_samples - (len(pre) if pre is not None else 0)
                    self.stt.start(pre)
                    pre_roll.clear()
                    shown = ""
                    speech_s = 0.0
                    if self.latency:
                        self.latency.next_turn()
                    self._mark("speech_start", at=happened)
                    self.emit(SpeechStart(happened))
                elif event.kind == "speech_end" and self.stt.active:
                    text = self.stt.finish(speech_end_sample=event.sample - origin)
                    ready = time.perf_counter()
                    info = self.stt.last_final
                    speech_s += max(0.0, happened - segment_start)
                    pending = _Pending(text, happened, ready, speech_s)
                    self._mark("speech_end", at=happened)
                    self._mark("stt_final", at=ready, mode=info.get("mode"))
                    self.emit(SpeechEnd(happened))
                    self.emit(FinalText(text, ready, info.get("ms", 0.0), info.get("mode", "")))

            if self.stt.active and self.stt.partial_text != shown:
                shown = self.stt.partial_text
                self.emit(PartialText(shown))

            if pending and arrived - pending.speech_end >= self.join_window:
                reason = ignore_reason(pending.text, pending.speech_s, self.min_speech_s)
                if reason:  # noise / a filler sound: don't answer, keep listening
                    self.ignored += 1
                    self.on_ignored(pending.text, reason)
                    self.stt.cancel()
                    pending, gap, speech_s = None, [], 0.0
                    continue
                now = time.perf_counter()
                self.listening.clear()  # the orchestrator resumes listening after the answer
                self._mark("turn_commit", at=now)
                self.emit(TurnCommit(pending.text, pending.speech_end, pending.final_ready, now))
                pending, gap = None, []


# ---------------------------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------------------------

@dataclass
class TurnStats:
    user_text: str
    answer: str = ""
    speech_end: float = 0.0
    final_ready: float = 0.0
    commit: float = 0.0
    events: dict[str, float] = field(default_factory=dict)
    error: str = ""
    spoken_s: float = 0.0  # seconds of speech that reached the sound card
    speaker: str = ""  # which speaker it went to

    def ms(self, start: str, end: str) -> float | None:
        points = {"speech_end": self.speech_end, "stt_final": self.final_ready,
                  "turn_commit": self.commit, **self.events}
        if start in points and end in points and points[start] and points[end]:
            return (points[end] - points[start]) * 1000
        return None


class ConsoleUI:
    """What the user sees in the terminal. Replace with another object for tests/servers."""

    def state(self, name: str) -> None:
        if name == "listening":
            print("\n[listening]", flush=True)
        elif name == "speaking":
            print("[speaking - not listening until the answer ends]", flush=True)

    def user_partial(self, text: str) -> None:
        print(f"\r  You: {text} ...", end="", flush=True)

    def user_final(self, text: str) -> None:
        print(f"\r  You: {text}      ", flush=True)

    def agent_start(self) -> None:
        print("Agent: ", end="", flush=True)

    def agent_delta(self, text: str) -> None:
        print(text, end="", flush=True)

    def turn_done(self, stats: TurnStats) -> None:
        print()
        parts = [("heard after", "speech_end", "playback_start"), ("STT", "speech_end", "stt_final"),
                 ("join wait", "stt_final", "turn_commit"), ("LLM 1st token", "turn_commit", "llm_first_token"),
                 ("1st audio", "llm_first_token", "tts_first_audio")]
        shown = [f"{label} {value:.0f} ms" for label, a, b in parts if (value := stats.ms(a, b)) is not None]
        if shown:
            print("   [" + " | ".join(shown) + "]", flush=True)
        if stats.error:
            print(f"   [error: {stats.error}]", flush=True)
        print(f"   [spoke {stats.spoken_s:.1f} s on {stats.speaker or 'no speaker'}]", flush=True)


class Orchestrator:
    def __init__(self, frontend: AudioFrontEnd, llm, builder: PromptBuilder, memory: Memory,
                 tts, player, latency: LatencyLog | None = None, warmer=None, sink=None,
                 ui: ConsoleUI | None = None, chunker_factory: Callable[[], ClauseChunker] = ClauseChunker,
                 on_state: Callable[[str], None] | None = None) -> None:
        self.frontend = frontend
        self.llm = llm
        self.builder = builder
        self.memory = memory
        self.tts = tts
        self.player = player
        self.latency = latency
        self.warmer = warmer
        self.ui = ui or ConsoleUI()
        self.chunker_factory = chunker_factory
        self.on_state = on_state or (lambda name: None)
        self.state = "listening"
        self.turns: list[TurnStats] = []
        self._events: dict[str, float] = {}
        self._stop = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="respond")
        self.player.on_event = self._record
        self.worker = TTSWorker(tts, sink or player, on_event=self._record)

    # -- events from the player / TTS worker (their own threads) ---------------------------------
    def _record(self, name: str, at: float) -> None:
        if name not in self._events:
            self._events[name] = at
            if self.latency and name in ("tts_first_audio", "playback_start"):
                self.latency.mark(name, at=at)

    def _set_state(self, name: str) -> None:
        self.state = name
        self.ui.state(name)
        self.on_state(name)

    # -- main loop -----------------------------------------------------------------------------
    async def run(self, max_turns: int | None = None) -> list[TurnStats]:
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue = asyncio.Queue()
        self.frontend.emit = lambda event: loop.call_soon_threadsafe(inbox.put_nowait, event)
        self.frontend.start()
        self._set_state("listening")
        try:
            while not self._stop.is_set():
                try:
                    event = await asyncio.wait_for(inbox.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(event, (SpeechStart, SpeechEnd)):
                    # Wake the GPU while the user is still talking, and again when they stop:
                    # measured, a single kick at speech start cools down again before a long
                    # command's LLM request (> ~3 s later: TTFT 51 -> ~220 ms).
                    if self.warmer:
                        self.warmer.kick()
                elif isinstance(event, PartialText):
                    self.ui.user_partial(event.text)
                elif isinstance(event, TurnCommit):
                    if not event.text.strip():  # noise, a cough: nothing to answer
                        self.frontend.resume_listening()
                        continue
                    self.ui.user_final(event.text)
                    self._set_state("thinking")
                    stats = await loop.run_in_executor(self._executor, self._respond, event)
                    self.turns.append(stats)
                    self.ui.turn_done(stats)
                    self.frontend.resume_listening()
                    self._set_state("listening")
                    if max_turns and len(self.turns) >= max_turns:
                        break
        finally:
            self.frontend.stop()
        return self.turns

    def stop(self) -> None:
        self._stop.set()
        self.llm.cancel()
        self.worker.cancel()
        self.player.stop()

    def close(self) -> None:
        self.stop()
        self.worker.close()
        self._executor.shutdown(wait=False)

    # -- one answer (executor thread) --------------------------------------------------------------
    def _respond(self, commit: TurnCommit) -> TurnStats:
        stats = TurnStats(commit.text, speech_end=commit.speech_end,
                          final_ready=commit.final_ready, commit=commit.at)
        self._events = {}
        messages = self.builder.build(self.memory, commit.text)
        self.worker.new_turn()
        self.player.reset_counter()
        self.player.begin_utterance()
        chunker = self.chunker_factory()
        answer: list[str] = []
        stream = self.llm.stream(messages)
        self.ui.agent_start()
        try:
            for delta in stream:
                if delta.content:
                    if "llm_first_token" not in self._events:
                        self._events["llm_first_token"] = time.perf_counter()
                        if self.latency:
                            self.latency.mark("llm_first_token", at=self._events["llm_first_token"])
                        self._set_state("speaking")
                    answer.append(delta.content)
                    self.ui.agent_delta(delta.content)
                    for clause in chunker.feed(delta.content):
                        self.worker.submit(clause)
            for clause in chunker.flush():
                self.worker.submit(clause)
            self._events["llm_done"] = time.perf_counter()
            if self.latency:
                self.latency.mark("llm_done", at=self._events["llm_done"])
            self.builder.calibrate(messages, stream.stats)
        except LLMError as error:
            stats.error = str(error)
        self.worker.wait_idle(60)
        self.player.wait_until_done()
        self.player.end_utterance()
        stats.answer = "".join(answer).strip()
        stats.events = dict(self._events)
        stats.spoken_s = self.player.played_samples / getattr(self.player, "sample_rate", 24_000)
        device = getattr(self.player, "device", None)
        if device is not None and getattr(self.player, "is_open", True):
            stats.speaker = device.label
        heard = self.worker.heard_text(self.player.played_samples) or stats.answer
        self.memory.add_user(commit.text)
        if heard:
            self.memory.add_assistant(heard)
        return stats
