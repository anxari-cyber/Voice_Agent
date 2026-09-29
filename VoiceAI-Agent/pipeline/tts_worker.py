"""TTS worker: clauses in, audio out to the speaker, with a ledger of what was queued.

A background thread takes clauses from a queue, synthesises each one, and writes the chunks
to the sink (the Player). It records how many samples every clause produced and where in the
turn's audio it starts, so context repair (roadmap Step 2.2) can map `played_samples` back to
the words the user actually heard.

`cancel()` drops queued clauses and stops the clause being synthesised within one chunk.
It does NOT silence the speaker: barge-in calls `player.stop()` and `worker.cancel()` together.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from voice.tts import TTS


class Sink(Protocol):
    def play(self, audio: np.ndarray) -> None: ...


@dataclass
class ClauseAudio:
    index: int
    text: str
    start_sample: int  # position in this turn's audio (TTS sample rate)
    samples: int = 0  # samples queued so far, including the trailing pause
    done: bool = False  # all of its audio was queued (not cut by a cancel)


class TTSWorker:
    def __init__(self, tts: TTS, sink: Sink,
                 on_event: Callable[[str, float], None] | None = None) -> None:
        self.tts = tts
        self.sink = sink
        self.on_event = on_event or (lambda name, at: None)
        self._queue: queue.Queue[tuple[int, int, str] | None] = queue.Queue()
        self._generation = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self.clauses: list[ClauseAudio] = []
        self.total_samples = 0
        self._first_audio_sent = False
        self._thread = threading.Thread(target=self._run, name="tts-worker", daemon=True)
        self._thread.start()

    # -- control (any thread) ----------------------------------------------------------------
    def new_turn(self) -> None:
        """Start a fresh ledger for the next answer (cancels anything still pending)."""
        self.cancel()
        with self._lock:
            self.clauses = []
            self.total_samples = 0
            self._first_audio_sent = False

    def submit(self, text: str) -> int:
        """Queue one clause for speech; returns its index in this turn."""
        with self._lock:
            index = len(self.clauses)
            self.clauses.append(ClauseAudio(index, text, start_sample=-1))
            self._idle.clear()
            self._queue.put((self._generation, index, text))  # under the lock: see _run()
        return index

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.tts.cancel()
        self.on_event("tts_cancel", time.perf_counter())
        self._idle.set()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """True once every submitted clause has been synthesised and queued to the sink."""
        return self._idle.wait(timeout)

    def close(self) -> None:
        self.cancel()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def heard_text(self, played_samples: int) -> str:
        """Clauses the user has heard (fully or partly) after `played_samples` of this turn."""
        with self._lock:
            return " ".join(c.text for c in self.clauses
                            if 0 <= c.start_sample < played_samples)

    # -- worker thread -----------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, index, text = item
            if generation != self._generation:
                continue
            with self._lock:
                clause = self.clauses[index]
                clause.start_sample = self.total_samples
            for chunk in self.tts.synthesize_stream(text):
                if generation != self._generation:
                    break
                self.sink.play(chunk)
                with self._lock:
                    clause.samples += len(chunk)
                    self.total_samples += len(chunk)
                    first = not self._first_audio_sent
                    self._first_audio_sent = True
                if first:
                    self.on_event("tts_first_audio", time.perf_counter())
            else:
                clause.done = generation == self._generation
            with self._lock:  # same lock as submit(), so "empty" and "idle" can't disagree
                if self._queue.empty():
                    self._idle.set()
