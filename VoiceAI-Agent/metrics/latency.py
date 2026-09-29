"""Timestamp every pipeline stage so latency can be measured instead of guessed.

Each call to `mark` produces one JSON line:
    {"run": "...", "turn": 3, "event": "stt_final", "t": 12.3456, "wall": "2026-09-26T14:02:11.123+05:00"}
`t` comes from time.perf_counter(), so only differences within one run are meaningful.

`mark` is called on the hot path (including audio callbacks), so it never touches the disk:
it puts a small tuple on a queue, and a background thread formats and writes the lines.
Call `flush()` before reading the file, and `close()` when done. `close()` also runs at exit.
"""

from __future__ import annotations

import atexit
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

# Stage events in pipeline order. Reports compute the gaps between these.
EVENTS = (
    "speech_start",
    "stt_partial",
    "speech_end",
    "turn_soft_end",
    "stt_final",
    "speculative_start",
    "turn_reopen",
    "turn_commit",
    "llm_first_token",
    "llm_done",
    "llm_cancel",
    "tts_first_audio",
    "tts_cancel",
    "playback_start",
    "bargein_detected",
    "playback_stopped",
)
_KNOWN = frozenset(EVENTS)
_STOP = object()


class LatencyLog:
    def __init__(self, path: Path | str, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.turn = 0
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        self._writer = threading.Thread(target=self._write_loop, name="latency-log", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    def next_turn(self) -> int:
        self.turn += 1
        return self.turn

    def mark(self, event: str, at: float | None = None, **extra: object) -> float:
        """Record `event` now, or at an earlier perf_counter() time passed as `at`. Non-blocking."""
        now = time.perf_counter() if at is None else at
        if event not in _KNOWN:
            raise ValueError(f"Unknown latency event: {event}")
        if not self._closed:  # after close() there is no writer; drop quietly
            self._queue.put((self.turn, event, now, time.time(), extra))
        return now

    def flush(self, timeout: float = 5.0) -> None:
        """Block until every record marked so far is written to disk."""
        if self._closed:
            return
        done = threading.Event()
        self._queue.put(done)
        done.wait(timeout)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(_STOP)
        self._writer.join(timeout=5.0)
        atexit.unregister(self.close)

    def _write_loop(self) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            while True:
                item = self._queue.get()
                lines = []
                # Drain everything already queued, then write it in one go.
                while True:
                    if item is _STOP:
                        file.write("".join(lines))
                        file.flush()
                        return
                    if isinstance(item, threading.Event):
                        file.write("".join(lines))
                        file.flush()
                        lines = []
                        item.set()
                    else:
                        lines.append(self._format(item))
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                if lines:
                    file.write("".join(lines))
                    file.flush()

    def _format(self, item: tuple) -> str:
        turn, event, t, wall, extra = item
        record = {
            "run": self.run_id,
            "turn": turn,
            "event": event,
            "t": round(t, 6),
            "wall": datetime.fromtimestamp(wall).astimezone().isoformat(timespec="milliseconds"),
            **extra,
        }
        return json.dumps(record) + "\n"
