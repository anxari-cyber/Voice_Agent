"""Timestamp every pipeline stage so latency can be measured instead of guessed.

Each call to `mark` appends one JSON line:
    {"turn": 3, "event": "stt_final", "t": 12.3456, "wall": "2026-09-26T14:02:11.123"}
`t` comes from time.perf_counter(), so only differences within one run are meaningful.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

# Stage events in pipeline order. Reports compute the gaps between these.
EVENTS = (
    "speech_start",
    "speech_end",
    "stt_final",
    "llm_first_token",
    "llm_done",
    "tts_first_audio",
    "playback_start",
    "bargein_detected",
    "playback_stopped",
)


class LatencyLog:
    def __init__(self, path: Path | str, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        self.turn = 0

    def next_turn(self) -> int:
        self.turn += 1
        return self.turn

    def mark(self, event: str, at: float | None = None, **extra: object) -> float:
        """Record `event` now, or at an earlier perf_counter() time passed as `at`."""
        if event not in EVENTS:
            raise ValueError(f"Unknown latency event: {event}")
        now = time.perf_counter() if at is None else at
        record = {
            "run": self.run_id,
            "turn": self.turn,
            "event": event,
            "t": round(now, 6),
            "wall": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        return now
