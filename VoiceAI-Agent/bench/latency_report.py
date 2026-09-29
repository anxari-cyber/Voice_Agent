"""Print p50/p95 latency per pipeline stage from logs/latency.jsonl.

Usage: python -m bench.latency_report [path] [--run RUN_ID]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# (label, from event, to event)
STAGES = (
    ("End of speech -> final text (STT)", "speech_end", "stt_final"),
    ("Final text -> first LLM token", "stt_final", "llm_first_token"),
    ("Final text -> full LLM reply", "stt_final", "llm_done"),
    ("First LLM token -> first TTS audio", "llm_first_token", "tts_first_audio"),
    ("First TTS audio -> playback start", "tts_first_audio", "playback_start"),
    ("Soft end -> speculative LLM start", "turn_soft_end", "speculative_start"),
    ("Soft end -> turn commit", "turn_soft_end", "turn_commit"),
    ("Turn commit -> playback start", "turn_commit", "playback_start"),
    ("TOTAL: end of speech -> first audio", "speech_end", "playback_start"),
    ("Barge-in -> playback stopped", "bargein_detected", "playback_stopped"),
    ("Barge-in -> LLM cancelled", "bargein_detected", "llm_cancel"),
    ("Barge-in -> TTS cancelled", "bargein_detected", "tts_cancel"),
)


def load(path: Path, run: str | None) -> dict[tuple[str, int], dict[str, float]]:
    turns: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if run and record["run"] != run:
            continue
        # Keep the first occurrence of each event within a turn.
        turns[(record["run"], record["turn"])].setdefault(record["event"], record["t"])
    return turns


def report(turns: dict[tuple[str, int], dict[str, float]]) -> str:
    lines = [f"{'Stage':<40} {'n':>4} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}"]
    for label, start, end in STAGES:
        gaps = [
            (events[end] - events[start]) * 1000
            for events in turns.values()
            if start in events and end in events
        ]
        if not gaps:
            continue
        values = np.array(gaps)
        lines.append(
            f"{label:<40} {len(values):>4} {np.percentile(values, 50):>8.0f} "
            f"{np.percentile(values, 95):>8.0f} {values.max():>8.0f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="logs/latency.jsonl")
    parser.add_argument("--run", default=None, help="Only include one run id")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"No latency log at {path}. Run 'voiceai run' first.")
    print(report(load(path, args.run)))


if __name__ == "__main__":
    main()
