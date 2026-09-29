"""Step 1.4 accuracy check: WER on your own recorded commands (from bench.record_commands).

Usage: python -m bench.stt_bench                 (every folder in bench/data/commands)
       python -m bench.stt_bench wired airpods   (compare mics: same sentences, different mic)
       python -m bench.stt_bench wired --whisper (also score faster-whisper, if installed)

For every recording: WER of a full Parakeet decode, and of the streaming final (StreamingSTT
fed in 20 ms blocks at audio-time scheduling, exactly as live). Target: WER < 8% (wired mic),
and the streaming final is not worse than the full decode. Text is normalised first (see
metrics/wer.py): case, punctuation and number formatting don't count as errors.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np

from config.settings import load_settings
from metrics.wer import WER, normalize, word_errors
from voice.stt import ParakeetSTT, StreamingSTT

ROOT = Path("bench/data/commands")


def load_folder(folder: Path) -> list[dict]:
    entries: dict[str, dict] = {}
    manifest = folder / "manifest.jsonl"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            entries[record["file"]] = record  # a re-recorded command: the last take wins
    for record in entries.values():
        with wave.open(str(folder / record["file"]), "rb") as file:
            frames = file.readframes(file.getnframes())
        record["audio"] = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768
    return sorted(entries.values(), key=lambda r: r["file"])


def streaming_final(stt: StreamingSTT, audio: np.ndarray, speech_end: int) -> tuple[str, float]:
    stt.start()
    for i in range(0, len(audio), 320):
        stt.feed(audio[i : i + 320])
    started = time.perf_counter()
    text = stt.finish(speech_end_sample=speech_end)
    return text, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="*", help="folders under bench/data/commands")
    parser.add_argument("--whisper", action="store_true", help="also score faster-whisper small.en")
    parser.add_argument("--show", action="store_true", help="print every transcript")
    args = parser.parse_args()
    labels = args.labels
    if not labels and ROOT.exists():
        labels = sorted(p.name for p in ROOT.iterdir() if (p / "manifest.jsonl").exists())
    if not labels:
        raise SystemExit("No recordings yet. Run:  python -m bench.record_commands --label wired")

    settings = load_settings()
    engine = ParakeetSTT(Path(settings.models_dir) / "parakeet-tdt-0.6b-v2")
    stt = StreamingSTT(engine, final_engine=engine, synchronous=True, trim_after_s=settings.stt_trim_after_s)
    whisper = None
    if args.whisper:
        from voice.stt import WhisperSTT

        whisper = WhisperSTT()

    summary = {}
    for label in labels:
        records = load_folder(ROOT / label)
        totals = {"full": WER(0, 0), "streaming": WER(0, 0), "whisper": WER(0, 0)}
        final_ms = []
        print(f"\n== {label}: {len(records)} recordings, mic {records[0]['mic']!r} ({records[0]['host_api']})")
        for record in records:
            speech = record["audio"][: record["speech_end_sample"]]
            full = engine.transcribe(speech)
            streamed, ms = streaming_final(stt, record["audio"], record["speech_end_sample"])
            final_ms.append(ms)
            full_wer = word_errors(record["text"], full)
            stream_wer = word_errors(record["text"], streamed)
            totals["full"] += full_wer
            totals["streaming"] += stream_wer
            line = f"  {record['file']}  full {full_wer.errors}/{full_wer.words}  streaming {stream_wer.errors}"
            if whisper:
                whisper_wer = word_errors(record["text"], whisper.transcribe(speech))
                totals["whisper"] += whisper_wer
                line += f"  whisper {whisper_wer.errors}"
            if args.show or full_wer.errors or stream_wer.errors:
                line += f"\n      ref: {normalize(record['text'])}\n      hyp: {normalize(streamed)}"
            print(line)
        summary[label] = totals
        print(f"  WER full decode {totals['full'].rate:.1%} | streaming final {totals['streaming'].rate:.1%}"
              + (f" | whisper {totals['whisper'].rate:.1%}" if whisper else "")
              + f"  (target < 8%) | synchronous final decode p50 {np.median(final_ms):.0f} ms")

    if len(summary) > 1:
        print("\nMic comparison (same sentences, Parakeet streaming final):")
        for label, totals in summary.items():
            print(f"  {label:<12} WER {totals['streaming'].rate:.1%}")
        best = min(summary, key=lambda k: summary[k]["streaming"].rate)
        for label, totals in summary.items():
            if label != best:
                share = totals["streaming"].rate - summary[best]["streaming"].rate
                print(f"  {label} vs {best}: +{share:.1%} WER is the mic's share")


if __name__ == "__main__":
    main()
