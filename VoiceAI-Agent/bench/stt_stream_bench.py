"""Step 1.4 benchmark: final-transcript latency for short and long utterances, fed in real time.

Usage: python -m bench.stt_stream_bench [--runs 3]

Speech is synthesised with Kokoro from known sentences (so the reference text is exact), then
fed to StreamingSTT in 20 ms blocks at real-time speed: partials run in the background exactly
like live. After the last word, 200 ms of silence is fed (the VAD's end wait), then finish().
Reports: final latency (target < 100 ms for 3 s and 20 s), how the final was produced
(reused / tail / window), WER of the streaming final vs the reference, and vs a full decode.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from config.settings import load_settings
from metrics.wer import word_errors
from voice.stt import SAMPLE_RATE, StreamingSTT, streaming_from_settings
from voice.tts import kokoro_from_settings

SHORT = ["Please open my project and check the login test."]
LONG = ["Please open my project and check why the login test is failing.",
        "After that, run all the unit tests again and tell me which ones still fail.",
        "Then look at the git history for the last week and summarize the changes.",
        "If the database migration is the problem, roll it back and try once more.",
        "Finally, write a short note for the team about what you found and fixed."]


def speech_for(tts, sentences: list[str]) -> np.ndarray:
    parts = []
    for text in sentences:
        audio = np.concatenate(list(tts.synthesize_stream(text)))
        pos = np.linspace(0, len(audio) - 1, int(len(audio) * SAMPLE_RATE / tts.sample_rate))
        parts.append(np.interp(pos, np.arange(len(audio)), audio).astype(np.float32))
    return np.concatenate(parts)


def run(stt: StreamingSTT, speech: np.ndarray, tail_ms: int = 200) -> tuple[str, float, dict]:
    """tail_ms: silence fed after the last word (200 = the VAD's end wait; 0 = worst case,
    finish() right at the last word, before any partial can have covered it)."""
    block = SAMPLE_RATE // 50  # 20 ms
    silence = np.zeros(SAMPLE_RATE * tail_ms // 1000, dtype=np.float32)
    audio = np.concatenate([speech, silence])
    stt.start()
    started = time.perf_counter()
    for i in range(0, len(audio), block):
        target = started + i / SAMPLE_RATE
        delay = target - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        stt.feed(audio[i : i + block])
    t = time.perf_counter()
    text = stt.finish(speech_end_sample=len(speech))
    return text, (time.perf_counter() - t) * 1000, dict(stt.last_final)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    settings = load_settings()
    tts = kokoro_from_settings(settings)
    tts.warm_up()
    stt = streaming_from_settings(settings)  # exactly what the app uses
    final_engine = stt.final_engine
    for label, sentences, tail_ms in (("short", SHORT, 200), ("long", LONG, 200),
                                      ("short, forced tail", SHORT, 0), ("long, forced tail", LONG, 0)):
        speech = speech_for(tts, sentences)
        reference = " ".join(sentences)
        full_started = time.perf_counter()
        full_text = final_engine.transcribe(speech)
        full_ms = (time.perf_counter() - full_started) * 1000
        print(f"\n{label}: {len(speech) / SAMPLE_RATE:.1f} s of speech. Full re-decode would take "
              f"{full_ms:.0f} ms, WER vs reference {word_errors(reference, full_text).rate:.1%}")
        latencies = []
        for i in range(args.runs):
            text, ms, info = run(stt, speech, tail_ms)
            latencies.append(ms)
            print(f"  run {i + 1}: final in {ms:5.1f} ms ({info['mode']}, decoded {info['decoded_s']:.2f} s) | "
                  f"WER vs reference {word_errors(reference, text).rate:.1%}, "
                  f"vs full decode {word_errors(full_text, text).rate:.1%}")
        print(f"  final latency p50 {np.median(latencies):.0f} ms, max {max(latencies):.0f} ms (target < 100 ms)")
        print(f"  final text: {text!r}")
    stt.close()


if __name__ == "__main__":
    main()
