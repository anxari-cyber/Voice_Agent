"""Phase 2 exit test: the agent talks, you interrupt, the audio must stop in < 150 ms.

Usage: python -m bench.bargein_demo [--tries 20]

It plays a long spoken paragraph. Start talking at any moment: playback stops, the stop
latency is printed, and after a short pause the paragraph plays again. Use headphones so
the microphone does not hear the speaker. Results are also written to logs/latency.jsonl,
so `python -m bench.latency_report` shows p50/p95 afterwards.
"""

from __future__ import annotations

import argparse
import queue
import time

import numpy as np

from config.settings import load_settings
from metrics.latency import LatencyLog
from voice.audio_io import MicStream
from voice.playback import Player
from voice.vad import SileroVAD

PARAGRAPH = (
    "I am going to keep talking for a while so you can interrupt me. "
    "First I will look at the project structure, then I will read the failing test, "
    "and after that I will explain what I found and how we could fix it. "
    "Start speaking whenever you like, and I should go quiet immediately."
)


def synthesize(text: str) -> tuple[np.ndarray, int]:
    from voice.tts import kokoro_from_settings

    tts = kokoro_from_settings(load_settings())
    tts.warm_up()
    return np.concatenate(list(tts.synthesize_stream(text))), tts.sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tries", type=int, default=20)
    args = parser.parse_args()

    settings = load_settings()
    latency = LatencyLog(settings.latency_log)
    print("Preparing speech...")
    audio, rate = synthesize(PARAGRAPH)

    stopped_at: queue.Queue[float] = queue.Queue()

    def on_event(name: str, at: float) -> None:
        latency.mark(name, at=at)
        if name == "playback_stopped":
            stopped_at.put(at)

    player = Player(sample_rate=rate, device=settings.speaker_device, on_event=on_event)
    mic = MicStream(device=settings.mic_device)
    vad = SileroVAD(start_ms=160)  # ~150 ms of real speech before we count it as barge-in
    player.start()
    mic.start()
    print(f"Mic: {mic.device.name} | Speaker: {player.device.name}")

    results: list[float] = []
    try:
        for attempt in range(1, args.tries + 1):
            latency.next_turn()
            vad.reset()
            while True:  # drain audio captured during the pause
                try:
                    mic.read(timeout=0)
                except queue.Empty:
                    break
            print(f"\n[{attempt}/{args.tries}] Agent speaking... interrupt me!")
            player.play(audio)
            interrupted = False
            while player.is_playing:
                try:
                    _, block = mic.read(timeout=0.1)
                except queue.Empty:
                    continue
                if any(event.kind == "speech_start" for event in vad.process(block)):
                    detected = latency.mark("bargein_detected")
                    player.stop()
                    stop_time = stopped_at.get(timeout=1.0)
                    ms = (stop_time - detected) * 1000
                    results.append(ms)
                    print(f"  Interrupted -> audio stopped in {ms:.0f} ms")
                    interrupted = True
                    break
            if not interrupted:
                print("  (finished without interruption)")
            time.sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        player.close()

    if results:
        values = np.array(results)
        print(f"\nStop latency over {len(values)} interruptions: "
              f"p50 {np.percentile(values, 50):.0f} ms, p95 {np.percentile(values, 95):.0f} ms, "
              f"max {values.max():.0f} ms (target < 150 ms)")
        if mic.dropped_blocks:
            print(f"Warning: {mic.dropped_blocks} mic blocks were dropped.")


if __name__ == "__main__":
    main()
