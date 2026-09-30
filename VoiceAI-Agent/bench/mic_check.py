"""Live microphone check: shows your volume and whether the agent hears speech.

Usage: python -m bench.mic_check        (10 seconds; talk normally while it runs)
"""

from __future__ import annotations

import queue
import time

import numpy as np

from voice.audio_io import MicStream
from voice.device_manager import AudioManager
from voice.playback import Player
from voice.vad import SileroVAD


def main(seconds: float = 10.0) -> None:
    mic = MicStream()
    manager = AudioManager(mic, Player(sample_rate=24_000))
    manager.start(monitor=False)
    if not mic.is_open:
        print("No working microphone found.")
        return
    vad = SileroVAD()
    print(f"\nTalk now for {seconds:.0f} seconds...\n")
    mic.drain()
    peak_all, speech_blocks, end = 0.0, 0, time.perf_counter() + seconds
    shown = time.perf_counter()
    level = 0.0
    while time.perf_counter() < end:
        try:
            _, block = mic.read(timeout=0.2)
        except queue.Empty:
            continue
        level = max(level, float(np.abs(block).max()))
        peak_all = max(peak_all, level)
        vad.process(block)
        speech_blocks += vad.last_probability >= 0.5
        if time.perf_counter() - shown >= 0.25:
            bar = "#" * min(40, int(level * 80))
            talking = "SPEECH" if vad.last_probability >= 0.5 else "      "
            print(f"\r  volume {level:5.3f} |{bar:<40}| {talking}", end="", flush=True)
            level, shown = 0.0, time.perf_counter()
    manager.stop()
    print(f"\n\nLoudest: {peak_all:.3f}   Speech detected in {speech_blocks * 20 / 1000:.1f} s of audio")
    if peak_all < 0.02:
        print("=> Almost no signal: the mic isn't really connected (plug/jack), or it's muted / volume 0.")
    elif speech_blocks == 0:
        print("=> Sound arrives, but it isn't recognised as speech: too quiet or noisy. "
              "Raise the mic level in Windows (Sound settings -> Input -> volume).")
    else:
        print("=> The mic works. The agent should hear you.")


if __name__ == "__main__":
    main()
