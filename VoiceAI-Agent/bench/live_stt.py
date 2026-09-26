"""Phase 3 live test: speak and watch the words appear while you talk.

Usage: python -m bench.live_stt [--engine parakeet|whisper] [--turns 10]

For every utterance it prints partial text while you speak, then the final text and
the time from the moment you stopped speaking to the final transcript.
"""

from __future__ import annotations

import argparse
import queue
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from config.settings import load_settings
from metrics.latency import LatencyLog
from voice.audio_io import MicStream
from voice.stt import SAMPLE_RATE, StreamingSTT, create_engine
from voice.vad import SileroVAD

PRE_ROLL_MS = 500  # audio kept from before VAD fired, so the first syllable isn't cut
# A pause shorter than this is part of the same sentence. The transcript is still decoded
# 200 ms after you stop (fast); it is only *confirmed* once this window passes in silence.
JOIN_WINDOW_MS = 700


def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=settings.stt_engine)
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args()

    latency = LatencyLog(settings.latency_log)
    print(f"Loading {args.engine}...")
    stt = StreamingSTT(create_engine(args.engine))
    vad = SileroVAD()
    mic = MicStream(device=settings.mic_device)
    mic.start()
    print(f"Mic: {mic.device.name}. Speak whenever you like (Ctrl+C to stop).\n")

    pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_MS // 20)
    gap: list[np.ndarray] = []  # audio during a pause that may still be mid-sentence
    samples_seen = 0
    finals: list[float] = []
    shown_partial = ""
    pending: tuple[str, float, float] | None = None  # (text, stopped_at, ms) awaiting confirmation
    try:
        while len(finals) < args.turns:
            try:
                arrived, block = mic.read(timeout=0.1)
            except queue.Empty:
                arrived, block = time.perf_counter(), None
            if block is not None:
                samples_seen += len(block)
                events = vad.process(block)
                if stt.active:
                    stt.feed(block)
                elif pending:
                    gap.append(block)
                else:
                    pre_roll.append(block)
            else:
                events = []
            for event in events:
                # When the event really happened, from the arrival time of this block.
                happened = arrived - (samples_seen - event.sample) / SAMPLE_RATE
                if event.kind == "speech_start":
                    if pending:  # short pause: same sentence, keep going
                        stt.resume(np.concatenate(gap) if gap else None)
                        pending, gap = None, []
                        continue
                    latency.next_turn()
                    latency.mark("speech_start", at=happened)
                    stt.start(np.concatenate(pre_roll) if pre_roll else None)
                    pre_roll.clear()
                    shown_partial = ""
                elif event.kind == "speech_end" and stt.active:
                    text = stt.finish()
                    done = time.perf_counter()
                    pending = (text, happened, (done - happened) * 1000)
            if pending and arrived - pending[1] >= JOIN_WINDOW_MS / 1000:
                text, stopped_at, ms = pending
                latency.mark("speech_end", at=stopped_at)
                latency.mark("stt_final", at=stopped_at + ms / 1000, engine=args.engine, text=text)
                finals.append(ms)
                save_wav(Path("logs") / f"utterance_{len(finals)}.wav", stt.audio())
                print(f"\r  FINAL (text ready {ms:.0f} ms after you stopped): {text}\n", flush=True)
                pending, gap = None, []
            if stt.active and stt.partial_text != shown_partial:
                shown_partial = stt.partial_text
                print(f"\r  ...{shown_partial}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.close()
        stt.close()

    if finals:
        values = np.array(finals)
        print(f"Speech end -> final text over {len(values)} turns: p50 {np.percentile(values, 50):.0f} ms, "
              f"p95 {np.percentile(values, 95):.0f} ms")
        print("Note: this includes the VAD's 200 ms silence wait before it decides you stopped.")


if __name__ == "__main__":
    main()
