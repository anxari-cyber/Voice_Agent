"""Record your own voice commands for the WER test (roadmap Step 1.4, amendment #7).

Usage: python -m bench.record_commands --label wired      (or --label airpods, etc.)
       python -m bench.record_commands --label wired --start 12   (continue at command 12)

For each command: press Enter, then say the sentence shown. Recording starts when you speak
and stops 0.7 s after you finish (same VAD + join window as the agent). Parakeet's transcript is
shown straight away, so you can redo a take (type r + Enter) if something went wrong.

Files: bench/data/commands/<label>/NN.wav + manifest.jsonl (reference text, mic, timings).
The folder is git-ignored: these are recordings of your voice.
"""

from __future__ import annotations

import argparse
import json
import queue
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np

from config.settings import load_settings
from voice.audio_io import MicStream
from voice.vad import SAMPLE_RATE, SileroVAD

COMMANDS = [
    "Open my project folder.",
    "Check the git status.",
    "Run all the unit tests.",
    "What does the login function do?",
    "Fix the first bug you found.",
    "Show me the last three commits.",
    "Create a new branch called feature login.",
    "How many files are in the source folder?",
    "Read the README file and summarize it for me.",
    "Why is the build failing on Windows?",
    "Rename the variable user name to account name.",
    "Install the requests package with pip.",
    "Stop the task.",
    "Yes, go ahead.",
    "No, the other file.",
    "What time is it?",
    "Set a reminder for ten thirty tomorrow morning.",
    "The version should be three point five, not three point four.",
    "Delete the temporary files in the logs folder.",
    "Can you explain what a REST API is in simple words?",
    "Search the code for every place that calls the database.",
    "Add a docstring to the transcribe function.",
    "Commit the changes with the message update the tests.",
    "Undo the last change you made to settings dot py.",
    "How much memory is the Python process using right now?",
    "Open example dot com in the browser.",
    "Increase the timeout to fifteen seconds and try again.",
    "Tell me which tests are still failing after the fix.",
    "Write a short note for the team about what changed today.",
    "Thank you, that's all for now.",
]
PRE_ROLL_MS = 500
JOIN_WINDOW_MS = 700
MAX_SECONDS = 15.0


def save_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def record_one(mic: MicStream, vad: SileroVAD) -> tuple[np.ndarray, int] | None:
    """Wait for speech, record until JOIN_WINDOW_MS of silence. Returns (audio, speech_end_sample)."""
    while True:  # drop audio captured while the user was reading the prompt
        try:
            mic.read(timeout=0)
        except queue.Empty:
            break
    vad.reset()
    pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_MS // 20)
    chunks: list[np.ndarray] = []
    seen = origin = 0
    speech_end = None
    deadline = time.perf_counter() + 10
    while True:
        try:
            _, block = mic.read(timeout=0.2)
        except queue.Empty:
            continue
        seen += len(block)
        events = vad.process(block)
        if chunks:
            chunks.append(block)
        else:
            pre_roll.append(block)
        for event in events:
            if event.kind == "speech_start":
                if not chunks:
                    chunks = list(pre_roll)
                    origin = seen - sum(len(c) for c in chunks)
                speech_end = None  # speech again: the pause was mid-sentence
            elif event.kind == "speech_end" and chunks:
                speech_end = event.sample - origin
        if not chunks and time.perf_counter() > deadline:
            return None
        if chunks:
            length = sum(len(c) for c in chunks)
            if speech_end is not None and length - speech_end >= SAMPLE_RATE * JOIN_WINDOW_MS // 1000:
                return np.concatenate(chunks), speech_end
            if length >= SAMPLE_RATE * MAX_SECONDS:
                return np.concatenate(chunks), length


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="e.g. wired or airpods (the folder name)")
    parser.add_argument("--start", type=int, default=1, help="command number to start at")
    parser.add_argument("--no-check", action="store_true", help="don't transcribe each take")
    args = parser.parse_args()
    settings = load_settings()
    folder = Path("bench/data/commands") / args.label
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.jsonl"

    stt = None
    if not args.no_check:
        from voice.stt import ParakeetSTT

        print("Loading Parakeet to show each transcript...")
        stt = ParakeetSTT(Path(settings.models_dir) / "parakeet-tdt-0.6b-v2")
    mic = MicStream(device=settings.mic_device)
    vad = SileroVAD()
    mic.start()
    print(f"\nMic: {mic.device.name} ({mic.device.host_api}). Saving to {folder}\n"
          "Speak normally, as you would talk to the agent. Ctrl+C stops (continue later with --start).\n")
    try:
        index = args.start
        while index <= len(COMMANDS):
            text = COMMANDS[index - 1]
            input(f"[{index:2d}/{len(COMMANDS)}] Press Enter, then say:  {text}")
            result = record_one(mic, vad)
            if result is None:
                print("   No speech heard in 10 s. Try again.")
                continue
            audio, speech_end = result
            if stt is not None:
                print(f"   Heard: {stt.transcribe(audio[:speech_end])!r}")
            if input("   Enter = keep, r + Enter = redo: ").strip().lower() == "r":
                continue
            path = folder / f"{index:02d}.wav"
            save_wav(path, audio)
            record = {"file": path.name, "text": text, "label": args.label, "mic": mic.device.name,
                      "host_api": mic.device.host_api, "seconds": round(len(audio) / SAMPLE_RATE, 2),
                      "speech_end_sample": int(speech_end)}
            with manifest.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            index += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        mic.close()
    print(f"\nDone. Now run:  python -m bench.stt_bench {args.label}")


if __name__ == "__main__":
    main()
