"""Phase 0 follow-up: Kokoro-82M through PyTorch on the GPU vs CPU.

Usage: python -m bench.kokoro_gpu_bench [--play]
Target: first audio for a short clause in < 120 ms.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

WEIGHTS = Path("models/kokoro-torch")
CLAUSES = ("Sure.", "Okay, I will check that.", "The login test fails because the token expired.")


def load(device: str):
    from kokoro import KModel, KPipeline

    model = KModel(
        repo_id="hexgrad/Kokoro-82M",
        config=str(WEIGHTS / "config.json"),
        model=str(WEIGHTS / "kokoro-v1_0.pth"),
    ).to(device).eval()
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=model)
    voice = pipeline.load_voice(str(WEIGHTS / "voices" / "af_heart.pt"))
    return pipeline, voice


def speak(pipeline, voice, text: str) -> np.ndarray:
    chunks = [result.audio for result in pipeline(text, voice=voice) if result.audio is not None]
    return torch.cat(chunks).cpu().numpy() if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play", action="store_true", help="Play the last clause through the speaker")
    args = parser.parse_args()

    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    audio = np.zeros(0, dtype=np.float32)
    for device in devices:
        pipeline, voice = load(device)
        with torch.inference_mode():
            speak(pipeline, voice, "Warm up.")  # first run compiles kernels
            print(f"\n{device.upper()}")
            for text in CLAUSES:
                runs = []
                for _ in range(5):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    audio = speak(pipeline, voice, text)
                    runs.append((time.perf_counter() - started) * 1000)
                seconds = len(audio) / 24_000
                print(f"  {np.median(runs):6.0f} ms  for {seconds:.1f}s of speech  {text!r}")
        if device == "cuda":
            print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    if args.play:
        from voice.playback import Player

        player = Player(sample_rate=24_000)
        player.start()
        player.play(audio)
        player.wait_until_done()
        time.sleep(0.2)
        player.close()


if __name__ == "__main__":
    main()
