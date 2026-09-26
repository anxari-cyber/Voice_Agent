"""Download the local models used by the voice pipeline (Phase 0.4).

Usage: python -m bench.download_models
Files already on disk are skipped, so it is safe to run again.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS_DIR = Path("models")
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v2-onnx"
PARAKEET_FILES = (
    "config.json",
    "vocab.txt",
    "nemo128.onnx",
    "decoder_joint-model.int8.onnx",
    "encoder-model.int8.onnx",
)
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"


def download(url: str, target: Path) -> None:
    if target.exists():
        print(f"  already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    partial = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.rename(target)


def main() -> None:
    print("Parakeet TDT 0.6B v2 (int8)")
    for name in PARAKEET_FILES:
        hf_hub_download(PARAKEET_REPO, name, local_dir=MODELS_DIR / "parakeet-tdt-0.6b-v2")
    print("Kokoro 82M")
    for name in KOKORO_FILES:
        download(f"{KOKORO_URL}/{name}", MODELS_DIR / "kokoro" / name)
    print("Smart Turn v3.2")
    hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_dir=MODELS_DIR / "smart_turn")
    print("Done.")


if __name__ == "__main__":
    main()
