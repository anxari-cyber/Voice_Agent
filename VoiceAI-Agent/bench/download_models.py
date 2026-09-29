"""Download the local models used by the voice pipeline (Phase 0.4).

Usage: python -m bench.download_models
Files already on disk are skipped, so it is safe to run again.
"""

from __future__ import annotations

from huggingface_hub import hf_hub_download

from config.settings import load_settings

MODELS_DIR = load_settings().models_dir
KOKORO_TORCH_REPO = "hexgrad/Kokoro-82M"
KOKORO_TORCH_FILES = ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt")
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


def main() -> None:
    print("Parakeet TDT 0.6B v2 (int8)")
    for name in PARAKEET_FILES:
        hf_hub_download(PARAKEET_REPO, name, local_dir=MODELS_DIR / "parakeet-tdt-0.6b-v2")
    print("Kokoro 82M (PyTorch, used by the pipeline)")
    for name in KOKORO_TORCH_FILES:
        hf_hub_download(KOKORO_TORCH_REPO, name, local_dir=MODELS_DIR / "kokoro-torch")
    print("Smart Turn v3.2")
    hf_hub_download(SMART_TURN_REPO, SMART_TURN_FILE, local_dir=MODELS_DIR / "smart_turn")
    import spacy.util

    if not spacy.util.is_package("en_core_web_sm"):  # used by Kokoro's English phonemizer
        import spacy.cli

        spacy.cli.download("en_core_web_sm")
    print("Done.")


if __name__ == "__main__":
    main()
