"""Check which onnxruntime execution providers actually work on this machine.

Usage: python -m bench.check_onnx_gpu [path/to/model.onnx]

Without a path, the Silero VAD model bundled with faster-whisper is used.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PREFERRED = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")


def default_model() -> Path:
    import faster_whisper

    return Path(faster_whisper.__file__).parent / "assets" / "silero_vad_v6.onnx"


def dummy_inputs(session: ort.InferenceSession) -> dict[str, np.ndarray]:
    inputs = {}
    for arg in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in arg.shape]
        if arg.name == "input" and len(shape) == 2:
            shape = [1, 576]
        dtype = np.int64 if "int64" in arg.type else np.float32
        if arg.name == "sr":
            inputs[arg.name] = np.array(16_000, dtype=np.int64)
        else:
            inputs[arg.name] = np.zeros(shape, dtype=dtype)
    return inputs


def check(model: Path, provider: str, runs: int = 50) -> None:
    try:
        session = ort.InferenceSession(str(model), providers=[provider])
    except Exception as error:  # noqa: BLE001 - report any provider failure
        print(f"  {provider}: FAILED to create session ({error})")
        return
    active = session.get_providers()[0]
    if active != provider:
        print(f"  {provider}: not used (fell back to {active})")
        return
    feeds = dummy_inputs(session)
    try:
        session.run(None, feeds)
        started = time.perf_counter()
        for _ in range(runs):
            session.run(None, feeds)
        elapsed_ms = (time.perf_counter() - started) * 1000 / runs
    except Exception as error:  # noqa: BLE001
        print(f"  {provider}: FAILED at inference ({error})")
        return
    # onnxruntime silently retries on CPU when a GPU kernel fails, so check again after running.
    if session.get_providers()[0] != provider:
        print(f"  {provider}: FAILED at inference (silently fell back to CPU)")
        return
    print(f"  {provider}: OK, {elapsed_ms:.2f} ms per run")


def main() -> None:
    model = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model()
    if hasattr(ort, "preload_dlls") and "CUDAExecutionProvider" in ort.get_available_providers():
        ort.preload_dlls()  # load CUDA/cuDNN DLLs from the nvidia-* pip packages
    available = ort.get_available_providers()
    print(f"onnxruntime {ort.__version__} ({ort.get_device()})")
    print(f"Available providers: {available}")
    print(f"Model: {model}")
    for provider in PREFERRED:
        if provider in available:
            check(model, provider)


if __name__ == "__main__":
    main()
