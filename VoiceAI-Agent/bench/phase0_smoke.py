"""Phase 0 check: every model loads, and a quick speed comparison of GPU (DirectML) vs CPU.

Round trip: Kokoro speaks a sentence -> Parakeet and faster-whisper transcribe it back.
Usage: python -m bench.phase0_smoke
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path("models")
SENTENCE = "Please check my project and fix the failing login test."
PROVIDERS = {
    "GPU (DirectML)": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "CPU": ["CPUExecutionProvider"],
}


def session_options(providers: list[str], optimize: bool = True) -> ort.SessionOptions:
    options = ort.SessionOptions()
    if "DmlExecutionProvider" in providers:
        # DirectML requirements: no memory pattern, sequential execution.
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not optimize:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return options


def timed(fn, runs: int = 5) -> tuple[object, float]:
    result = fn()  # warm-up
    started = time.perf_counter()
    for _ in range(runs):
        result = fn()
    return result, (time.perf_counter() - started) * 1000 / runs


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    positions = np.linspace(0, len(audio) - 1, int(len(audio) * target_rate / source_rate))
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def check_kokoro() -> tuple[np.ndarray, int]:
    from kokoro_onnx import Kokoro

    print("\nKokoro TTS (6-word clause / full sentence)")
    audio, rate = None, 24_000
    variants = [("GPU (DirectML)", PROVIDERS["GPU (DirectML)"], True),
                ("GPU (DML basic)", PROVIDERS["GPU (DirectML)"], False),
                ("CPU", PROVIDERS["CPU"], True)]
    for label, providers, optimize in variants:
        session = ort.InferenceSession(
            str(MODELS / "kokoro" / "kokoro-v1.0.onnx"),
            sess_options=session_options(providers, optimize),
            providers=providers,
        )
        tts = Kokoro.from_session(session, str(MODELS / "kokoro" / "voices-v1.0.bin"))
        try:
            _, clause_ms = timed(lambda tts=tts: tts.create("Okay, I will check that.", voice="af_heart"))
            (audio, rate), full_ms = timed(lambda tts=tts: tts.create(SENTENCE, voice="af_heart"))
        except Exception as error:  # noqa: BLE001 - report and try the next provider
            print(f"  {label:<15} FAILED: {str(error).splitlines()[0][:120]}")
            continue
        seconds = len(audio) / rate
        print(f"  {label:<15} clause {clause_ms:6.0f} ms | sentence {full_ms:6.0f} ms "
              f"for {seconds:.1f}s audio (RTF {full_ms / 1000 / seconds:.2f})")
    return audio, rate


def check_parakeet(audio16k: np.ndarray) -> None:
    import onnx_asr

    print("\nParakeet TDT 0.6B v2 int8 STT")
    for label, providers in PROVIDERS.items():
        try:
            model = onnx_asr.load_model(
                "nemo-parakeet-tdt-0.6b-v2",
                MODELS / "parakeet-tdt-0.6b-v2",
                quantization="int8",
                providers=providers,
                sess_options=session_options(providers),
            )
            text, ms = timed(lambda model=model: model.recognize(audio16k, sample_rate=16_000))
        except Exception as error:  # noqa: BLE001
            print(f"  {label:<15} FAILED: {str(error).splitlines()[0][:120]}")
            continue
        print(f"  {label:<15} {ms:6.0f} ms for {len(audio16k) / 16_000:.1f}s audio -> {text!r}")


def check_whisper(audio16k: np.ndarray) -> None:
    from voice.transcriber import Transcriber

    print("\nfaster-whisper small.en (CUDA) for comparison")
    transcriber = Transcriber("small.en")
    text, ms = timed(lambda: transcriber.transcribe(audio16k))
    print(f"  {transcriber.device:<15} {ms:6.0f} ms -> {text!r}")


def check_smart_turn() -> None:
    print("\nSmart Turn v3.2")
    path = next((MODELS / "smart_turn").glob("*.onnx"))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {i.name: (i.shape, i.type) for i in session.get_inputs()}
    print(f"  loaded {path.name}, inputs: {inputs}")


def main() -> None:
    print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")
    audio, rate = check_kokoro()
    audio16k = resample(audio, rate, 16_000)
    check_parakeet(audio16k)
    check_whisper(audio16k)
    check_smart_turn()


if __name__ == "__main__":
    main()
