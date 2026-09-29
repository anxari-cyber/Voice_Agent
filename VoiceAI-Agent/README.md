# VoiceAI-Agent

A local, low-latency voice agent for Windows. Everything runs on your own PC with open-source
models: no cloud APIs and no API keys, and it works offline after setup. You can interrupt it
while it talks.

- **Plan:** [../plan_v3_roadmap.md](../plan_v3_roadmap.md)
- **Measured results so far:** [../implementation.md](../implementation.md)

## How it works

```
Mic (always on) → Silero VAD → Parakeet STT (live) → Qwen3-4B via Ollama (streaming)
      ↑                                                      ↓
      └──── barge-in stops playback ←── Speaker ←── Kokoro TTS (per clause)
```

| Part | Model | Runs on | Measured |
|---|---|---|---|
| Voice activity detection | Silero VAD v6 (bundled in `voice/assets/`) | CPU | < 1 ms per 32 ms window |
| Speech-to-text | NVIDIA Parakeet TDT 0.6B v2 (int8 ONNX) | CPU | 65–130 ms per utterance |
| LLM ("Voice Brain") | Qwen3-4B-Instruct-2507 Q4_K_M via Ollama | GPU | 120 tokens/s |
| Text-to-speech | Kokoro-82M (PyTorch) | GPU | 56–72 ms per clause |
| Turn detection | Smart Turn v3.2 (ONNX) | CPU | (roadmap Step 2.1) |

Tested on an RTX 5050 8 GB (Blackwell), a Ryzen 5 7500F, 32 GB RAM and Windows 11.

## Install

You need **Python 3.10–3.12**, **[Ollama](https://ollama.com/download)**, and about **6 GB** of
free disk space (PyTorch ≈ 4 GB, models ≈ 1.3 GB, LLM 2.5 GB).
The **order matters**. Follow these steps exactly:

```powershell
cd VoiceAI-Agent
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1. PyTorch with CUDA 12.8 FIRST. It supports RTX 50-series GPUs.
#    Without this step pip installs a CPU-only torch from PyPI.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. The project and its dependencies
python -m pip install -e ".[dev]"

# 3. Models (Parakeet, Kokoro, Smart Turn, and a small English model for pronunciation)
python -m bench.download_models

# 4. The LLM
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

**Why onnxruntime-directml?** The CUDA build of onnxruntime can't run on RTX 50-series GPUs
yet. DirectML works on any Windows GPU. Only **one** onnxruntime package may be installed at a
time.

**Slow download from download.pytorch.org?** Download the `.whl` file for your Python version
from `https://download.pytorch.org/whl/cu128/torch/` with any download manager, then run
`python -m pip install path\to\torch-...whl`.

### Optional: Whisper as a backup STT engine

```powershell
python -m pip install -e ".[whisper]"
# faster-whisper pulls in the plain "onnxruntime" package. Put DirectML back:
python -m pip uninstall -y onnxruntime onnxruntime-directml
python -m pip install onnxruntime-directml
```

Then set `VOICEAI_STT_ENGINE=whisper` in `.env`.

### Settings

Copy `.env.example` to `.env`. Every value in it is the default, so only change what you need.
Useful ones:

- `VOICEAI_MIC_DEVICE` / `VOICEAI_SPEAKER_DEVICE`: part of a device name, or an index. Leave
  them empty to use the Windows default.
- `VOICEAI_MODELS_DIR`: put the models on another drive, e.g. `D:/voiceai/models`.

**Use a wired headset if you can.** A Bluetooth headset's microphone switches to phone-call
quality, which causes wrong words. Headphones also stop the agent from hearing, and
interrupting, its own voice.

## Run

```powershell
python -m app.main --list-devices     # show microphones and speakers
python -m bench.live_stt              # live speech-to-text: speak and watch the words appear
python -m bench.bargein_demo          # the agent talks; start speaking and it stops
python -m bench.kokoro_gpu_bench      # TTS speed, GPU vs CPU
python -m bench.latency_report        # p50 / p95 per stage from logs/latency.jsonl
voiceai run                           # one voice question, answered by the local LLM (text for now)
```

`voiceai run` is a temporary bridge. The full streaming conversation loop (talk → answer out
loud → interrupt) is roadmap Step 1.5.

## Development

```powershell
python -m pytest        # no microphone or .env needed
python -m ruff check .
```

Third-party files and model licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
