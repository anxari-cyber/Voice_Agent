# Implementation Plan — Low-Latency Local Voice Agent

This file is the step-by-step guide we follow. Architecture: see [plan_v2_low_latency.md](plan_v2_low_latency.md).
Each phase has a **Done when** check. We do not start the next phase until it passes.

## What's on the PC now (checked 2026-09-26)

| Item | Status |
|---|---|
| Python venv | 3.10.11 |
| faster-whisper 1.2.1 + ctranslate2 4.8.2 | ✅ installed, **CUDA works** (1 device) |
| faster-whisper `small.en` model | ✅ downloaded |
| Silero VAD v6 ONNX | ✅ already bundled inside faster-whisper (`faster_whisper/assets/silero_vad_v6.onnx`), no download needed |
| onnxruntime 1.23.2 | ✅ installed, **CPU only** (no GPU provider) |
| sounddevice 0.5.6, numpy 2.2.6, httpx | ✅ |
| **Ollama 0.33.3** | ✅ installed. Models: `qwen3:14b`, `qwen2.5vl:7b` |
| **Qwen3-4B-Instruct-2507** | ❌ **not installed**. Available as `qwen3:4b-instruct-2507-q4_K_M` (2.5 GB) |
| llama-server (llama.cpp) | ❌ not installed, and not needed (see LLM decision) |
| Parakeet, Kokoro, smart-turn | ❌ not installed |
| GPU / driver | RTX 5050 (Blackwell, sm_120), driver CUDA 13.1 |
| Free disk on C: | ⚠️ **29 GB** |

## Tool decisions

| Part | Choice | Why | Backup |
|---|---|---|---|
| **STT** | **NVIDIA Parakeet-TDT-0.6B-v2** (English) via `onnx-asr` | More accurate *and* much faster than Whisper for English. TDT decoding has no beam search and no hallucination loops, and it gives word timestamps. | faster-whisper `small.en`, already installed and working on CUDA |
| **TTS** | **Kokoro-82M** via **PyTorch (CUDA 12.8)** *(changed in Phase 0: onnxruntime can't run Kokoro on the RTX 5050 GPU)* | Natural, human-like voice at a very small size (82M), fast, with a streaming API (`create_stream`). Piper is faster on CPU but sounds robotic. | Piper |
| **LLM** | **Qwen3-4B-Instruct-2507, Q4_K_M, via Ollama** | Ollama is already installed and runs llama.cpp inside. It streams, `keep_alive` keeps the model in VRAM, and closing the HTTP stream cancels generation (needed for barge-in). It's non-thinking, so it answers quickly. | llama-server, only if the benchmark shows Ollama's time to first token is over 150 ms |
| VAD | Silero v6 (already bundled) | Already on disk | none |
| Turn detection | Pipecat smart-turn v3 (ONNX, small) | Semantic end-of-turn detection, local | silence-only rule |

**Known risk: GPU support for onnxruntime on the RTX 50 series.**
- The official `onnxruntime-gpu` wheels have historically lacked sm_120 (Blackwell) kernels. See [onnxruntime #26177](https://github.com/microsoft/onnxruntime/issues/26177) and [#27875](https://github.com/microsoft/onnxruntime/issues/27875).
- Parakeet and Kokoro both run on onnxruntime, so Phase 0 **tests this first** and picks the fastest option that works:
  1. `onnxruntime-gpu` (CUDA)
  2. `onnxruntime-directml` (any GPU on Windows)
  3. CPU int8
- The winner is written to `.env`. Phase 3 has a benchmark gate: if Parakeet on this PC can't beat faster-whisper on CUDA, we use faster-whisper.

**Disk:**
- The voice stack needs about 5–6 GB.
- The Worker model in Phase 9 (Qwen3-Coder-30B-A3B, ~19 GB) would leave very little free space. Before Phase 9, free space or delete unused Ollama/HF models, e.g. `qwen2.5vl:7b` (6 GB), with the user's OK.

---


## Working rules

1. **One phase at a time.** A phase is finished only when its **Done when** check passes.
2. **Git:** create a branch `feature/low-latency-voice` from `main`. Commit after each step with a clear message. Ask the user before pushing.
3. **Every step:** code → `pytest` → `ruff check` → a short manual run.
4. **Every component sits behind a small interface** (`STT`, `TTS`, `LLM`, `VAD`), so it can be swapped by changing `.env` only.
5. **Everything is local.** Gemini is removed from the pipeline in Phase 4.
6. **Every phase that affects speed records numbers** in `logs/latency.jsonl`. Results go into a "Results" table at the bottom of `implementation.md`.

## Phase 0: Setup and verification

**0.1 Create the branch**
- Create `feature/low-latency-voice`.

**0.2 Install the LLM**
- Run `ollama pull qwen3:4b-instruct-2507-q4_K_M`.
- Test it: `ollama run … "hi"`, then `ollama ps` should show it at 100% GPU.

**0.3 GPU check for onnxruntime**
- Write `bench/check_onnx_gpu.py`, which lists providers and runs a small model on each.
- Try the three options in order:
  1. Swap `onnxruntime` → `onnxruntime-gpu`. They can't both be installed, and `onnxruntime-gpu` also includes CPU.
  2. If CUDA fails on sm_120, try `onnxruntime-directml`.
  3. Otherwise stay on CPU.
- Record the winner.

**0.4 Install the packages and models**
- Packages: `onnx-asr[hub]` and `kokoro-onnx`.
- Downloads:
  - Parakeet v2 model (via `onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")`; confirm the exact name, and the int8 variant if we're on CPU)
  - `kokoro-v1.0.onnx` + `voices-v1.0.bin` → `models/kokoro/`
  - smart-turn v3 ONNX → `models/smart_turn/`
- Add all of them to `pyproject.toml`. Remove `google-genai` in Phase 4.

**Done when:** every model loads once in a Python shell without errors, and the GPU/CPU result is noted.

## Phase 1: Fixes, mic selection, latency logging (R0)

**1.1 Mic selection** in `voice/audio_devices.py`
- Default is the Windows default input.
- Add an optional `VOICEAI_MIC_DEVICE` (name or index) and `VOICEAI_SPEAKER_DEVICE`.
- When a device appears under several host APIs, prefer WASAPI.
- If the chosen device is missing, show a clear warning and fall back to the default.
- Add a `--list-devices` command.
- Remove the hardcoded "AirPods Pro".

**1.2 Fix the `main.py` crash**
- Remove `ollama_url`.
- `voiceai` becomes the real entry point: `voiceai run`, `voiceai --list-devices`.

**1.3 Transcriber fixes**
- Make the segment filter match faster-whisper's own rule: drop a segment only if `no_speech_prob` is high **and** `avg_logprob` is low.
- Only fall back to CPU on CUDA-specific errors.

**1.4 Latency logging** in `metrics/latency.py`
- Log events with `time.perf_counter()` → `logs/latency.jsonl`.
- Add `bench/latency_report.py`, which prints p50/p95 for each stage.

**1.5 Settings**
- Add all new keys to `config/settings.py` and `.env.example`.
- Make tests independent of the user's `.env`.

**Done when:**
- `voiceai --list-devices` works.
- `voiceai run` works with any mic.
- The baseline latency of the current pipeline is recorded.
- Tests pass.

## Phase 2: Full-duplex audio + VAD + instant stop (R1)

**2.1 Audio I/O** in `voice/audio_io.py`
- One long-running mic `InputStream` and one speaker `OutputStream`, both open at the same time.
- 16 kHz mono, 20 ms frames (320 samples), WASAPI.

**2.2 VAD** in `voice/vad.py`
- Silero v6 ONNX (the bundled file), fed with the 512-sample windows Silero expects.
- Emits `speech_start` and `speech_end` events.
- Noise calibration happens once at startup.

**2.3 Player** in `voice/playback.py`
- Queue-based.
- `stop()` empties the queue within one audio callback (~20 ms).
- Counts the samples actually played, needed later for context repair.

**2.4 Demo** `bench/bargein_demo.py`
- Plays a long WAV. Talking into the mic stops it.

**Done when:** stop latency is **< 150 ms** over 20 tries, measured from `bargein_detected` to `playback_stopped`.

## Phase 3: STT, Parakeet streaming (R2)

**3.1 Interface** `voice/stt.py`
- `STT` interface with `feed(frames)`, `partial()` and `final()`.
- Two implementations: `ParakeetSTT` and `WhisperSTT` (the existing code, wrapped).

**3.2 Streaming**
- While you speak, decode the growing buffer every ~300 ms to produce partial text.
- At `speech_end`, run the final decode.

**3.3 Benchmark** `bench/stt_bench.py`
- The user records 30 of their own commands into `bench/data/` (a script helps).
- Compare Parakeet and Whisper on accuracy (WER), final latency and VRAM.

**Done when:**
- Final text arrives **< 100 ms** after speech ends, and WER is **< 8%**.
- The winner is set in `.env` (`VOICEAI_STT_ENGINE`).

## Phase 4: Voice Brain LLM + conversation memory (R3)

**4.1 LLM client** `agent/llm.py`
- Streaming client for Ollama `/api/chat`.
- Reuses one persistent `httpx.Client`.
- `keep_alive: -1` keeps the model loaded.
- `think: false`.
- Can be cancelled: closing the stream stops generation.
- Pre-warms the model at startup.

**4.2 Prompt builder** `agent/prompt.py`
- Combines the system rules (short, speakable answers, no markdown), the last N turns, and the new message.
- The system prompt stays identical every time, so Ollama/llama.cpp can reuse the cached prefix.

**4.3 Memory** `agent/memory.py`
- Conversation history.
- `truncate_last_assistant(spoken_text)` keeps only what was actually spoken, for barge-in.

**4.4 Clean-up**
- Remove Gemini: `agent/gemini_client.py`, the `google-genai` dependency and its settings.
- Update the tests.

**4.5 Benchmark** `bench/llm_bench.py`
- Measures time to first token (TTFT) and tokens per second over 30 prompts.

**Done when:** TTFT is **< 150 ms** with a warm model, and speed is **> 60 tok/s**. If not, try llama-server.

## Phase 5: TTS, Kokoro streaming + clause chunker (R4)

**5.1 Interface** `voice/tts.py`
- `TTS` interface with `synthesize_stream(text)` → audio chunks.
- `KokoroTTS` (default voice `af_heart`, resampled to the output device rate).
- `PiperTTS` is a later backup.

**5.2 Clause chunker** `pipeline/chunker.py`
- Cuts the token stream at `. ! ? , ; :`.
- Minimum ~4 words, and the first chunk is kept short to speed up first audio.

**5.3 Benchmark** `bench/tts_bench.py`
- First audio for a 6-word clause, real-time factor, and the user's 1–5 score for how natural it sounds.

**Done when:** first audio is **< 120 ms** and the voice quality score is **≥ 4**.

## Phase 6: The full streaming pipeline (R5)

**6.1 Event bus** `pipeline/bus.py`
- asyncio queues with typed events:
  - `SpeechStart`, `SpeechEnd`, `PartialText`, `FinalText`
  - `LLMToken`, `Clause`, `AudioChunk`
  - `BargeIn`

**6.2 Orchestrator** `pipeline/orchestrator.py`
- State machine: LISTENING → THINKING → SPEAKING → LISTENING.
- Each stage runs as its own asyncio task.
- Blocking model calls go through `run_in_executor`.

**6.3 Entry point**
- `voiceai run` starts one long-running process. Models load once.

**6.4 Test** `bench/conversation_test.py`
- Plays 50 pre-recorded turns through VB-Cable (a virtual audio cable) so results are repeatable.

**Done when:** latency from stopping speaking to first audio is **p50 < 800 ms** over 50 turns.

## Phase 7: Turn detection + barge-in + context repair (R6)

**7.1 Turn detector** `voice/turn_detector.py`
- Combines VAD silence (~200 ms) with smart-turn.
- If the sentence isn't finished, waits up to ~700 ms.

**7.2 Barge-in** `pipeline/bargein.py`
- While SPEAKING, if the user speaks for ≥150–200 ms and the first partial contains real words:
  - `player.stop()`,
  - cancel the LLM stream,
  - clear the TTS queue.
- Then:
  - `memory.truncate_last_assistant(played_text)`,
  - go back to LISTENING.

**7.3 Played-text tracking**
- Map played samples to their clause, so we know exactly what was said.

**Done when:** in 20 scripted interruptions:
- 100% stop in **< 150 ms**,
- **0** false interruptions from the agent's own voice (with headphones),
- follow-up answers are correct.

## Phase 8: Latency tricks (R7)

**8.1 Speculative LLM start**
- Start the LLM when the pause begins. Cancel it if the user keeps talking.

**8.2 Filler cache**
- Pre-synthesise acknowledgements ("Okay, checking that…") at startup.

**8.3 Keep-warm pings and threshold tuning**
- Tune thresholds using the latency data.

**Done when:** p50 is **< 600 ms**.

## Phase 9: Worker Agent (R8, planned in detail when we get here)

- OpenHands SDK and a coding model running in the background.
  - Qwen3-Coder-30B-A3B if disk space allows. Otherwise the already-downloaded `qwen3:14b`.
- Tools: files, terminal, git, tests.
- Permissions: Level 1/2/3, with voice confirmation.
- Progress events are spoken by the Voice Brain.

**Done when:** voice latency p95 is unchanged while the Worker runs.

## Phase 10 and later

- Long-term memory, GUI, advanced tools.

## Results (filled in during implementation)

| Phase | Metric | Target | Measured |
|---|---|---|---|


| 0 | Qwen3-4B placement | 100% GPU | ✅ 100% GPU, 3.2 GB VRAM |
| 0 | Qwen3-4B generation speed | > 60 tok/s | ✅ 120 tok/s |
| 0 | Qwen3-4B warm prompt processing | — | 11 ms (cold load 9.4 s, once at startup) |
| 0 | onnxruntime CUDA on RTX 5050 | works | ❌ session loads, but the first Conv fails in cuDNN and silently falls back to CPU |
| 0 | onnxruntime DirectML on RTX 5050 | works | ✅ works for Parakeet. ❌ Kokoro fails (ConvTranspose error) |
| 0 | **Parakeet v2 int8, DirectML** (3.4 s clip) | < 100 ms | ✅ **74 ms**, perfect transcript |
| 0 | Parakeet v2 int8, CPU | — | 104 ms, perfect transcript |
| 0 | faster-whisper small.en, CUDA | — | 108 ms, perfect transcript |
| 0 | Kokoro ONNX fp32, CPU, 6-word clause | < 120 ms | ❌ 307 ms (int8 is worse: 1120 ms) |
| 0 | Smart Turn v3.2 (CPU) | loads | ✅ input `[batch, 80, 800]` mel features |

### Phase 0 notes

- `onnxruntime-directml` replaces the CPU `onnxruntime` package. Only one onnxruntime package can be installed at a time.
- `kokoro-onnx` is installed with `--no-deps` so it doesn't pull the CPU `onnxruntime` back in. Its other dependencies (`espeakng-loader`, `phonemizer-fork`) are installed separately.
- Parakeet starts with the **int8** model (~670 MB) because the fp32 model is 2.4 GB. If int8 isn't fast enough on DirectML, Phase 3 downloads fp32.
- **Decision (user):** Kokoro sounds much more natural than Piper, so we keep Kokoro and run it through **PyTorch CUDA 12.8**, which supports Blackwell (sm_120). This costs a ~3 GB download.
- **STT confirmed:** Parakeet on DirectML beats faster-whisper on CUDA. faster-whisper stays as the backup.
- Silero VAD stays on the CPU (0.08 ms per frame, which is faster than the GPU for such a tiny model).
