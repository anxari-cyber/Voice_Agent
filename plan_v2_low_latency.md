# Improved R&D Plan: Fully Local, Very Low-Latency Voice Agent with Interruption

## Context

- **Your current plan** ([plan.md](plan.md), [local_voice_ai_agent_complete_plan.md](local_voice_ai_agent_complete_plan.md)) is a waterfall pipeline: faster-whisper → Qwen3-Coder/Ollama → OpenHands agent → Piper.
  - Latency work waits until Phase 16.
  - Interruption waits until Phase 12.
- **The current code** ([VoiceAI-Agent/app/transcribe.py](VoiceAI-Agent/app/transcribe.py)) finishes each step before the next one starts: record everything → transcribe everything → one non-streaming Gemini call → print.
  - This design can't reach low latency or support interruption, however much you tune it later.
  - Latency and barge-in (interrupting the agent while it speaks) have to be built into the architecture from day 1. They can't be added as a final optimisation phase.
- **Your requirements:**
  1. 100% local.
  2. Very low latency.
  3. Natural conversation.
  4. If you speak while the agent is talking, it stops immediately, listens, understands, then replies correctly.
- **Hardware:** RTX 5050 (8 GB VRAM), Ryzen 5 7500F, 32 GB RAM.
  - Qwen3-Coder-30B does **not** fit in VRAM next to STT + TTS.
  - So the plan uses a "two-brain" design, explained in section 1.

**Target (stop speaking → first audio heard):** 500–700 ms at p50 and under 900 ms at p95.
**Barge-in target (you start speaking → agent audio stops):** under 150 ms.

---

## 1. Key architecture changes from your plan

| # | Your plan | Improved plan | Why |
|---|---|---|---|
| 1 | Sequential: STT → LLM → Agent → TTS | **Fully streaming, overlapping pipeline** (asyncio, one process). Every stage consumes partial output from the previous one. | Stage times overlap, so they no longer add up. |
| 2 | One brain (Qwen3-Coder does everything) | **Two brains**, described below. | A 30B model can't answer in under 1 s on 8 GB. A 4B model can't do serious coding. |
| 3 | faster-whisper (batch) | **Streaming STT**: partial transcripts while you speak (candidates in R2). | When you stop, only the last ~300 ms still needs decoding. |
| 4 | Fixed 0.7 s silence = end of turn | **Silero VAD + semantic end-of-turn model** (Pipecat *smart-turn* ONNX, local). A short silence plus "sentence is complete" ends the turn. | Saves 300–500 ms and cuts people off less often. |
| 5 | Ollama, non-streaming | **llama.cpp `llama-server`** (CUDA, streaming, `cache_prompt`, pre-warmed, persistent HTTP connection) | Lower overhead than Ollama, and more control over the KV cache and cancellation. |
| 6 | Piper, runs after the full answer | **Streaming TTS per clause**: Kokoro-82M (quality) or Piper (speed), whichever wins R4. Speech starts after the first ~6–10 tokens. | The first audio no longer waits for the whole answer. |
| 7 | Interruption in Phase 12 | **Full-duplex audio + barge-in from R1**: the mic is always listening, even while the agent speaks. | This is your main requirement. |
| 8 | AirPods (Bluetooth) mic, hardcoded in [microphone.py:11](VoiceAI-Agent/voice/microphone.py#L11) | **Wired headset** (or USB mic + headphones). The device is chosen in config. | Bluetooth headset mode adds 100–200 ms and lowers audio quality. Headphones stop echo from triggering false interruptions. |
| 9 | Latency measured in Phase 16 | **Latency measurement harness in R0**, before anything else | You can't optimise what you don't measure. |

The two brains:

- **Fast "Voice Brain":** Qwen3-4B-Instruct-2507, Q4_K_M, on the GPU. It talks with you, understands intent, gives short replies and confirmations, and speaks progress updates. It runs in non-thinking mode.
- **Slow "Worker Agent":** a coding model on CPU+GPU offload, e.g. Qwen3-Coder-30B-A3B using llama.cpp `--n-cpu-moe`. It runs in the background through OpenHands and never blocks the voice loop.

### Target architecture

```
 Mic (wired, 16 kHz, 20 ms frames) ──► Silero VAD (always on, also while agent speaks)
        │                                   │ speech start ──► BARGE-IN: stop playback, cancel LLM + TTS
        ▼                                   ▼
 Streaming STT (partials) ──► Turn detector (VAD silence + smart-turn)
                                            │ end of turn (+ speculative start on pause)
                                            ▼
                  Voice Brain (llama-server, Qwen3-4B, streaming, cached system prompt)
                     │ tokens                       │ tool intent
                     ▼                              ▼
          Clause chunker ──► Streaming TTS ──► Speaker (small buffer, can be flushed instantly)
                                                    ▲
   Worker Agent (OpenHands + Qwen3-Coder-30B-A3B, background) ── progress events ──┘
                (permission questions are sent back to the Voice Brain)
```

### VRAM budget (8 GB)

| Component | Approx. VRAM |
|---|---|
| STT (Parakeet 0.6B fp16 / Whisper small) | 1.0–1.3 GB |
| Voice Brain Qwen3-4B Q4_K_M + 8k KV cache | 3.0–3.5 GB |
| TTS Kokoro-82M | 0.3–0.5 GB |
| Headroom / Worker attention layers | ~2.5 GB |

The Worker's MoE experts sit in system RAM (32 GB). **Risk:** the Worker and the Voice Brain compete for the GPU. R8 has to verify that the voice path keeps priority, for example by pausing the Worker while the Voice Brain generates.

---

## 2. Latency budget (stop speaking → first audio)

| Stage | Budget | How |
|---|---|---|
| End-of-turn decision | 200–300 ms | VAD silence of ~200 ms, confirmed by smart-turn. Up to 700 ms if the sentence is incomplete. |
| Final STT | 30–80 ms | Streaming, so only the tail of the audio still needs decoding. |
| LLM time to first token | 50–150 ms | 4B on GPU, system prompt KV cached, server warm. |
| First clause (~6–10 tokens) | 60–120 ms | ~80–120 tok/s |
| TTS first audio chunk | 50–120 ms | Streaming synthesis of the first clause. |
| Output buffer | 20–40 ms | Small WASAPI buffer |
| **Total** | **~450–800 ms** | |

**Extra tricks, tested in R7:**

- **Speculative generation:** start the LLM when the pause begins. Cancel it if you keep talking.
- **Instant pre-recorded acknowledgements:** e.g. "Okay, checking that…". They play in under 200 ms when a task goes to the Worker.

---

## 3. Barge-in (interruption) design

This is the core behaviour you asked for.

1. The mic and VAD run all the time, including while the agent is speaking.
2. **Echo control:**
   - Headphones are the primary fix.
   - For speakers, add WebRTC AEC (e.g. `livekit` APM or `speexdsp`). Treat this as an R&D item.
3. **Interrupt trigger:** VAD reports speech for at least ~150–200 ms while the agent is talking. Optionally, the first STT partial must be real words, so coughs and "mm-hmm" are ignored. Then, all in under 150 ms:
   - flush the speaker buffer (stop the audio at once),
   - cancel the LLM stream (abort the HTTP request),
   - cancel queued TTS jobs.
4. **Context repair:** the conversation history stores only what the agent **actually said** before the interruption (tracked against playback position). It never stores the full planned reply. Then the Voice Brain knows exactly what you heard.
5. The agent listens to the full new turn, then answers with that context. For example, if you interrupt with "no, the other file", it knows which sentence you cut off.
6. **Background tasks:** if the Worker Agent is running a task, an interruption stops only the *speech*, not the task. The exception is an explicit command such as "stop the task" or "cancel", which the Voice Brain turns into a cancel signal for the Worker.

---

## 4. R&D phases (replaces Phases 0–17; latency and barge-in come first)

Each phase has a **measurable exit criterion**.

- **R0: Measurement harness (1–2 days)**
  - Add a timestamped event log (JSONL) for each stage: `speech_end`, `stt_final`, `llm_first_token`, `tts_first_audio`, `playback_start`, `bargein_detected`, `playback_stopped`.
  - Add a script that reports p50/p95 values.
  - Fix the existing bugs:
    - `settings.ollama_url` in [app/main.py:7](VoiceAI-Agent/app/main.py#L7).
    - The hardcoded device in [voice/microphone.py:11](VoiceAI-Agent/voice/microphone.py#L11).
    - The over-strict segment filter in [voice/transcriber.py:58](VoiceAI-Agent/voice/transcriber.py#L58).
    - The catch-all CPU fallback in [voice/transcriber.py:32](VoiceAI-Agent/voice/transcriber.py#L32).
  - **Mic selection (any mic works, no hardcoded name):**
    - Default = the Windows default input device (`sd.default.device`).
    - Optional override is `VOICEAI_MIC_DEVICE` in `.env`, as a name or index. Leave it empty to use the default.
    - Add a `--list-mics` command.
    - When the same device appears under several host APIs, prefer WASAPI.
    - If the chosen mic is missing, print a clear warning and fall back to the default instead of crashing.
  - *Exit:* the current pipeline has a baseline measurement.
- **R1: Full-duplex audio + streaming VAD**
  - Mic input and speaker output run in parallel: `sounddevice`, WASAPI, 20 ms frames.
  - Silero VAD (ONNX) runs on every frame.
  - The ambient-noise measurement is done once at startup, not once per turn.
  - Models are loaded once, in a long-running process.
  - *Exit:* speaking while a WAV plays stops that audio in under 150 ms.
- **R2: STT bake-off**
  - Candidates:
    - faster-whisper `small.en` / `distil-large-v3`, re-decoding the growing audio buffer (whisper_streaming LocalAgreement),
    - NVIDIA Parakeet-TDT-0.6B via `onnx-asr`/sherpa-onnx,
    - Moonshine.
  - Measure accuracy on 30 of your own recorded coding commands (word error rate, WER), plus final-transcript latency and VRAM.
  - *Exit:* final transcript in under 100 ms after speech ends, with WER under 8%.
- **R3: Voice Brain LLM bake-off**
  - Run `llama-server` with CUDA.
  - Candidates: Qwen3-4B-Instruct-2507, Gemma-3-4B, Llama-3.2-3B, Qwen3-1.7B (all Q4_K_M).
  - Measure time to first token with a cached system prompt, tokens per second, and the quality of intent/tool-routing on 30 test prompts.
  - Replace [agent/ollama_client.py](VoiceAI-Agent/agent/ollama_client.py) with a streaming OpenAI-compatible client. Keep the current `LLM` interface idea so providers stay swappable.
  - *Exit:* time to first token under 150 ms, and 90% or more correct routing.
- **R4: TTS bake-off**
  - Candidates: Kokoro-82M (`kokoro-onnx`), Piper (medium voice), and optionally Kyutai TTS or Chatterbox if VRAM allows.
  - Measure time to first audio for a 6-word clause, real-time factor (RTF), and how natural it sounds (your own score from 1–5).
  - *Exit:* first audio under 120 ms, quality of 4 or more.
- **R5: Streaming pipeline integration**
  - Build an asyncio event bus: STT partials → turn detector → LLM token stream → clause chunker → TTS → playback queue.
  - Every task can be cancelled.
  - Treat Pipecat's local pipeline as a *reference baseline* only. Build a lean custom pipeline for full control.
  - *Exit:* p50 under 800 ms end to end over 50 turns.
- **R6: Turn detection + barge-in + context repair**
  - Add smart-turn and the interrupt rules from section 3.
  - Track the playback position so history is truncated to what was actually spoken.
  - *Exit:* 20 scripted interruptions:
    - 100% stop in under 150 ms,
    - 0 false interrupts from the agent's own voice,
    - follow-up answers are correct.
- **R7: Latency tricks**
  - Speculative LLM start on a pause.
  - Pre-synthesised filler/acknowledgement cache.
  - Keep-warm pings.
  - Tune the silence threshold.
  - *Exit:* p50 under 600 ms.
- **R8: Worker Agent (your original Phases 4–8)**
  - OpenHands SDK + Qwen3-Coder-30B-A3B (MoE offload), running as a background task.
  - Tools, permissions and progress events keep your existing Level 1/2/3 design.
  - Level-3 confirmations are *asked by voice* through the Voice Brain.
  - *Exit:* the voice latency p95 is unchanged while the Worker runs.
- **R9 onwards:** memory, GUI, advanced tools. Same as your Phases 13–15.

The **stated principle stays the same:** replace providers, not the application. Every component (VAD, STT, turn detection, LLM, TTS, AEC) sits behind a small interface, so each bake-off winner can be swapped in.

---

## 5. Files (when implementation starts)

**Keep/refactor:**

- [voice/microphone.py](VoiceAI-Agent/voice/microphone.py): becomes a full-duplex `AudioIO`.
- [voice/transcriber.py](VoiceAI-Agent/voice/transcriber.py): becomes a streaming `STT` interface.
- [agent/ollama_client.py](VoiceAI-Agent/agent/ollama_client.py): becomes a streaming `llama_server_client.py`.
- [config/settings.py](VoiceAI-Agent/config/settings.py): add the device, model and threshold settings.

**New:**

- `voice/vad.py`, `voice/turn_detector.py`, `voice/tts.py`, `voice/playback.py`
- `pipeline/bus.py`, `pipeline/orchestrator.py`, `pipeline/bargein.py`
- `metrics/latency.py`, `bench/` (the bake-off scripts)

**Docs:** save this improved plan into the repo as `plan_v2_low_latency.md` next to [plan.md](plan.md).

## 6. Verification

- `bench/` scripts give p50/p95 per stage for each R&D phase, and the results are compared with the exit criteria above.
- A scripted conversation test runs 50 turns and 20 barge-ins using pre-recorded WAVs played into a virtual audio cable (VB-Cable), so results are repeatable.
- Manual test: talk naturally for 5 minutes with headphones, interrupting several times. The agent stops at once, and its follow-up answers refer correctly to what it had said.
- `pytest` smoke tests ([tests/test_smoke.py](VoiceAI-Agent/tests/test_smoke.py)) keep passing.

## 7. Honest trade-offs

- **Fully local + very low latency means the model you *talk to* is small (3–4B).** It is fast and good at conversation and routing, but it is not a strong coder. Serious coding happens in the background Worker, which is slower. That is fine because you hear progress updates while it works.
- **On speakers without headphones, barge-in reliability depends on echo cancellation.** That is harder on Windows, so headphones are strongly recommended.
- Latency under 500 ms is possible but not guaranteed on this GPU. 500–700 ms is the realistic "feels instant" range.
