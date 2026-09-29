# Plan v3 — Roadmap to a Low-Latency, Use-Anywhere Local Voice Agent

> **For Claude Code:** this is the working plan. Read it together with
> [implementation.md](implementation.md) (what is already built, with measured results) and
> [plan_v2_low_latency.md](plan_v2_low_latency.md) (architecture background).
> Work **one step at a time**. Every step ends with a **Checkpoint report** (template at the
> bottom). The user sends that report for review before the next step starts.
> Do not skip ahead, and do not start a step whose previous "Done when" check has not passed.

---

## 0. Goal

Build a voice agent that is:

1. **100% local and free**: open-source models only, no paid APIs, runs offline after setup.
2. **Extremely low latency**: it feels instant, and you can interrupt it at any moment.
3. **Usable everywhere**: the same engine serves the local mic, a web page, a chatbot, apps
   and, later, phones.
4. **General purpose**: it talks naturally, and tools/skills add abilities (coding is only
   one skill).

### Targets

| Metric | Target |
|---|---|
| Stop speaking → first audio heard | **p50 < 600 ms**, p95 < 900 ms |
| Start speaking while the agent talks → agent audio stops | **< 150 ms** |
| False interruptions from the agent's own voice (headphones) | 0 in 20 tries |
| Final transcript ready after speech ends (excluding VAD wait) | < 100 ms |
| LLM time to first token (warm) | < 150 ms |
| TTS first audio for the first clause | < 120 ms |
| Word error rate on the user's own 30 commands (wired mic) | < 8% |

### Hardware (fixed)

RTX 5050 8 GB (Blackwell, sm_120) · Ryzen 5 7500F · 32 GB RAM · Windows 11, running natively (no WSL).

---

## 1. Where we are now (baseline)

Already built and measured (see `implementation.md` → Results):

| Part | Status |
|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M via Ollama | ✅ 100% GPU, 120 tok/s, 3.2 GB VRAM |
| Parakeet TDT 0.6B v2 int8 (CPU) | ✅ 65–130 ms per utterance |
| Kokoro-82M, PyTorch cu128 (GPU) | ✅ 61 ms first audio, 663 MB VRAM |
| Silero VAD v6 streaming (`voice/vad.py`) | ✅ |
| Full-duplex mic (`voice/audio_io.py`) + instant-stop player (`voice/playback.py`) | ✅ stop in ~45–60 ms |
| Streaming STT with partials (`voice/stt.py`) + live test (`bench/live_stt.py`) | ✅ works, accuracy issue caused by the Bluetooth mic |
| Smart Turn v3.2 ONNX | ✅ downloaded, **not connected yet** |
| Latency log + report (`metrics/latency.py`, `bench/latency_report.py`) | ✅ |

### Problems found in the code review (all fixed by this plan)

| # | Problem | Where | Cost | Fixed in |
|---|---|---|---|---|
| L1 | Final text is confirmed only after **700 ms** of silence, and the LLM waits for it | `bench/live_stt.py` `JOIN_WINDOW_MS = 700` | ~700 ms | Step 1.6 |
| L2 | Each partial re-decodes the **whole** utterance | `StreamingSTT.feed()` | grows with length | Step 1.4 |
| L3 | `finish()` waits for the running partial, then decodes everything again | `StreamingSTT.finish()` | +65–130 ms | Step 1.4 |
| L4 | The LLM doesn't stream, opens a new connection per call, and keeps no history | `agent/ollama_client.py` | ~1 s | Step 1.2 |
| L5 | TTS is not in the pipeline | none | none | Step 1.3 |
| L6 | The entry point is sequential and still uses Gemini | `app/transcribe.py` | whole design | Step 1.5 |
| L7 | The latency log does blocking file I/O on the hot path | `metrics/latency.py` `mark()` | jitter | Step 1.1 |
| L8 | GIL risk: audio callbacks, Torch, ONNX and asyncio all share one process | `playback.py`, `audio_io.py` | crackle under load | measured in Step 1.5 |

**Estimated path today:** 200 ms VAD + 700 ms join + ~200 ms STT + ~1 s full LLM answer + TTS ≈ **2+ s**.
**Target after Phase 1:** ≈ 450–600 ms.

---

## 1a. Amendments agreed on 2026-09-29 (after comparing with the code)

1. **Disk:** C: had only 10 GB free (D: has 3.2 TB), so models move to D: in **Phase 0** (Step 0.4), after the user's OK.
2. **Branch:** `feature/v3-roadmap` is created from `feature/low-latency-voice`, so the 4 earlier commits are kept.
3. **Drop `kokoro-onnx`:** it fails on DirectML and is 5–10× slower on CPU than Kokoro in PyTorch. In Step 1.3, switch `bench/bargein_demo.py` and `bench/phase0_smoke.py` to `KokoroTTS`, then remove the package and `models/kokoro/`.
4. **Step 1.4, long speech (over 8 s):** only build it if measurements show it's needed. Measure Parakeet CPU time on 20 s of audio first.
5. **Step 1.6, first check:** confirm that Ollama really stops a cancelled generation, and that the new request doesn't queue behind it (`OLLAMA_NUM_PARALLEL`).
6. **Step 2.1:** Smart Turn needs 80-bin Whisper log-mel features. Reuse `transformers.WhisperFeatureExtractor` (already installed).
7. **New tool:** `bench/record_commands.py` records the user's 30 commands with reference text, for WER (Step 1.4).
8. **Hardware:** several checks assume a **wired mic**. The user currently has AirPods (Bluetooth). Checkpoint reports flag this.
9. **Docs:** this file is the plan. `implementation.md` stays as the log of results.
10. **VAD model:** the Silero VAD ONNX file is copied into `models/`, so faster-whisper can become optional (Step 0.1).

---

## 2. Working rules (apply to every step)

1. **One step at a time.** Finish, run the checks, write the Checkpoint report, then stop and wait.
2. **Git:** work on branch `feature/v3-roadmap`. Commit after each step with a clear message.
   Never push without asking.
3. **Every step:** code → `pytest` → `ruff check` → a short manual run → record numbers.
4. **Interfaces first.** Every component (VAD, STT, TurnDetector, LLM, TTS, AudioSource,
   AudioSink, Tool) sits behind a small `Protocol`. Swapping one only needs a `.env` change.
5. **Local only.** No cloud calls in the pipeline. Gemini is removed in Step 0.2.
6. **Measure everything that affects speed** in `logs/latency.jsonl`. Add results to the
   Results table in `implementation.md`.
7. **Everything must be cancellable** (LLM stream, TTS jobs, playback). Barge-in depends on it.
8. **Models load once** in one long-running process. No loading per turn.
9. **Tests must not depend on the user's `.env`**, and must not need a mic. Use WAV fixtures.
10. **Copying code:** code from Apache-2.0/MIT projects (e.g. `huggingface/speech-to-speech`)
    may be copied or adapted. Keep a header comment naming the source and license, and list it
    in `THIRD_PARTY_NOTICES.md`. Do **not** copy code from `vtmate` (separate commercial and
    non-commercial licenses).

---

## 3. Target architecture

```
                         ┌──────────────── Clients ─────────────────┐
                         │ Local CLI (mic/speaker) · Web page/widget │
                         │ Chatbot backend · Apps · (later) phone     │
                         └───────────────┬──────────────────────────┘
                                         │ AudioSource / AudioSink
                                         │ (local device, WebSocket, later WebRTC/SIP)
                                         ▼
 ┌────────────────────────────── Session (one per conversation) ──────────────────────────────┐
 │  VAD (Silero, always on) ──► Streaming STT (partials) ──► Turn detector (VAD + Smart Turn)  │
 │        │ speech while SPEAKING ► BARGE-IN: stop sink, cancel LLM + TTS, repair memory       │
 │        ▼                                                 │ soft end ► SPECULATIVE start     │
 │                                                          ▼ commit / reopen                  │
 │  Voice Brain LLM (Ollama, streaming, cached system prompt, tools)                           │
 │        │ tokens                          │ tool calls ──► Tools / Skills / Worker (async)   │
 │        ▼                                                                                    │
 │  Speakable-text normaliser ► Clause chunker ► Streaming TTS (Kokoro) ► AudioSink            │
 │  Memory (turns, truncated to what was actually heard) · Persona (prompt, voice, language)   │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
                   Shared models (loaded once): Parakeet · Kokoro · Smart Turn · Ollama
```

**Session state machine:** `LISTENING → (soft end) SPECULATING → (commit) SPEAKING → LISTENING`.
`SPECULATING → LISTENING` on reopen (the user continued). `SPEAKING → LISTENING` on barge-in.

---

## Phase 0 — Cleanup (small, do first)

### Step 0.1 Dependencies
- Fix the `onnxruntime` / `onnxruntime-directml` conflict (`faster-whisper` pulls in plain
  `onnxruntime`). Document the exact install order in the README, or move faster-whisper into an
  optional extra `[whisper]`.
- Document the Torch cu128 install command in the README.

### Step 0.2 Remove Gemini
- Delete `agent/gemini_client.py` and the `google-genai` dependency.
- Remove the Gemini keys from `settings.py` and `.env.example`, and update the tests.
- Until Step 1.5 replaces it, `app/transcribe.py` uses Ollama instead.

### Step 0.3 Settings
Add these keys to `config/settings.py` and `.env.example`, with defaults:

```
VOICEAI_LLM_URL=http://127.0.0.1:11434
VOICEAI_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M
VOICEAI_LLM_NUM_CTX=4096
VOICEAI_LLM_MAX_TOKENS=200
VOICEAI_TTS_ENGINE=kokoro
VOICEAI_TTS_VOICE=af_heart
VOICEAI_TTS_DEVICE=cuda
VOICEAI_VAD_THRESHOLD=0.5
VOICEAI_VAD_START_MS=64
VOICEAI_VAD_END_MS=200
VOICEAI_PRE_ROLL_MS=500
VOICEAI_JOIN_WINDOW_MS=700
VOICEAI_BARGEIN_MIN_MS=160
VOICEAI_SMART_TURN_THRESHOLD=0.5
VOICEAI_SYSTEM_PROMPT_FILE=config/system_prompt.md
```

### Step 0.4 README + disk
- Rewrite the README to match reality: local stack, Parakeet, Kokoro, Ollama, streaming.
- Report free space on C:. Do not delete anything without the user's OK.

**Done when:** a clean install works by following the README, `pytest` and `ruff` pass, and
nothing imports Gemini.

---

## Phase 1 — Latency core

### Step 1.1 Non-blocking latency log (fixes L7)
- `LatencyLog.mark()` puts the record on a `queue.SimpleQueue`. A daemon thread writes the
  JSONL. Call `flush()` on exit.
- Add events: `stt_partial`, `turn_soft_end`, `turn_commit`, `turn_reopen`, `llm_cancel`,
  `tts_cancel`, `speculative_start`.

**Done when:** `mark()` takes < 50 µs (micro-benchmark), and the report still works.

### Step 1.2 Streaming LLM client (fixes L4) → `agent/llm.py`
- An `LLM` protocol: `stream(messages, tools=None) -> Iterator[Delta]` plus `cancel()`.
- `OllamaLLM`: `/api/chat`, `stream: true`, `think: false`, `keep_alive: -1`.
  - One persistent `httpx.Client`.
  - `temperature ≈ 0.6`, `num_predict` taken from settings.
- Pre-warm at startup (a one-token request with the real system prompt, so the prefix is cached).
- Cancel = close the response stream. Check that Ollama stops generating (`ollama ps` / GPU usage).
- `agent/prompt.py`: a **fixed** system prompt (from `config/system_prompt.md`) + the last N
  turns + the new message. The system prompt must be byte-identical every turn so the
  prefix cache works.
  - The prompt demands short, speakable answers (1–3 sentences, no markdown, no lists unless asked).
- `agent/memory.py`: turn history with `add_user`, `add_assistant`, and
  `truncate_last_assistant(spoken_text)`.
- `bench/llm_bench.py`: TTFT and tok/s over 30 prompts, warm.

**Done when:** warm TTFT p50 < 150 ms, > 60 tok/s, and cancel stops generation within 100 ms.

### Step 1.3 Streaming TTS + clause chunker (fixes L5)
- `voice/tts.py`: a `TTS` protocol with `synthesize_stream(text) -> Iterator[np.ndarray]` and
  `cancel()`.
  - `KokoroTTS` (PyTorch cu128, voice from settings) outputs 24 kHz.
  - The sink resamples if the device needs 48 kHz.
- `pipeline/chunker.py`: turns the token stream into clauses.
  - The **first clause is short** (cut at the first `, . ! ? ; :` after ≥ 2–3 words, or at
    ~6 words), so first audio comes quickly.
  - Later clauses are ≥ 4 words and cut at punctuation.
  - Never cut inside numbers such as "3.5" or abbreviations such as "Dr." or "e.g.".
- A TTS worker thread takes clauses from a queue and pushes audio to the sink.
  It can be cancelled between clauses and mid-clause.
- Record which clause each audio chunk belongs to (needed for context repair in Step 2.2).
- `bench/tts_bench.py`: first-audio time and real-time factor, plus the user's 1–5
  naturalness score.

**Done when:** first audio for the first clause is < 120 ms, there are no audible gaps between
clauses, and cancel stops TTS within one chunk.

### Step 1.4 STT tail decoding + partial reuse (fixes L2, L3)
- **Partial reuse:** at speech end, if no new audio arrived after the last partial started,
  use that partial as the final. Otherwise decode.
- **Don't wait for stale partials:** `finish()` marks the running partial as stale and decodes at
  once. The stale result is thrown away.
  - Parakeet runs on the CPU, so two decodes can overlap. Make sure it stays thread-safe:
    use a lock or two sessions.
- **Bounded cost for long speech:** once the utterance is longer than ~8 s, keep a *confirmed
  prefix*. Text that is stable across 2 partials (LocalAgreement) is kept, and only the audio
  after the last confirmed word boundary is re-decoded. Parakeet gives word timestamps; use them.
- Extend `tests/test_phase3.py` with a 20-second WAV fixture.

**Done when:** final text is < 100 ms after speech end for 3 s **and** 20 s utterances, and WER
doesn't get worse on the 30 recorded commands (recorded with a wired mic).

### Step 1.5 Orchestrator, end-to-end local loop (fixes L6, measures L8)
- `pipeline/events.py`: typed events (`SpeechStart`, `SpeechEnd`, `PartialText`, `FinalText`,
  `LLMDelta`, `Clause`, `AudioChunk`, `BargeIn`, `TurnCommit`, `TurnReopen`).
- `pipeline/orchestrator.py`: an asyncio state machine (Section 3).
  - Blocking model calls run through `run_in_executor` / worker threads.
  - Audio callbacks never touch asyncio directly; they only use queues.
- `app/main.py`: `voiceai run` starts a long-running conversation loop.
  - It loads models once, pre-warms, then prints "ready".
  - `app/transcribe.py` is deleted, and its useful parts move into the pipeline.
- In this step a turn commits after the join window (700 ms) exactly as now. Speculation comes
  in Step 1.6.
- **GIL check (L8):** run 10 minutes of conversation and log output underruns (`status` in the
  player callback). If there are underruns: first raise the output buffer to 40–60 ms; if that
  isn't enough, move the TTS into a separate process that sends audio over a queue/pipe.

**Done when:** a multi-turn spoken conversation works, with no crackle in 10 minutes. Record the
baseline p50 for speech end → first audio.

### Step 1.6 Speculative start + reopen (fixes L1, the biggest gain)
Pattern adapted from `huggingface/speech-to-speech` (Apache-2.0). Credit the source.

1. At **soft end** (VAD silence ≥ 200 ms):
   - finalize the STT (Step 1.4),
   - start the LLM → chunker → TTS right away,
   - **hold** the audio in a buffer instead of playing it.
2. **Commit** the turn when either:
   - Smart Turn says the turn is complete (only once Step 2.1 exists; before that, skip this), **or**
   - the silence reaches the join window (default 700 ms, then tuned lower).
   On commit: release the held audio to the sink at once, and add the user turn to memory.
3. **Reopen** if speech starts again before commit:
   - cancel the LLM and TTS, and discard the held audio,
   - `stt.resume(gap)`,
   - go back to LISTENING,
   - log `turn_reopen`.
4. Never play speculative audio before commit. The user must never hear an answer to half a sentence.

**Done when:**
- p50 speech end → first audio **< 800 ms** over 50 scripted turns (VB-Cable, pre-recorded WAVs),
- and with Smart Turn active (after Step 2.1): **< 600 ms**,
- zero answers to half sentences in 20 scripted "pause mid-sentence" tests.

---

## Phase 2 — Natural conversation

### Step 2.1 Smart Turn → `voice/turn_detector.py`
- Run Smart Turn v3.2 (CPU ONNX) on the last ≤ 8 s of the turn audio at each soft end.
- If complete (probability ≥ threshold): commit right away.
- If incomplete: wait up to ~1.5–2 s of silence before committing (a cap, so it never hangs).
- Keep "silence-only" as the fallback engine.

**Done when:** in 20 scripted tests of complete sentences, the median commit is ≤ 300 ms after
speech end. In 20 tests with "umm… pauses", 0 are cut off.

### Step 2.2 Barge-in + context repair → `pipeline/bargein.py`
- While SPEAKING, VAD speech ≥ `BARGEIN_MIN_MS` (160 ms) → **stop the sink, cancel the LLM,
  cancel the TTS**. All three together must take < 150 ms.
- **False-trigger filter** (optional, switchable): confirm with the first STT partial. If it's
  empty or only a backchannel word ("mm", "hmm", "uh-huh", "okay"), resume or ignore instead of
  stopping.
  - Keep a fast path: a hard stop right away, and let the filter only decide whether to *resume*.
- **Context repair:** map `played_samples` → clause → words, and call
  `memory.truncate_last_assistant(heard_text)`. The LLM only remembers what the user actually heard.
- Start the new user turn with pre-roll, so the interrupting words aren't lost.

**Done when:** 20 scripted interruptions show 100% stop in < 150 ms, 0 false stops from the
agent's own voice (headphones), and follow-up answers that correctly refer to the cut-off sentence.

### Step 2.3 Speakable-text normaliser → `pipeline/normalize.py`
- Runs on each clause before TTS:
  - numbers, times ("10:30" → "ten thirty"), dates, currency ("Rs. 500"), percentages,
  - abbreviations ("Dr.", "e.g."), URLs and emails (say them briefly or skip them),
  - strip markdown (`*`, `#`, backticks, list markers) and emojis.
- Unit tests with ≥ 30 cases.

**Done when:** all tests pass, and a listening test of 10 tricky sentences sounds right.

### Step 2.4 Instant acknowledgements (filler cache)
- At startup, pre-synthesise a few short phrases ("Okay.", "One second.", "Let me check that.").
- Use them only when a tool call or Worker task will take > 700 ms. Never on normal replies.

**Done when:** a tool-type request plays an acknowledgement in < 200 ms.

### Step 2.5 (Optional) Noise suppression / echo cancellation
- Local mode: optional RNNoise or DeepFilterNet on the mic input (measure the added latency;
  keep it under 10 ms).
- Speaker mode without headphones: test WebRTC AEC (e.g. `livekit` APM or speexdsp). This is
  R&D; headphones stay the recommended setup.
- Browser clients get AEC for free (`echoCancellation: true`, Phase 3).

---

## Phase 3 — Usable everywhere

### Step 3.1 AudioSource / AudioSink interfaces → `voice/io.py`
- `AudioSource`: an async iterator of 16 kHz mono float32 blocks with timestamps.
- `AudioSink`: `write(chunk)`, `stop()`, `played_samples`, events (`playback_start` / `playback_stopped`).
- `LocalMicSource` (wraps `MicStream`), `LocalSpeakerSink` (wraps `Player`).
- Resampling lives in the source/sink, never in the pipeline.

### Step 3.2 Session + SessionManager → `pipeline/session.py`
- A `Session` owns its per-conversation state: VAD state, StreamingSTT, turn detector,
  memory, persona (system prompt, voice, language) and orchestrator.
- **Shared** models (Parakeet, Kokoro, Smart Turn, Ollama client) are loaded once and used through
  locks or queues.
- `SessionManager`: `max_sessions` setting (default 2), a queue or rejection beyond that, cleanup
  on disconnect.
- `voiceai run` becomes: one Session with a local source/sink.

**Done when:** the local mode works exactly as before through Session. A test runs two sessions
fed from WAV files at the same time without errors.

### Step 3.3 Realtime WebSocket server → `server/realtime.py`
- Implement the **core OpenAI Realtime event set**, adapted from `huggingface/speech-to-speech`
  (credit the source):
  - Client → server: `session.update`, `input_audio_buffer.append` (base64 PCM16 at 24 kHz or
    16 kHz), `input_audio_buffer.commit`, `conversation.item.create`, `response.create`,
    `response.cancel`.
  - Server → client: `session.created/updated`, `input_audio_buffer.speech_started/stopped`,
    `conversation.item.input_audio_transcription.delta/completed`, `response.created`,
    `response.output_audio.delta`, `response.output_audio_transcript.delta`, `response.done`,
    `error`.
- `session.update` sets the persona: instructions, voice, language and turn-detection settings.
- A barge-in in a network session sends `speech_started` plus the cancel. The client must
  flush its own playback buffer when it receives it (document this).
- Command: `voiceai serve --host 127.0.0.1 --port 8765`, with the endpoint at `ws://…/v1/realtime`.
- **Security:**
  - Bind to localhost by default.
  - API keys from `config/api_keys.txt` (`Authorization: Bearer …` or a query param).
  - Per-key session limit.
  - No keys, no connection when bound to anything other than localhost.

**Done when:**
- the official `openai` Python SDK (`client.realtime.connect`, pointed at our server) completes a
  spoken round trip from a WAV file,
- latency over WebSocket on localhost is within +50 ms of local mode,
- unauthorised connections are rejected.

### Step 3.4 Local client over the server (optional)
- `voiceai talk --url ws://…` is a small mic/speaker client that uses the server, like the
  HF `talk` command. It proves the local and network paths are the same.

---

## Phase 4 — Clients and brain

### Step 4.1 Web client → `web/`
- One static page plus an embeddable widget (`<script>` + a mic button).
  - `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all on.
  - AudioWorklet → PCM16 → WebSocket.
  - Playback through an AudioWorklet with an instantly flushable buffer on `speech_started`.
- Served by `voiceai serve` at `/`.
- Shows live transcripts, the agent text and latency numbers.

**Done when:** a conversation in Chrome on the same PC works with barge-in through laptop
speakers (browser AEC), and p50 is within +100 ms of local mode.

### Step 4.2 Tools / skills → `agent/tools/`
- A `Tool` protocol: name, description, JSON schema, `async run(args)`, and a permission level
  (1 = read, 2 = write, 3 = dangerous). This keeps the original Level 1/2/3 design.
- Ollama native tool calling with Qwen3.
- The Voice Brain decides: answer directly, call a quick tool, or hand a long job to the Worker.
- Starter skills: time/date, calculator, notes/reminders (local JSON), read local files in an
  allowed folder, and optional local web search (SearXNG, self-hosted).
- Level 3 actions ask for **voice confirmation**.
- Tool results can be spoken, and they are summarised by the LLM.

### Step 4.3 Worker Agent (background)
- This is the original Phase 9, as a skill. It uses OpenHands SDK + a coding model
  (`qwen3:14b` already downloaded; Qwen3-Coder-30B-A3B only if disk space allows).
- It runs asynchronously, and progress events are spoken by the Voice Brain.
- While the Voice Brain generates, the Worker pauses its GPU use, so voice keeps priority.

**Done when:** voice latency p95 is unchanged while the Worker runs.

### Step 4.4 Knowledge (RAG) → `agent/knowledge/`
- A local embedding model (e.g. `bge-m3` or `nomic-embed-text` via Ollama) plus a local vector
  store (e.g. LanceDB or SQLite-vec).
- `voiceai ingest <folder>` indexes documents.
- Retrieval is a tool, so it's only used when needed and doesn't slow normal replies.

### Step 4.5 Memory
- Long-term memory (facts the user asked it to remember) in a local store. It's added to the
  prompt *after* the cached system prompt, so prefix caching still works.

---

## Phase 5 — Reach and languages

### Step 5.1 Multilingual (Urdu/Hindi and others)
- A second STT engine: faster-whisper `large-v3-turbo` (multilingual) on CUDA, chosen per
  session (`language` in `session.update`).
- Kokoro Hindi voices for spoken Urdu/Hindi. Test the LLM's answer quality (Qwen3-4B vs a Gemma
  4B-class model) and pick per language.
- Language auto-detect is optional.

### Step 5.2 WebRTC transport
- `aiortc` (or similar) transport with Opus. Same Session, new AudioSource/AudioSink.
- Needed for good latency over the internet or on mobile networks.

### Step 5.3 Phone (later)
- SIP via a local PBX (e.g. Asterisk) or an open-source SIP bridge, mapped to a Session.

### Step 5.4 Remote access (later)
- Expose the server safely (Cloudflare Tunnel or a VPN such as Tailscale), with auth and rate limits.

---

## 4. File map

| Status | File |
|---|---|
| change | `config/settings.py`, `.env.example`, `README.md`, `pyproject.toml`, `metrics/latency.py`, `voice/stt.py`, `app/main.py` |
| delete | `agent/gemini_client.py`, `app/transcribe.py` (after Step 1.5), `agent/ollama_client.py` (replaced by `agent/llm.py`) |
| new (P1) | `agent/llm.py`, `agent/prompt.py`, `agent/memory.py`, `config/system_prompt.md`, `voice/tts.py`, `pipeline/chunker.py`, `pipeline/events.py`, `pipeline/orchestrator.py`, `bench/llm_bench.py`, `bench/tts_bench.py`, `bench/conversation_test.py` |
| new (P2) | `voice/turn_detector.py`, `pipeline/bargein.py`, `pipeline/normalize.py`, `pipeline/fillers.py` |
| new (P3) | `voice/io.py`, `pipeline/session.py`, `server/realtime.py`, `server/auth.py`, `THIRD_PARTY_NOTICES.md` |
| new (P4) | `web/index.html`, `web/widget.js`, `web/worklets/*.js`, `agent/tools/*`, `agent/knowledge/*` |

---

## 5. Testing approach

- **Unit tests** for the chunker, normaliser, memory truncation, state-machine transitions and
  the event protocol (no audio devices needed).
- **Scripted conversation test** (`bench/conversation_test.py`): plays pre-recorded WAVs
  through VB-Cable, or injects them straight into an `AudioSource`, for repeatable latency and
  barge-in numbers. It covers 50 turns, 20 interruptions and 20 "pause mid-sentence" cases.
- **Manual test** every phase: 5 minutes of natural talk with a **wired headset**, with several
  interruptions.
- Latency reports: `python -m bench.latency_report` gives p50/p95 per stage. Always compare with
  the previous step.

---

## 6. Risks

| Risk | Plan |
|---|---|
| VRAM (8 GB): LLM 3.2 GB + Kokoro 0.7 GB + KV cache, with the Worker/RAG models later | Watch it with `nvidia-smi`. The Worker model uses CPU offload. The embedding model runs on the CPU |
| GIL / audio crackle | Measured in Step 1.5, with a separate TTS process as the fallback |
| Speculation wastes GPU time on reopens | A reopen is cheap (cancel). Log the reopen rate and tune the VAD end time |
| Echo on speakers | Headphones locally. Browser AEC for web. WebRTC AEC is R&D |
| Disk space on C: (10 GB free on 2026-09-29) | Move models to D: in Phase 0 (Step 0.4) using `OLLAMA_MODELS`, `HF_HOME` and `VOICEAI_MODELS_DIR` |
| Small LLM quality | Keep answers short. Hand hard tasks to tools/the Worker. Benchmark alternative 4B models |

---

## 7. Checkpoint report (Claude Code fills this in after every step)

```markdown
## Checkpoint — Step X.Y <name>

**What changed**
- files added / changed / deleted (one line each)

**How to run it**
- exact commands

**Checks**
- pytest: N passed / N failed
- ruff: clean / N issues
- Done-when criteria: each one ✅ / ❌ with the measured number

**Latency (if relevant)**
| Stage | p50 | p95 | previous |
|---|---|---|---|

**Problems / decisions needed**
- anything the user must decide or test by hand (e.g. "please test with a wired mic")

**Next step**
- Step X.Z, not started until approved
```
