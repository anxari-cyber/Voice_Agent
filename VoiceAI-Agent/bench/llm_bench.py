"""Step 1.2 benchmark for the Voice Brain LLM (needs Ollama running).

Usage: python -m bench.llm_bench [--prompts 30]

Measures, with a warm model:
  TTFT      request sent -> first chunk with NON-empty content (empty chunks don't count)
  tok/s     Ollama's eval_count / eval_duration
  cache     prompt_eval_cached_count: proves the fixed system prompt prefix is reused
  think     the raw response has no "thinking" field and no <think> text
  cancel    cancel after 5 tokens; then (a) Ollama's log shows the abort, (b) the GPU goes idle,
            (c) a new request's TTFT is not delayed by the cancelled one
Targets: TTFT p50 < 150 ms, > 60 tok/s, cancel effective within 100 ms.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from agent.llm import OllamaLLM
from agent.memory import Memory
from agent.prompt import PromptBuilder
from config.settings import load_settings

PROMPTS = [
    "What's the capital of France?", "Give me a quick tip to focus better.",
    "How do I undo my last git commit?", "What does HTTP 404 mean?",
    "Tell me a fun fact about octopuses.", "How long should I boil an egg?",
    "What's the difference between a list and a tuple in Python?", "Suggest a name for a cat.",
    "Why is the sky blue?", "How many minutes are in a day?",
    "What is a REST API?", "Can you recommend a good stretch for my back?",
    "What does RAM do in a computer?", "Explain recursion in one sentence.",
    "What's a healthy breakfast idea?", "How do I rename a file in PowerShell?",
    "What time zone is Pakistan in?", "What's the boiling point of water?",
    "How can I make my code run faster?", "What is machine learning?",
    "Give me a motivating quote.", "What's the tallest mountain on Earth?",
    "How do I center a div in CSS?", "What's two hundred times three?",
    "Why do we need sleep?", "What is an operating system?",
    "How do vaccines work?", "Recommend a short book.",
    "What is the speed of light?", "How do I say thank you in Spanish?",
]
LONG_PROMPT = "Count slowly from one to three hundred, writing every number as a word."


def ollama_log() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"


class GpuSampler:
    """Samples GPU utilisation every 50 ms with nvidia-smi's loop mode."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=subprocess.PIPE, text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            if line.strip().isdigit():
                self.samples.append((time.perf_counter(), int(line)))

    def stop(self) -> None:
        self._proc.terminate()

    def between(self, start: float, end: float) -> list[int]:
        return [u for t, u in self.samples if start <= t <= end]


def run_turn(llm: OllamaLLM, messages: list[dict]) -> dict:
    started = time.perf_counter()
    ttft = None
    text = []
    stats = {}
    for delta in llm.stream(messages):
        if delta.content and ttft is None:
            ttft = (time.perf_counter() - started) * 1000
        text.append(delta.content)
        if delta.done:
            stats = delta.stats
    return {"ttft": ttft, "text": "".join(text), "stats": stats,
            "total": (time.perf_counter() - started) * 1000}


def check_think_false(settings, builder: PromptBuilder) -> tuple[bool, str]:
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    body = llm.payload(builder.build(Memory(), "Why is the sky blue? Think carefully."))
    body["stream"] = False
    raw = httpx.post(f"{settings.llm_url}/api/chat", json=body, timeout=60).json()
    message = raw.get("message", {})
    ok = not message.get("thinking") and "<think>" not in message.get("content", "")
    return ok, message.get("content", "")[:80]


def stream_for(llm: OllamaLLM, messages: list[dict], seconds: float | None = None,
               tokens: int | None = None) -> float:
    """Stream until `seconds` pass or `tokens` content chunks arrive; return the stop time."""
    started = time.perf_counter()
    count = 0
    stream = llm.stream(messages)
    for delta in stream:
        count += bool(delta.content)
        if (tokens and count >= tokens) or (seconds and time.perf_counter() - started >= seconds):
            break
    stream.cancel()  # cancel this request only
    return time.perf_counter()


def cancel_test(llm: OllamaLLM, builder: PromptBuilder, gpu: GpuSampler, baseline_ttft: float) -> dict:
    messages = builder.build(Memory(), LONG_PROMPT)

    # Control: the sampler must be able to SEE a busy GPU, otherwise "idle" proves nothing.
    control_start = time.perf_counter()
    control_end = stream_for(llm, messages, seconds=1.5)
    busy = gpu.between(control_start + 0.4, control_end)
    time.sleep(1.5)

    # Test: cancel after 5 tokens, then 1.5 s with NO other request. If Ollama kept generating
    # the rest of the reply (~2 s of work at ~100 tok/s), the GPU would be busy in this window.
    log_path = ollama_log()
    log_size = log_path.stat().st_size if log_path.exists() else 0
    stream = llm.stream(messages)
    count = 0
    for delta in stream:
        count += bool(delta.content)
        if count >= 5:
            break
    cancel_at = time.perf_counter()
    stream.cancel()
    cancel_call_ms = (time.perf_counter() - cancel_at) * 1000
    time.sleep(1.6)
    idle = gpu.between(cancel_at + 0.2, cancel_at + 1.5)

    # A new request right after a cancel: with OLLAMA_NUM_PARALLEL=1 it would queue behind a
    # still-running generation, so a normal TTFT shows the old one really stopped.
    stream_for(llm, messages, tokens=5)
    immediate = run_turn(llm, builder.build(Memory(), "Say OK."))

    log_text = ""
    if log_path.exists():
        with log_path.open("rb") as file:
            file.seek(log_size)
            log_text = file.read().decode("utf-8", "replace")
    cancels = [line.strip() for line in log_text.splitlines() if "cancel task" in line]
    return {
        "cancel_call_ms": cancel_call_ms,
        "gpu_busy": (float(np.mean(busy)) if busy else float("nan"), len(busy)),
        "gpu_after_cancel": (float(np.mean(idle)) if idle else float("nan"),
                             max(idle) if idle else None, len(idle)),
        "log_cancels": len(cancels),
        "log_excerpt": cancels[:1],
        "new_ttft": immediate["ttft"],
        "baseline_ttft": baseline_ttft,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=int, default=30)
    args = parser.parse_args()
    settings = load_settings()
    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)

    started = time.perf_counter()
    llm.warm_up(builder.warm_up_messages())
    print(f"Warm-up (load + prefix cache): {(time.perf_counter() - started) * 1000:.0f} ms")

    results = []
    memory = Memory()
    for prompt in PROMPTS[: args.prompts]:
        messages = builder.build(memory, prompt)
        result = run_turn(llm, messages)
        estimated = builder.estimator.messages_tokens(messages)
        result["estimate_ratio"] = estimated / max(1, result["stats"].get("prompt_eval_count", 1))
        builder.calibrate(messages, result["stats"])
        results.append(result)
        memory.add_user(prompt)
        memory.add_assistant(result["text"])

    ttfts = np.array([r["ttft"] for r in results if r["ttft"] is not None])
    speeds = np.array([r["stats"]["eval_count"] / (r["stats"]["eval_duration"] / 1e9)
                       for r in results if r["stats"].get("eval_duration")])
    cached = [r["stats"].get("prompt_eval_cached_count", 0) for r in results]
    prompt_tokens = [r["stats"].get("prompt_eval_count", 0) for r in results]
    ratios = np.array([r["estimate_ratio"] for r in results])
    print(f"\nTTFT over {len(ttfts)} warm turns (history growing): p50 {np.percentile(ttfts, 50):.0f} ms, "
          f"p95 {np.percentile(ttfts, 95):.0f} ms, max {ttfts.max():.0f} ms   (target p50 < 150)")
    print(f"Generation speed: p50 {np.percentile(speeds, 50):.0f} tok/s, min {speeds.min():.0f}   (target > 60)")
    print(f"Prompt cache: {sum(c > 0 for c in cached)}/{len(cached)} turns reused a cached prefix; "
          f"last turn {cached[-1]}/{prompt_tokens[-1]} prompt tokens cached")
    print(f"Token estimate / real prompt_eval_count: first {ratios[0]:.2f}, last {ratios[-1]:.2f}, "
          f"min {ratios.min():.2f} (>= 1.0 = never under-counts); "
          f"learned {builder.estimator.chars_per_token:.2f} chars/token after {builder.estimator.samples} samples")
    print(f"Sample answer: {results[3]['text'][:120]!r}")

    ok, sample = check_think_false(settings, builder)
    print(f"\nthink:false -> no thinking field / <think> text: {'OK' if ok else 'FAILED'} ({sample!r})")

    gpu = GpuSampler()
    time.sleep(0.5)
    cancel = cancel_test(llm, builder, gpu, float(np.percentile(ttfts, 50)))
    gpu.stop()
    print("\nCancel after 5 tokens:")
    busy_mean, busy_n = cancel["gpu_busy"]
    idle_mean, idle_max, idle_n = cancel["gpu_after_cancel"]
    print(f"  cancel() call returned in {cancel['cancel_call_ms']:.2f} ms")
    print(f"  GPU while generating (control): mean {busy_mean:.0f}% over {busy_n} samples")
    print(f"  GPU 0.2-1.5 s after cancel:     mean {idle_mean:.0f}%, max {idle_max}% over {idle_n} samples")
    print(f"  Ollama log 'cancel task' lines: {cancel['log_cancels']}  {cancel['log_excerpt']}")
    print(f"  request right after a cancel: TTFT {cancel['new_ttft']:.0f} ms "
          f"vs normal p50 {cancel['baseline_ttft']:.0f} ms (queued behind it would be ~1-2 s)")
    llm.close()


if __name__ == "__main__":
    main()
