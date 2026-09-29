"""LLM TTFT after the GPU has been idle, with and without a warm-up nudge (plan amendment #11).

Usage: python -m bench.gpu_idle_bench [--idle 0 3 10 20] [--nudge-ms 150]

The GPU drops to its idle clock (P8, ~200 MHz) after a few seconds without work, and the first
request afterwards pays for the clock ramp-up. This prints the GPU state and TTFT per idle time.
"""

from __future__ import annotations

import argparse
import subprocess
import time

from agent.llm import OllamaLLM
from agent.memory import Memory
from agent.prompt import PromptBuilder
from config.settings import load_settings


def gpu_state() -> str:
    result = subprocess.run(["nvidia-smi", "--query-gpu=pstate,clocks.sm", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip()


def ttft(llm: OllamaLLM, builder: PromptBuilder) -> float:
    started = time.perf_counter()
    stream = llm.stream(builder.build(Memory(), "Say OK."))
    for delta in stream:
        if delta.content:
            stream.cancel()
            return (time.perf_counter() - started) * 1000
    return float("nan")


def make_nudge():
    import torch

    x = torch.randn(1024, 1024, device="cuda")

    def nudge(ms: float) -> float:
        started = time.perf_counter()
        end = started + ms / 1000
        while time.perf_counter() < end:
            x @ x  # the work itself is the point
        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    return nudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle", type=float, nargs="+", default=[0, 0, 3, 10, 20])
    parser.add_argument("--nudge-ms", type=float, default=150)
    args = parser.parse_args()
    settings = load_settings()
    builder = PromptBuilder.from_file(settings.system_prompt_file, settings.llm_num_ctx, settings.llm_max_tokens)
    llm = OllamaLLM(settings.llm_url, settings.llm_model, settings.llm_num_ctx, settings.llm_max_tokens)
    llm.warm_up(builder.warm_up_messages())

    print("Without nudge:")
    for idle in args.idle:
        time.sleep(idle)
        state = gpu_state()
        print(f"  after {idle:4.0f} s idle: GPU {state:<16} -> TTFT {ttft(llm, builder):5.0f} ms")
    if args.nudge_ms > 0:
        nudge = make_nudge()
        print(f"With a {args.nudge_ms:.0f} ms nudge right before the request:")
        for idle in [i for i in args.idle if i >= 10] or [15]:
            time.sleep(idle)
            cost = nudge(args.nudge_ms)
            print(f"  after {idle:4.0f} s idle: nudge took {cost:4.0f} ms -> TTFT {ttft(llm, builder):5.0f} ms")
    llm.close()


if __name__ == "__main__":
    main()
