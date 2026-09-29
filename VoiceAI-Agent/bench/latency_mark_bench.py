"""Step 1.1 micro-benchmark: how long one LatencyLog.mark() call blocks the caller.

Usage: python -m bench.latency_mark_bench [--calls 20000]
Target: < 50 µs per call (p50), because mark() runs on the hot path, including audio callbacks.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from metrics.latency import LatencyLog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=20_000)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as folder:
        log = LatencyLog(Path(folder) / "latency.jsonl", run_id="bench")
        log.next_turn()
        durations = np.empty(args.calls)
        for i in range(args.calls):
            started = time.perf_counter_ns()
            log.mark("stt_final", text="hello world")
            durations[i] = (time.perf_counter_ns() - started) / 1000
        flush = getattr(log, "flush", None)
        if flush:
            flush_started = time.perf_counter()
            flush()
            print(f"flush() of {args.calls} records: {(time.perf_counter() - flush_started) * 1000:.0f} ms")
        lines = (Path(folder) / "latency.jsonl").read_text(encoding="utf-8").count("\n")
        close = getattr(log, "close", None)
        if close:
            close()

    print(f"mark() over {args.calls} calls: p50 {np.percentile(durations, 50):.1f} µs, "
          f"p95 {np.percentile(durations, 95):.1f} µs, p99 {np.percentile(durations, 99):.1f} µs, "
          f"max {durations.max():.0f} µs (target p50 < 50 µs)")
    print(f"records written: {lines} of {args.calls}")


if __name__ == "__main__":
    main()
