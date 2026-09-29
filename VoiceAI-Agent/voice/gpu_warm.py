"""Wake the GPU up before it's needed (plan amendment #11).

After ~10 s without work the RTX 5050 drops to its idle clock (P8, ~200 MHz), and the first LLM
request afterwards pays ~120 ms extra TTFT while the clock ramps up. A short burst of CUDA work
fixes that (measured: 155-228 ms -> 34-45 ms). GPU clocks are device-wide, so Ollama benefits too.

The orchestrator calls `kick()` at VAD speech_start and speech_end: the burst runs in the
background while the user is still talking or in the join window, so its cost is hidden. One kick
at speech_start alone was not enough for long commands: the clock drops again within ~3 s.
"""

from __future__ import annotations

import threading
import time


class GpuWarmer:
    def __init__(self, burst_ms: float = 150, min_interval_s: float = 0.5) -> None:
        self.burst_ms = burst_ms
        self.min_interval_s = min_interval_s
        self._busy = threading.Lock()
        self._last = 0.0
        self._tensor = None
        self.kicks = 0
        self.available = self._init()

    def _init(self) -> bool:
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            self._torch = torch
            self._tensor = torch.randn(1024, 1024, device="cuda")
            return True
        except Exception:  # noqa: BLE001 - no GPU warming is fine, just slower TTFT
            return False

    def kick(self) -> bool:
        """Start a burst in the background (no-op if one ran very recently or is running)."""
        if not self.available or time.perf_counter() - self._last < self.min_interval_s:
            return False
        if not self._busy.acquire(blocking=False):
            return False
        self._last = time.perf_counter()
        self.kicks += 1
        threading.Thread(target=self._burst, name="gpu-warm", daemon=True).start()
        return True

    def _burst(self) -> None:
        try:
            self._torch.cuda.set_device(self._tensor.device)  # new thread: bind the CUDA context
            end = time.perf_counter() + self.burst_ms / 1000
            while time.perf_counter() < end:
                self._tensor @ self._tensor  # the work itself is the point
            self._torch.cuda.synchronize()
        finally:
            self._busy.release()
