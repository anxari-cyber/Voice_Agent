"""Streaming LLM client for the Voice Brain (Ollama /api/chat).

Latency rules this client follows (roadmap Step 1.2):
- One persistent HTTP connection (httpx.Client), created once.
- `options` (num_ctx, num_predict, temperature) are built once and sent unchanged on every
  request, including the warm-up. A different num_ctx makes Ollama reload the model.
- `think: false`, `keep_alive: -1` (the model stays in VRAM).
- `stream()` yields Deltas as tokens arrive; `cancel()` closes the stream from any thread.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass(frozen=True)
class Delta:
    """One piece of a streamed reply. The last Delta has done=True and carries stats."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    done: bool = False
    stats: dict = field(default_factory=dict)


class LLM(Protocol):
    def stream(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> Iterator[Delta]: ...

    def cancel(self) -> None: ...


class OllamaLLM:
    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b-instruct-2507-q4_K_M",
        num_ctx: int = 4096,
        max_tokens: int = 200,
        temperature: float = 0.6,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.options = {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature}
        self._client = httpx.Client(
            base_url=url.rstrip("/"),
            timeout=httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=5.0),
            transport=transport,
        )
        self._lock = threading.Lock()
        self._response: httpx.Response | None = None
        self._cancelled = threading.Event()

    def payload(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> dict:
        body = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "think": False,
            "keep_alive": -1,
            "options": self.options,
        }
        if tools:
            body["tools"] = list(tools)
        return body

    def stream(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> Iterator[Delta]:
        self._cancelled.clear()
        try:
            with self._client.stream("POST", "/api/chat", json=self.payload(messages, tools)) as response:
                if response.status_code != 200:
                    response.read()
                    raise LLMError(f"Ollama returned {response.status_code}: {response.text[:300]}")
                with self._lock:
                    self._response = response
                for line in response.iter_lines():
                    if self._cancelled.is_set():
                        return
                    if not line:
                        continue
                    delta = _parse(line)
                    yield delta
                    if delta.done:
                        return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.StreamClosed):
            if self._cancelled.is_set():
                return  # cancel() closed the socket under us
            raise
        except httpx.HTTPError as error:
            raise LLMError(f"Ollama request failed: {error}") from error
        finally:
            with self._lock:
                self._response = None

    def cancel(self) -> None:
        """Stop the current stream. Safe to call from another thread, or when idle."""
        self._cancelled.set()
        with self._lock:
            response = self._response
        if response is not None:
            response.close()  # closing the connection makes Ollama stop generating

    def warm_up(self, messages: Sequence[dict]) -> dict:
        """Load the model and cache the prompt prefix, using exactly the real options.

        Stops after the first token instead of changing num_predict, so the warm-up request
        is identical to a real one.
        """
        for delta in self.stream(messages):
            if delta.content or delta.done:
                stats = delta.stats
                self.cancel()
                return stats
        return {}

    def close(self) -> None:
        self.cancel()
        self._client.close()


def _parse(line: str) -> Delta:
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError as error:
        raise LLMError(f"Bad line from Ollama: {line[:200]!r}") from error
    if "error" in chunk:
        raise LLMError(f"Ollama error: {chunk['error']}")
    message = chunk.get("message") or {}
    calls = tuple(
        ToolCall(
            name=call.get("function", {}).get("name", ""),
            arguments=call.get("function", {}).get("arguments") or {},
            id=call.get("id"),
        )
        for call in message.get("tool_calls") or ()
    )
    done = bool(chunk.get("done"))
    stats = {k: v for k, v in chunk.items() if k not in ("message", "model", "created_at")} if done else {}
    return Delta(content=message.get("content") or "", tool_calls=calls, done=done, stats=stats)
