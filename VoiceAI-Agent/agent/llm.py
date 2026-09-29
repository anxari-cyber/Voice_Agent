"""Streaming LLM client for the Voice Brain (Ollama /api/chat).

Latency rules this client follows (roadmap Step 1.2):
- One persistent HTTP connection pool (httpx.Client), created once.
- `options` (num_ctx, num_predict, temperature) are built once and sent unchanged on every
  request, including the warm-up. A different num_ctx makes Ollama reload the model.
- `think: false`, `keep_alive: -1` (the model stays in VRAM).
- `stream()` returns an `LLMStream`: iterate it for Deltas, call its own `cancel()` to stop it.
  Each request has its own handle, so cancelling one never affects another (overlapping
  streams happen with speculative starts and barge-in).
- A request with no first chunk after `header_timeout` (2 s) is retried once with
  `first_token_timeout` (10 s): one request got stuck without headers for 10 s in testing.
  A stream that stalls after content arrived is an error (no retry: the user heard part of it).
  The warm-up gets a long timeout because it may load the model.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
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


class LLMStream(Protocol):
    def __iter__(self) -> Iterator[Delta]: ...

    def cancel(self) -> None: ...


class LLM(Protocol):
    def stream(self, messages: Sequence[dict], tools: Sequence[dict] | None = None) -> LLMStream: ...

    def cancel(self) -> None: ...


class OllamaStream:
    """One streamed request. Iterate it once; `cancel()` works from any thread, any time."""

    def __init__(self, client: httpx.Client, payload: dict, timeout: float,
                 on_finish: Callable[[OllamaStream], None] = lambda stream: None,
                 header_timeout: float | None = None) -> None:
        self._client = client
        self._payload = payload
        self._timeout = timeout
        # First attempt uses the short header timeout; if NOTHING arrived, retry once with
        # the normal timeout. Ollama sends headers together with the first chunk (measured:
        # 140 ms even for ~2,400 uncached prompt tokens), so 2 s only trips on a stuck request.
        self._attempts = [header_timeout, timeout] if header_timeout and header_timeout < timeout else [timeout]
        self.retries = 0
        self._on_finish = on_finish
        self._lock = threading.Lock()
        self._response: httpx.Response | None = None
        self._cancelled = threading.Event()
        self.stats: dict = {}

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            response = self._response
        if response is not None:
            response.close()  # closing the connection makes Ollama stop generating

    def __iter__(self) -> Iterator[Delta]:
        try:
            for attempt, read_timeout in enumerate(self._attempts):
                if self._cancelled.is_set():
                    return
                received = False
                try:
                    for delta in self._request(read_timeout):
                        received = True
                        yield delta
                    return
                except httpx.ReadTimeout as error:
                    if received or attempt == len(self._attempts) - 1:
                        raise LLMError(f"Ollama sent nothing for {read_timeout:.0f} s") from error
                    self.retries += 1  # stuck before the first chunk: try once more
        finally:
            with self._lock:
                self._response = None
            self._on_finish(self)

    def _request(self, read_timeout: float) -> Iterator[Delta]:
        timeout = httpx.Timeout(connect=3.0, read=read_timeout, write=10.0, pool=5.0)
        try:
            with self._client.stream("POST", "/api/chat", json=self._payload, timeout=timeout) as response:
                if response.status_code != 200:
                    response.read()
                    raise LLMError(f"Ollama returned {response.status_code}: {response.text[:300]}")
                with self._lock:
                    self._response = response
                if self._cancelled.is_set():  # cancelled while the request was being sent
                    return
                for line in response.iter_lines():
                    if self._cancelled.is_set():
                        return
                    if not line:
                        continue
                    delta = _parse(line)
                    if delta.done:
                        self.stats = delta.stats
                    yield delta
                    if delta.done:
                        return
        except httpx.ReadTimeout:
            raise
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.StreamClosed):
            if self._cancelled.is_set():
                return  # cancel() closed the socket under us
            raise
        except httpx.HTTPError as error:
            if self._cancelled.is_set():
                return
            raise LLMError(f"Ollama request failed: {error}") from error
        finally:
            with self._lock:
                self._response = None


class OllamaLLM:
    def __init__(
        self,
        url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:4b-instruct-2507-q4_K_M",
        num_ctx: int = 4096,
        max_tokens: int = 200,
        temperature: float = 0.6,
        first_token_timeout: float = 10.0,
        warm_up_timeout: float = 120.0,
        header_timeout: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.options = {"num_ctx": num_ctx, "num_predict": max_tokens, "temperature": temperature}
        self.first_token_timeout = first_token_timeout
        self.warm_up_timeout = warm_up_timeout
        self.header_timeout = header_timeout
        self._client = httpx.Client(base_url=url.rstrip("/"), transport=transport)
        self._active: set[OllamaStream] = set()
        self._active_lock = threading.Lock()

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

    def stream(self, messages: Sequence[dict], tools: Sequence[dict] | None = None,
               timeout: float | None = None) -> OllamaStream:
        long_wait = timeout is not None  # warm-up: may load the model, no short first attempt
        stream = OllamaStream(self._client, self.payload(messages, tools),
                              timeout or self.first_token_timeout, on_finish=self._forget,
                              header_timeout=None if long_wait else self.header_timeout)
        with self._active_lock:
            self._active.add(stream)
        return stream

    def cancel(self) -> None:
        """Cancel every stream that is still running (e.g. at shutdown)."""
        with self._active_lock:
            streams = list(self._active)
        for stream in streams:
            stream.cancel()

    def warm_up(self, messages: Sequence[dict]) -> None:
        """Load the model and cache the prompt prefix, using exactly the real options.

        Stops after the first token instead of changing num_predict, so the warm-up request is
        identical to a real one. Uses the long timeout: a cold load from disk can take 30 s.
        """
        stream = self.stream(messages, timeout=self.warm_up_timeout)
        for delta in stream:
            if delta.content or delta.done:
                stream.cancel()
                return

    def close(self) -> None:
        self.cancel()
        self._client.close()

    def _forget(self, stream: OllamaStream) -> None:
        with self._active_lock:
            self._active.discard(stream)


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
