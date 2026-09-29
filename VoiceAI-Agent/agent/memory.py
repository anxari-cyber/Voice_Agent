"""Conversation history for the Voice Brain, kept within a token budget.

Ollama has no tokenize endpoint, so token counts are estimated by `TokenEstimator`, which
calibrates itself from Ollama's real `prompt_eval_count` after every reply. It starts
conservative (3 characters per token) and never assumes more than MAX_CHARS_PER_TOKEN, so an
odd measurement can't make it under-count and overflow num_ctx.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

MESSAGE_OVERHEAD = 5  # <|im_start|>role\n ... <|im_end|>\n
START_CHARS_PER_TOKEN = 3.0  # safe start (real Qwen English is ~4)
MIN_CHARS_PER_TOKEN = 2.0  # most conservative the estimate gets (code, symbols, Urdu)
MAX_CHARS_PER_TOKEN = 4.0  # safe floor on the token estimate: never assume denser text than this
SAFETY = 0.9  # use 90% of the measured ratio, so the estimate stays above the real count


class TokenEstimator:
    def __init__(self, chars_per_token: float = START_CHARS_PER_TOKEN, smoothing: float = 0.3) -> None:
        self.measured = chars_per_token  # running average of observed chars per token
        self.smoothing = smoothing
        self.samples = 0
        self._lock = threading.Lock()

    @property
    def chars_per_token(self) -> float:
        """Ratio actually used: measured ratio minus a safety margin, clamped to a safe range."""
        return min(MAX_CHARS_PER_TOKEN, max(MIN_CHARS_PER_TOKEN, self.measured * SAFETY))

    def tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 1 + MESSAGE_OVERHEAD

    def messages_tokens(self, messages: list[dict]) -> int:
        return sum(self.tokens(m.get("content", "")) for m in messages)

    def calibrate(self, messages: list[dict], prompt_eval_count: int) -> None:
        """Learn from one real request: `prompt_eval_count` is Ollama's count for `messages`."""
        chars = sum(len(m.get("content", "")) for m in messages)
        text_tokens = prompt_eval_count - MESSAGE_OVERHEAD * len(messages)
        if chars < 200 or text_tokens <= 0:  # too small to say anything reliable
            return
        observed = chars / text_tokens
        with self._lock:
            if self.samples == 0:
                self.measured = min(self.measured, observed)  # first sample: only get safer fast
            self.measured += self.smoothing * (observed - self.measured)
            self.samples += 1


DEFAULT_ESTIMATOR = TokenEstimator()


def estimate_tokens(text: str) -> int:
    return DEFAULT_ESTIMATOR.tokens(text)


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str

    def as_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class Memory:
    def __init__(self, estimator: TokenEstimator | None = None) -> None:
        self.turns: list[Turn] = []
        self.estimator = estimator or DEFAULT_ESTIMATOR

    def add_user(self, text: str) -> None:
        self.turns.append(Turn("user", text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(Turn("assistant", text))

    def truncate_last_assistant(self, spoken_text: str) -> None:
        """Keep only what the user actually heard of the last answer (barge-in).

        If nothing was heard, the answer is removed completely.
        """
        for index in range(len(self.turns) - 1, -1, -1):
            if self.turns[index].role == "assistant":
                if spoken_text.strip():
                    self.turns[index].content = spoken_text.strip()
                else:
                    del self.turns[index]
                return

    def clear(self) -> None:
        self.turns.clear()

    def messages(self, budget_tokens: int) -> list[dict]:
        """The most recent turns that fit in `budget_tokens`, oldest first.

        Consecutive turns with the same role are merged (e.g. two user turns in a row after
        an unheard answer was removed), whole turns are dropped from the oldest end, and the
        history never starts with an assistant turn.
        """
        merged = merge_same_role([turn.as_message() for turn in self.turns])
        kept: list[dict] = []
        used = 0
        for message in reversed(merged):
            cost = self.estimator.tokens(message["content"])
            if used + cost > budget_tokens:
                break
            kept.append(message)
            used += cost
        kept.reverse()
        while kept and kept[0]["role"] == "assistant":
            kept.pop(0)
        return kept


def merge_same_role(messages: list[dict]) -> list[dict]:
    """Join consecutive messages from the same speaker into one (speech fragments -> sentence)."""
    merged: list[dict] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"] and message["role"] != "system":
            merged[-1] = {**merged[-1], "content": f"{merged[-1]['content']} {message['content']}".strip()}
        else:
            merged.append(dict(message))
    return merged
