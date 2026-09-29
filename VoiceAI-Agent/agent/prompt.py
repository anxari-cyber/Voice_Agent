"""Builds the message list for every LLM call: fixed system prompt + recent history + new turn.

The system prompt is read ONCE and the same dict is reused for every request, so its bytes
never change. Ollama/llama.cpp can then reuse the cached prompt prefix instead of
re-processing it every turn. Never put anything dynamic (time, date, names that change) in it;
dynamic context belongs after it, in the history or the user turn.
"""

from __future__ import annotations

from pathlib import Path

from agent.memory import DEFAULT_ESTIMATOR, Memory, TokenEstimator, merge_same_role

SAFETY_MARGIN = 64  # tokens kept free for template details the estimate doesn't cover


class PromptBuilder:
    def __init__(self, system_prompt: str, num_ctx: int, max_tokens: int,
                 estimator: TokenEstimator | None = None) -> None:
        if not system_prompt.strip():
            raise ValueError("The system prompt is empty.")
        self.system_message = {"role": "system", "content": system_prompt}
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.estimator = estimator or DEFAULT_ESTIMATOR
        system_tokens = self.estimator.tokens(system_prompt)
        if system_tokens + max_tokens + SAFETY_MARGIN >= num_ctx:
            raise ValueError(
                f"System prompt (~{system_tokens} tokens) + max_tokens ({max_tokens}) "
                f"does not fit in num_ctx ({num_ctx})."
            )

    @classmethod
    def from_file(cls, path: Path | str, num_ctx: int, max_tokens: int,
                  estimator: TokenEstimator | None = None) -> PromptBuilder:
        text = Path(path).read_text(encoding="utf-8").strip()
        return cls(text, num_ctx, max_tokens, estimator)

    @property
    def system_tokens(self) -> int:
        return self.estimator.tokens(self.system_message["content"])

    def history_budget(self, user_text: str) -> int:
        """Tokens left for history once the system prompt, new turn and the reply are reserved."""
        reserved = (self.system_tokens + self.estimator.tokens(user_text)
                    + self.max_tokens + SAFETY_MARGIN)
        return max(0, self.num_ctx - reserved)

    def build(self, memory: Memory, user_text: str) -> list[dict]:
        history = memory.messages(self.history_budget(user_text))
        # If the history already ends with the user speaking (their last turn got no answer),
        # the new words continue that turn instead of creating two user turns in a row.
        turns = merge_same_role([*history, {"role": "user", "content": user_text}])
        return [self.system_message, *turns]

    def calibrate(self, messages: list[dict], stats: dict, used_tools: bool = False) -> None:
        """Teach the estimator from a finished reply. Skipped when tools were sent, because
        the tool schemas are counted in prompt_eval_count but aren't in `messages`."""
        count = stats.get("prompt_eval_count")
        if count and not used_tools:
            self.estimator.calibrate(messages, count)

    def warm_up_messages(self) -> list[dict]:
        """Same prefix as a real turn, so warming up caches exactly what real turns reuse."""
        return [self.system_message, {"role": "user", "content": "Hi."}]
