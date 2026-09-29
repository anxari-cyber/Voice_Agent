"""Conversation history for the Voice Brain, kept within a token budget.

Ollama has no tokenize endpoint, so token counts are estimated. The estimate is deliberately
conservative (it over-counts English text for the Qwen tokenizer), so the prompt never
overflows num_ctx. bench/llm_bench.py compares it with Ollama's real prompt_eval_count.
"""

from __future__ import annotations

from dataclasses import dataclass

MESSAGE_OVERHEAD = 5  # <|im_start|>role\n ... <|im_end|>\n


def estimate_tokens(text: str) -> int:
    """~3 characters per token (real Qwen English is ~4), plus the chat-template overhead."""
    return len(text) // 3 + 1 + MESSAGE_OVERHEAD


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str

    def as_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class Memory:
    def __init__(self) -> None:
        self.turns: list[Turn] = []

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

        Whole turns are dropped from the oldest end, and the history never starts with an
        assistant turn, so the model always sees a user turn first.
        """
        kept: list[Turn] = []
        used = 0
        for turn in reversed(self.turns):
            cost = estimate_tokens(turn.content)
            if used + cost > budget_tokens:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        while kept and kept[0].role == "assistant":
            kept.pop(0)
        return [turn.as_message() for turn in kept]
