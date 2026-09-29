"""Word error rate with text normalisation, so formatting differences don't count as errors.

normalize(): lowercase, numbers to words ("3" / "three", "3.5" / "three point five",
"10:30" / "ten thirty", "50%" / "fifty percent"), "example.com" -> "example dot com",
"&" -> "and", no punctuation, hyphens and
slashes become spaces, apostrophes are dropped ("it's" == "its"), whitespace collapsed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from num2words import num2words


def _number_words(match: re.Match) -> str:
    token = match.group(0)
    if ":" in token:  # a time: 10:30 -> ten thirty, 9:05 -> nine oh five
        hours, minutes = token.split(":", 1)
        minute = int(minutes)
        spoken_minutes = "" if minute == 0 else ("oh " + num2words(minute) if minute < 10 else num2words(minute))
        return f" {num2words(int(hours))} {spoken_minutes} "
    number = token.replace(",", "")
    if "." in number:
        whole, fraction = number.split(".", 1)
        return f" {num2words(int(whole or 0))} point {' '.join(num2words(int(d)) for d in fraction)} "
    return f" {num2words(int(number))} "


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", " dot ", text)  # example.com / settings.py
    text = text.replace("&", " and ").replace("%", " percent ")
    text = re.sub(r"\d{1,2}:\d{2}|\d[\d,]*(?:\.\d+)?", _number_words, text)
    text = text.replace("-", " ").replace("/", " ")
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


@dataclass(frozen=True)
class WER:
    errors: int  # substitutions + deletions + insertions
    words: int  # words in the reference

    @property
    def rate(self) -> float:
        return self.errors / self.words if self.words else 0.0

    def __add__(self, other: WER) -> WER:
        return WER(self.errors + other.errors, self.words + other.words)


def word_errors(reference: str, hypothesis: str) -> WER:
    """Levenshtein distance over normalised words."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
        previous = current
    return WER(previous[-1], len(ref))
