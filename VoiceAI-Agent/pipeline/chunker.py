"""Turns the LLM token stream into clauses that are sent to TTS one by one.

Rules (roadmap Step 1.3):
- The FIRST clause is short, so the first audio starts quickly: cut at the first , . ! ? ; :
  after >= FIRST_MIN_WORDS words, or at a word boundary once FIRST_MAX_WORDS words are waiting.
- Later clauses have >= MIN_WORDS words and are cut at punctuation, or at a word boundary
  after MAX_WORDS words so a long unpunctuated run can't block the voice.
- A punctuation mark only ends a clause when the NEXT character is whitespace (or the stream
  ends). That is what keeps "3.5", "1,000", "10:30", "example.com" and "a.m." in one piece:
  the chunker waits for the next token before deciding.
- Abbreviations ("Dr.", "e.g.", "Mr.", single initials like "J.") never end a clause.
- A newline always ends a clause (list items, paragraphs).
"""

from __future__ import annotations

import re

BOUNDARY = set(",.!?;:")
STRONG = set(".!?")
CLOSERS = "\"')]}»”’"
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "eg", "ie",
    "a.m", "p.m", "u.s", "u.k", "fig", "inc", "ltd", "approx", "dept", "mt", "rs",
}  # not "no", "min", "max": they are common last words of a sentence
FIRST_MIN_WORDS = 2
FIRST_MAX_WORDS = 6
MIN_WORDS = 4
MAX_WORDS = 25
EXTRA_WORDS = 3
WEAK_ENDINGS = {
    "a", "an", "the", "and", "or", "but", "so", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "it", "its", "that", "this", "if",
    "than", "then", "my", "your", "our", "their", "his", "her", "i", "you", "we", "they",
    "can", "will", "would", "should", "could", "do", "does", "not", "very",
}


def count_words(text: str) -> int:
    return len(text.split())


class ClauseChunker:
    def __init__(self, first_min_words: int = FIRST_MIN_WORDS, first_max_words: int = FIRST_MAX_WORDS,
                 min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> None:
        self.first_min_words = first_min_words
        self.first_max_words = first_max_words
        self.min_words = min_words
        self.max_words = max_words
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0

    @property
    def pending(self) -> str:
        return self._buffer

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return the clauses that are now complete (possibly none)."""
        self._buffer += text
        clauses = []
        while True:
            clause = self._next_clause(final=False)
            if clause is None:
                break
            clauses.append(clause)
        return clauses

    def flush(self) -> list[str]:
        """The stream ended: return whatever is left, split at safe boundaries."""
        clauses = []
        while True:
            clause = self._next_clause(final=True)
            if clause is None:
                break
            clauses.append(clause)
        rest = self._buffer.strip()
        self._buffer = ""
        if rest:
            clauses.append(rest)
            self._emitted += 1
        return clauses

    # -- internals -------------------------------------------------------------------------
    def _next_clause(self, final: bool) -> str | None:
        first = self._emitted == 0
        min_words = self.first_min_words if first else self.min_words
        max_words = self.first_max_words if first else self.max_words
        cut = self._find_punctuation_cut(min_words, final)
        if cut is None:
            cut = self._find_length_cut(max_words, final)
        if cut is None:
            return None
        clause, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:]
        if not clause:
            return None
        self._emitted += 1
        return clause

    def _find_punctuation_cut(self, min_words: int, final: bool) -> int | None:
        text = self._buffer
        for index, char in enumerate(text):
            if char == "\n":
                if text[:index].strip():
                    return index + 1
                continue
            if char not in BOUNDARY:
                continue
            end = index + 1
            while end < len(text) and text[end] in CLOSERS:  # keep closing quotes/brackets
                end += 1
            if end < len(text):
                if not text[end].isspace():
                    continue  # "3.5", "10:30", "example.com", "...": not a boundary (yet)
            elif not final:
                return None  # can't know what comes next: wait for the next token
            if char == "." and self._is_abbreviation(text[:index]):
                continue
            words = count_words(text[:end])
            # Too short for a clause, except strong punctuation one word short of a first
            # clause ("Okay." / "Sure!"), so the first answer isn't held back.
            short_but_final = char in STRONG and min_words <= 2 and words >= min_words - 1
            if words < min_words and not short_but_final:
                continue
            return end
        return None

    def _find_length_cut(self, max_words: int, final: bool) -> int | None:
        """Cut after `max_words` complete words (a word is complete once whitespace follows).

        A forced cut doesn't end on a function word ("... and it" / "... the"): it moves up to
        EXTRA_WORDS further, so the clause sounds finished.
        """
        words: list[tuple[str, int]] = []  # (word, index of the whitespace after it)
        for match in re.finditer(r"(\S+)(\s)", self._buffer):
            words.append((match.group(1), match.end(1)))
        if len(words) < max_words:
            return None
        for count in range(max_words, min(len(words), max_words + EXTRA_WORDS) + 1):
            word, end = words[count - 1]
            if word.lower().strip(",.!?;:\"'") not in WEAK_ENDINGS:
                return end
        if len(words) >= max_words + EXTRA_WORDS or final:
            return words[min(len(words), max_words + EXTRA_WORDS) - 1][1]
        return None  # wait for more words: the next one may be a good place to cut

    @staticmethod
    def _is_abbreviation(before: str) -> bool:
        match = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
        if not match:
            return False
        word = match.group(1).lower().rstrip(".")
        if len(word) == 1 and match.group(1)[0].isupper() and word != "i":
            return True  # initials: "J. K. Rowling" (but not "... than I.")
        return word in ABBREVIATIONS
