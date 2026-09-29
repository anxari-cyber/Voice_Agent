import threading
import time

import numpy as np
import pytest

from metrics.wer import normalize, word_errors
from voice.stt import (
    SAMPLE_RATE,
    StreamingSTT,
    Word,
    align_on_anchor,
    join_words,
    words_from_tokens,
)

# A scripted utterance: word i starts at 0.1 + 0.4 * i seconds. It repeats the common word
# "the" on purpose, the case that once made alignment match the wrong word.
SENTENCE = ("please open the project and check the login test then run the unit tests and "
            "tell me which of the tests still fail before the end of the day")
SCRIPT = SENTENCE.split(" ")
WORD_TIMES = [0.1 + 0.4 * i for i in range(len(SCRIPT))]
SPEECH_END = WORD_TIMES[-1] + 0.35


class FakeTimedEngine:
    """Knows which audio it got: every sample's VALUE is its position in the utterance.

    Imitates Parakeet's quirks: timestamps are late, so the word before the window start is
    heard again ("remnant"); optionally the first word of a trimmed window is skipped.
    """

    name = "fake-timed"

    def __init__(self, delay: float = 0.0, skip_first: bool = False) -> None:
        self.delay = delay
        self.skip_first = skip_first
        self.calls: list[float] = []

    def transcribe_words(self, audio: np.ndarray) -> list[Word]:
        time.sleep(self.delay)
        if audio.size == 0:
            return []
        start = float(audio[0]) / SAMPLE_RATE
        end = float(audio[-1]) / SAMPLE_RATE
        self.calls.append(end - start)
        words = []
        for text, t in zip(SCRIPT, WORD_TIMES):
            heard = t >= start - 0.25 if start > 0 else t >= 0  # late timestamps -> remnant
            if heard and t + 0.1 <= end:
                words.append(Word(text, max(0.0, t - start)))
        if self.skip_first and start > 0 and words:
            words = words[1:]
        return words

    def transcribe(self, audio: np.ndarray) -> str:
        return " ".join(w.text for w in self.transcribe_words(audio))


def utterance(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32)


def feed_all(stt: StreamingSTT, audio: np.ndarray, block: int = 320) -> None:
    stt.start()
    for i in range(0, len(audio), block):
        stt.feed(audio[i : i + block])


@pytest.mark.parametrize("skip_first", [False, True])
def test_long_utterance_has_no_duplicates_or_losses(skip_first: bool) -> None:
    engine = FakeTimedEngine(skip_first=skip_first)
    stt = StreamingSTT(engine, synchronous=True)
    feed_all(stt, utterance(SPEECH_END + 0.2))
    text = stt.finish(int(SPEECH_END * SAMPLE_RATE))
    assert text == " ".join(SCRIPT)


def test_decode_window_stays_short_for_long_speech() -> None:
    engine = FakeTimedEngine()
    stt = StreamingSTT(engine, synchronous=True, trim_after_s=2.0)  # force trims on the ~11 s script
    feed_all(stt, utterance(SPEECH_END + 0.2))
    assert stt._window_start > 0  # the window moved
    assert max(engine.calls) < 3.5  # no decode ever covered the whole ~11 s utterance
    stt.finish(int(SPEECH_END * SAMPLE_RATE))
    assert stt.last_final["decoded_s"] < 3.5


def test_last_partial_is_reused_when_it_covered_the_speech() -> None:
    stt = StreamingSTT(FakeTimedEngine(), synchronous=True)
    feed_all(stt, utterance(SPEECH_END + 0.6))  # a partial runs after the speech ended
    stt.finish(int(SPEECH_END * SAMPLE_RATE))
    assert stt.last_final["mode"] == "reused"
    assert stt.last_final["decoded_s"] == 0.0


def test_final_does_not_wait_for_a_slow_stale_partial() -> None:
    slow, fast = FakeTimedEngine(delay=0.5), FakeTimedEngine()
    stt = StreamingSTT(slow, final_engine=fast, partial_every_ms=100)
    stt.start()
    audio = utterance(1.5)
    for i in range(0, len(audio), 320):
        stt.feed(audio[i : i + 320])
    started = time.perf_counter()
    text = stt.finish()
    assert time.perf_counter() - started < 0.3  # did not wait ~0.5 s for the partial
    assert text.startswith("please open")
    time.sleep(0.6)  # the stale partial finishes now...
    assert stt.partial_text == text  # ...and its result is ignored
    stt.close()


def test_overlapping_partial_and_final_decodes_are_safe() -> None:
    # Two engines -> no shared state. Run many overlapping finals and partials in threads.
    stt = StreamingSTT(FakeTimedEngine(delay=0.002), final_engine=FakeTimedEngine(), partial_every_ms=20)
    errors = []

    def one() -> None:
        try:
            feed_all(stt, utterance(1.0), block=160)
            stt.finish()
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=one) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stt.close()
    assert errors == []


def test_align_needs_the_full_anchor_near_its_time() -> None:
    anchor = [Word("the", 11.4), Word("changes.", 11.6)]
    # The decoder skipped the anchor; "the" appears later in "the database": must NOT match it.
    decoded = [Word("If", 12.3), Word("the", 12.5), Word("database", 12.7)]
    assert [w.text for w in align_on_anchor(decoded, anchor)] == ["the", "changes.", "If", "the", "database"]
    # Found properly, with a remnant of the previous word before it: the remnant is dropped.
    decoded = [Word("rize", 11.2), Word("the", 11.4), Word("changes.", 11.6), Word("If", 12.3)]
    assert [w.text for w in align_on_anchor(decoded, anchor)] == ["the", "changes.", "If"]


def test_words_from_tokens_joins_subwords() -> None:
    tokens = [" He", "ll", "o", ",", " can", " you", " ch", "e", "ck"]
    stamps = [0.64, 1.12, 1.28, 1.44, 1.6, 1.76, 1.92, 2.0, 2.08]
    assert words_from_tokens(tokens, stamps) == [Word("Hello,", 0.64), Word("can", 1.6),
                                                 Word("you", 1.76), Word("check", 1.92)]


# ---- WER normalisation ----------------------------------------------------------------------

@pytest.mark.parametrize(("a", "b"), [
    ("Open the project.", "open the project"),
    ("It's 3 files", "its three files"),
    ("Version 3.5", "version three point five"),
    ("At 10:30 today", "at ten thirty today"),
    ("At 9:05", "at nine oh five"),
    ("It's 50% done", "its fifty percent done"),
    ("1,000 users", "one thousand users"),
    ("git-status / log", "git status log"),
    ("Tom & Jerry!", "tom and jerry"),
    ("Open example.com now.", "open example dot com now"),
    ("edit settings.py", "edit settings dot py"),
])
def test_normalize_makes_formatting_irrelevant(a: str, b: str) -> None:
    assert normalize(a) == normalize(b)
    assert word_errors(a, b).errors == 0


def test_word_errors_counts_substitutions_insertions_deletions() -> None:
    result = word_errors("open the login test", "open a login test now")
    assert result.words == 4
    assert result.errors == 2  # "the" -> "a", + "now"
    assert result.rate == 0.5
    assert word_errors("", "").rate == 0.0
    assert (word_errors("a b", "a c") + word_errors("x", "x")).rate == pytest.approx(1 / 3)


def test_anchor_matches_even_when_the_spelling_changes() -> None:
    anchor = [Word("what", 1.12), Word("Arrest", 1.28)]
    decoded = [Word("Explain", 0.98), Word("what", 1.3), Word("AREST", 1.46), Word("API", 1.86)]
    assert [w.text for w in align_on_anchor(decoded, anchor)] == ["what", "Arrest", "API"]
    # Only the last anchor word re-spelled, and the sequence not found: still no duplicate.
    decoded = [Word("AREST", 1.46), Word("API", 1.86)]
    assert [w.text for w in align_on_anchor(decoded, anchor)] == ["what", "Arrest", "API"]


def test_join_words_attaches_leading_punctuation_like_the_decoder() -> None:
    tokens = [" made", " to", " set", "ting", "s", " ", ".", "p", "y", "."]
    stamps = [float(i) for i in range(len(tokens))]
    words = words_from_tokens(tokens, stamps)
    assert join_words([w.text for w in words]) == "made to settings.py."
    assert join_words(["Hello", ",", "world", '"quoted"']) == 'Hello, world "quoted"'
