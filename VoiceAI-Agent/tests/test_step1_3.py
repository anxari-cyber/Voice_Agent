import threading
import time

import numpy as np
import pytest

from pipeline.chunker import ClauseChunker
from pipeline.tts_worker import TTSWorker
from voice.playback import Player
from voice.resample import StatefulResampler
from voice.tts import SAMPLE_RATE, pause_after, trim_silence


def chunk_stream(text: str, step: int = 3, **kwargs) -> list[str]:
    """Feed `text` in small pieces, like an LLM token stream."""
    chunker = ClauseChunker(**kwargs)
    clauses = []
    for i in range(0, len(text), step):
        clauses += chunker.feed(text[i : i + step])
    return clauses + chunker.flush()


# ---- chunker --------------------------------------------------------------------------------

def test_first_clause_is_short_then_normal_clauses() -> None:
    clauses = chunk_stream("Okay. Let me check the login test for you, it should only take a moment.")
    assert clauses == ["Okay.", "Let me check the login test for you,", "it should only take a moment."]


@pytest.mark.parametrize("step", [1, 2, 3, 7, 1000])
def test_result_does_not_depend_on_token_size(step: int) -> None:
    text = "Sure, here it is. The file has 3.5 thousand lines, and it was changed at 10:30 today."
    assert chunk_stream(text, step) == chunk_stream(text, 1)
    assert " ".join(chunk_stream(text, step)) == text


@pytest.mark.parametrize("text", [
    "The version is 3.5 and it works well on Windows now.",
    "It costs 1,000 rupees or about 3.50 dollars at the moment.",
    "The meeting is at 10:30 in the main hall tomorrow morning.",
    "Please open example.com and then the file config.yaml again.",
    "Send it to test.user@example.org before the end of the day.",
])
def test_never_cut_inside_numbers_times_urls_or_emails(text: str) -> None:
    joined = " ".join(chunk_stream(text))
    assert joined == text
    for clause in chunk_stream(text):
        assert not clause.endswith(("3.", "1,", "10:", "example.", "config.", "test.")), clause


@pytest.mark.parametrize("text", [
    "Dr. Smith said the results look good today.",
    "Bring fruit, e.g. apples or pears, to the party.",
    "Mr. and Mrs. Khan arrived at 5 p.m. with J. K. Rowling.",
    "You can use tools, i.e. the terminal and git, for this.",
])
def test_abbreviations_do_not_end_a_clause(text: str) -> None:
    for clause in chunk_stream(text):
        assert not clause.endswith(("Dr.", "e.g.", "Mr.", "Mrs.", "p.m.", "J.", "K.", "i.e.")), clause


def test_waits_for_the_next_token_before_cutting_at_a_dot() -> None:
    chunker = ClauseChunker()
    assert chunker.feed("The value is 3.") == []  # could be "3.5": wait
    assert chunker.feed("5 today, as expected.") == ["The value is 3.5 today,"]
    assert chunker.flush() == ["as expected."]


def test_question_exclamation_ellipsis_and_closing_quotes() -> None:
    assert chunk_stream('Really? Yes! She said "we are done." Then... we left the room.') == [
        "Really?", 'Yes! She said "we are done."', "Then... we left the room."]


def test_newlines_end_clauses_for_list_items() -> None:
    assert chunk_stream("Here are the steps:\n1. open the folder\n2. run the tests") == [
        "Here are the steps:", "1. open the folder", "2. run the tests"]


def test_long_text_without_punctuation_is_cut_at_word_boundaries_not_on_weak_words() -> None:
    text = " ".join(["word"] * 3 + ["and", "the"] + ["word"] * 60)
    clauses = chunk_stream(text)
    assert " ".join(clauses) == text
    assert all(len(c.split()) <= 25 + 3 for c in clauses)
    assert not clauses[0].split()[-1] in ("and", "the")


def test_tokens_split_inside_words_are_never_cut_mid_word() -> None:
    clauses = chunk_stream("Supercalifragilistic expialidocious words are long, really long.", step=2)
    assert clauses[0].startswith("Supercalifragilistic expialidocious")


def test_empty_and_whitespace_input() -> None:
    chunker = ClauseChunker()
    assert chunker.feed("") == []
    assert chunker.feed("   \n  ") == []
    assert chunker.flush() == []


def test_i_at_the_end_is_not_treated_as_an_initial() -> None:
    assert chunk_stream("Nobody knows it better than I. That is the truth, my friend.")[0].endswith("I.")


def test_reset_makes_the_next_clause_short_again() -> None:
    chunker = ClauseChunker()
    chunker.feed("Okay. Then we continue with the rest.")
    chunker.flush()
    chunker.reset()
    assert chunker.feed("Sure. And") == ["Sure."]


# ---- TTS helpers ----------------------------------------------------------------------------

def test_trim_silence_removes_lead_and_tail_but_keeps_margins() -> None:
    silence = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    tone = np.sin(np.linspace(0, 200, SAMPLE_RATE)).astype(np.float32) * 0.5
    trimmed = trim_silence(np.concatenate([silence, tone, silence]))
    assert abs(len(trimmed) - len(tone)) < SAMPLE_RATE * 0.06  # margins only: 15 + 40 ms
    assert trim_silence(np.zeros(1000, dtype=np.float32)).size == 0


@pytest.mark.parametrize(("text", "ms"), [
    ("Done.", 280), ("Really?", 300), ("Wait,", 120), ("Then:", 200), ('He said "ok."', 280),
    ("cut for length", 60),
])
def test_pause_depends_on_final_punctuation(text: str, ms: int) -> None:
    assert len(pause_after(text)) == SAMPLE_RATE * ms // 1000


# ---- resampler ------------------------------------------------------------------------------

def test_stateful_resampler_has_no_seams_between_chunks() -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    whole = StatefulResampler(24_000, 48_000).process(signal)
    resampler = StatefulResampler(24_000, 48_000)
    pieces = np.concatenate([resampler.process(signal[i : i + 960]) for i in range(0, len(signal), 960)])
    n = min(len(whole), len(pieces))
    assert abs(len(pieces) - 48_000) <= 2
    assert np.max(np.abs(whole[:n] - pieces[:n])) < 1e-5  # chunking changes nothing


def test_resampler_down_and_same_rate() -> None:
    signal = np.random.default_rng(0).normal(size=4800).astype(np.float32)
    assert StatefulResampler(24_000, 24_000).process(signal) is not None
    down = StatefulResampler(48_000, 24_000)
    out = np.concatenate([down.process(signal[i : i + 100]) for i in range(0, 4800, 100)])
    assert abs(len(out) - 2400) <= 1


# ---- worker ---------------------------------------------------------------------------------

class FakeTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, chunk: int = 960, chunks_per_clause: int = 5, delay: float = 0.0) -> None:
        self.chunk = chunk
        self.chunks = chunks_per_clause
        self.delay = delay
        self._cancel = threading.Event()

    def synthesize_stream(self, text: str):
        self._cancel.clear()
        for _ in range(self.chunks):
            if self._cancel.is_set():
                return
            time.sleep(self.delay)
            yield np.full(self.chunk, 0.1, dtype=np.float32)

    def cancel(self) -> None:
        self._cancel.set()


class FakeSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def play(self, audio: np.ndarray) -> None:
        self.chunks.append(audio)


def test_worker_records_samples_and_start_of_every_clause() -> None:
    events = []
    sink = FakeSink()
    worker = TTSWorker(FakeTTS(), sink, on_event=lambda name, at: events.append(name))
    for text in ("Okay.", "Let me check that for you,", "it takes a moment."):
        worker.submit(text)
    assert worker.wait_idle(2)

    starts = [c.start_sample for c in worker.clauses]
    assert starts == [0, 4800, 9600]
    assert all(c.samples == 4800 and c.done for c in worker.clauses)
    assert worker.total_samples == sum(len(c) for c in sink.chunks) == 14_400
    assert events.count("tts_first_audio") == 1
    assert worker.heard_text(5000) == "Okay. Let me check that for you,"
    assert worker.heard_text(0) == ""
    worker.close()


def test_worker_cancel_stops_within_one_chunk_and_drops_the_queue() -> None:
    sink = FakeSink()
    worker = TTSWorker(FakeTTS(chunks_per_clause=50, delay=0.005), sink)
    for i in range(5):
        worker.submit(f"clause {i}")
    time.sleep(0.05)
    worker.cancel()
    count = len(sink.chunks)
    time.sleep(0.1)
    assert len(sink.chunks) - count <= 1  # at most the chunk that was in flight
    assert not worker.clauses[-1].done
    worker.new_turn()
    worker.submit("fresh answer")
    assert worker.wait_idle(2)
    assert worker.clauses[0].start_sample == 0 and worker.clauses[0].done
    worker.close()


# ---- player counters ------------------------------------------------------------------------

def pull(player: Player, frames: int = 480) -> np.ndarray:
    out = np.ones((frames, 1), dtype=np.float32)
    player._callback(out, frames, None, None)
    return out[:, 0]


def test_player_counts_gaps_only_inside_an_utterance() -> None:
    player = Player(sample_rate=24_000)
    pull(player)  # idle: not a gap
    player.begin_utterance()
    player.play(np.full(480, 0.1, dtype=np.float32))
    pull(player)
    pull(player)  # queue empty while the utterance is still going...
    assert player.starved_blocks == 0  # ...but only a gap once more audio arrives
    player.play(np.full(480, 0.1, dtype=np.float32))
    assert player.starved_blocks == 1
    pull(player)
    pull(player)  # runs dry at the end of the answer
    player.end_utterance()
    player.play(np.full(480, 0.1, dtype=np.float32))  # next answer: the end was not a gap
    assert player.starved_blocks == 1
