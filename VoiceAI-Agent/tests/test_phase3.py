import time

import numpy as np
import pytest

from voice.stt import StreamingSTT, create_engine


class FakeEngine:
    name = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        time.sleep(self.delay)
        self.calls.append(len(audio))
        return f"{len(audio)} samples"


def block(ms: int = 20) -> np.ndarray:
    return np.zeros(16 * ms, dtype=np.float32)


def test_final_transcript_covers_pre_roll_and_all_blocks() -> None:
    stt = StreamingSTT(FakeEngine(), partial_every_ms=10_000)
    stt.start(pre_roll=block(300))
    for _ in range(10):
        stt.feed(block())
    assert stt.finish() == f"{16 * 500} samples"
    assert not stt.active


def test_partials_run_in_background_while_feeding() -> None:
    engine = FakeEngine(delay=0.01)
    stt = StreamingSTT(engine, partial_every_ms=0)
    stt.start()
    for _ in range(5):
        stt.feed(block())
        time.sleep(0.02)
    time.sleep(0.05)
    assert stt.partial_text.endswith("samples")
    assert len(engine.calls) >= 2  # several partial decodes happened
    stt.finish()
    stt.close()


def test_feed_is_ignored_when_not_active() -> None:
    engine = FakeEngine()
    stt = StreamingSTT(engine)
    stt.feed(block())
    assert stt.duration == 0
    assert engine.calls == []


def test_cancel_drops_audio() -> None:
    stt = StreamingSTT(FakeEngine(), partial_every_ms=10_000)
    stt.start(pre_roll=block())
    stt.cancel()
    assert not stt.active
    assert stt.duration == 0


def test_unknown_engine_name() -> None:
    with pytest.raises(ValueError, match="Unknown STT engine"):
        create_engine("nope")


def test_resume_joins_a_short_pause_into_the_same_utterance() -> None:
    stt = StreamingSTT(FakeEngine(), partial_every_ms=10_000)
    stt.start(pre_roll=block(100))
    stt.feed(block(400))
    stt.finish()  # tentative final after the pause starts
    stt.resume(gap=block(300))  # user kept talking: pause audio stays in order
    stt.feed(block(200))
    assert stt.finish() == f"{16 * 1000} samples"
