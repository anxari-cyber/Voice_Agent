import pytest

from pipeline.turn_filter import ignore_reason


@pytest.mark.parametrize("text, speech_s", [
    ("", 1.0), ("Mm-hmm.", 0.8), ("Uh", 0.5), ("Hmm. Hmm.", 1.2), ("um, uh", 0.9),
    ("Open my project.", 0.2),  # real words, but far too short to be real speech
])
def test_noise_and_fillers_are_ignored(text, speech_s) -> None:
    assert ignore_reason(text, speech_s) is not None


@pytest.mark.parametrize("text, speech_s", [
    ("Yes.", 0.4), ("No, the other file.", 0.9), ("Uh, can you talk to me?", 1.5),
    ("Stop.", 0.36), ("What time is it?", 0.8),
])
def test_real_answers_are_kept(text, speech_s) -> None:
    assert ignore_reason(text, speech_s) is None
