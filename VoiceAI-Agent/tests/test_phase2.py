import wave
from pathlib import Path

import numpy as np
import pytest

from voice.playback import Player
from voice.vad import SileroVAD

SPEECH_WAV = Path(__file__).parent / "data" / "speech_16k.wav"  # 1 s silence, speech, 1 s silence


def load_speech() -> np.ndarray:
    with wave.open(str(SPEECH_WAV), "rb") as file:
        frames = file.readframes(file.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768


def run_vad(vad: SileroVAD, audio: np.ndarray, block: int = 320):
    events = []
    for i in range(0, len(audio), block):
        events += vad.process(audio[i : i + block])
    return events


def test_vad_finds_one_speech_segment() -> None:
    audio = load_speech()
    events = run_vad(SileroVAD(), audio)

    assert [e.kind for e in events] == ["speech_start", "speech_end"]
    assert 0.9 <= events[0].seconds <= 1.3
    assert 2.6 <= events[1].seconds <= 3.1


def test_vad_ignores_silence_and_quiet_noise() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.002, 16_000 * 3).astype(np.float32)
    assert run_vad(SileroVAD(), noise) == []


def test_vad_block_size_does_not_change_result() -> None:
    audio = load_speech()
    small = run_vad(SileroVAD(), audio, block=160)
    large = run_vad(SileroVAD(), audio, block=4000)
    assert small == large


def test_vad_reset_clears_state() -> None:
    vad = SileroVAD()
    run_vad(vad, load_speech()[: 16_000 * 2])  # stop in the middle of speech
    assert vad.speaking
    vad.reset()
    assert not vad.speaking


def pull(player: Player, frames: int = 480) -> np.ndarray:
    out = np.ones((frames, 1), dtype=np.float32)
    player._callback(out, frames, None, None)
    return out[:, 0]


def test_player_plays_queued_audio_across_blocks() -> None:
    player = Player(sample_rate=24_000)
    player.play(np.full(700, 0.5, dtype=np.float32))

    first, second = pull(player), pull(player)

    assert np.all(first == 0.5)
    assert np.all(second[:220] == 0.5) and np.all(second[220:] == 0.0)
    assert player.played_samples == 700
    assert not player.is_playing


def test_player_stop_silences_the_next_block() -> None:
    events = []
    player = Player(sample_rate=24_000, on_event=lambda name, t: events.append(name))
    player.play(np.full(24_000, 0.5, dtype=np.float32))  # 1 s of audio

    pull(player)
    player.stop()
    after_stop = pull(player)

    assert np.all(after_stop == 0.0)
    assert player.played_samples == 480  # only what reached the sound card
    assert events == ["playback_start", "playback_stopped"]


@pytest.mark.parametrize("chunks", [1, 5])
def test_player_counts_played_seconds(chunks: int) -> None:
    player = Player(sample_rate=24_000)
    for _ in range(chunks):
        player.play(np.zeros(2_400, dtype=np.float32))
    for _ in range(chunks * 5):
        pull(player)
    assert player.played_seconds == pytest.approx(chunks * 0.1)
