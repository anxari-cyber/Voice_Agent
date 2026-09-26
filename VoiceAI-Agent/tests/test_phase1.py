import json

import pytest

from bench.latency_report import load, report
from metrics.latency import LatencyLog
from voice import audio_devices
from voice.audio_devices import AudioDeviceError, resolve_device
from voice.transcriber import _is_cuda_error

HOST_APIS = (
    {"name": "MME", "default_input_device": 0, "default_output_device": 1},
    {"name": "Windows WASAPI", "default_input_device": 2, "default_output_device": 3},
)
DEVICES = (
    {"name": "Headset Mic", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0,
     "default_samplerate": 44_100.0},
    {"name": "Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 44_100.0},
    {"name": "Headset Mic", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0,
     "default_samplerate": 48_000.0},
    {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 48_000.0},
    {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0,
     "default_samplerate": 44_100.0},
)


@pytest.fixture
def fake_devices(monkeypatch):
    monkeypatch.setattr(audio_devices.sd, "query_hostapis", lambda: HOST_APIS)
    monkeypatch.setattr(audio_devices.sd, "query_devices", lambda: DEVICES)


def test_default_microphone_prefers_wasapi(fake_devices) -> None:
    device = resolve_device("input")
    assert device.index == 2
    assert device.host_api == "Windows WASAPI"


def test_named_microphone_is_found_by_substring(fake_devices) -> None:
    assert resolve_device("input", "usb").name == "USB Mic"
    assert resolve_device("input", "headset").index == 2  # WASAPI copy wins
    assert resolve_device("input", "4").index == 4


def test_missing_microphone_falls_back_to_default(fake_devices, capsys) -> None:
    assert resolve_device("input", "AirPods Pro").index == 2
    assert "not found" in capsys.readouterr().out


def test_no_microphone_at_all_gives_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(
        audio_devices.sd, "query_hostapis",
        lambda: ({"name": "MME", "default_input_device": -1, "default_output_device": 1},),
    )
    monkeypatch.setattr(audio_devices.sd, "query_devices", lambda: DEVICES[1:2])
    with pytest.raises(AudioDeviceError, match="No microphone found"):
        resolve_device("input")


def test_cuda_errors_are_recognised() -> None:
    assert _is_cuda_error(RuntimeError("CUDA failed with error out of memory"))
    assert _is_cuda_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert not _is_cuda_error(RuntimeError("Invalid input shape"))


def test_latency_log_and_report(tmp_path) -> None:
    log = LatencyLog(tmp_path / "latency.jsonl", run_id="test")
    log.next_turn()
    log.mark("speech_end", at=10.0)
    log.mark("stt_final", at=10.25)
    log.mark("playback_start", at=10.6)

    records = [json.loads(line) for line in (tmp_path / "latency.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["speech_end", "stt_final", "playback_start"]

    text = report(load(tmp_path / "latency.jsonl", run=None))
    assert "End of speech -> final text (STT)" in text
    assert " 250 " in text  # STT gap in ms
    assert " 600 " in text  # total gap in ms


def test_latency_log_rejects_unknown_event(tmp_path) -> None:
    with pytest.raises(ValueError):
        LatencyLog(tmp_path / "x.jsonl").mark("typo_event")
