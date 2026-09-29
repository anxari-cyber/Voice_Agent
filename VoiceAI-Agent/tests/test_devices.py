"""Automatic device selection and hot-plug, with fake device lists (no real audio devices)."""

import queue

import pytest

from voice.audio_devices import (
    AudioDevice,
    AudioDeviceError,
    bluetooth_output_for,
    candidates,
    same_device,
)
from voice.device_manager import AudioManager

MME, DS, WASAPI, WDMKS = "MME", "Windows DirectSound", "Windows WASAPI", "Windows WDM-KS"
HOST_APIS_ORDER = [MME, DS, WASAPI, WDMKS]


def dev(index: int, name: str, api: str, inputs: int = 0, outputs: int = 0, rate: float = 48_000.0) -> AudioDevice:
    return AudioDevice(index, name, api, inputs, outputs, rate)


def host_apis(default_in: dict[str, int] | None = None, default_out: dict[str, int] | None = None) -> list[dict]:
    default_in, default_out = default_in or {}, default_out or {}
    return [{"name": api, "default_input_device": default_in.get(api, -1),
             "default_output_device": default_out.get(api, -1)} for api in HOST_APIS_ORDER]


# ---- ranking ---------------------------------------------------------------------------------

def test_windows_default_first_then_wasapi_mme_directsound() -> None:
    devices = [
        dev(0, "Microphone (USB Audio)", MME, inputs=1),
        dev(1, "Headset (AirPods Pro)", MME, inputs=1),
        dev(2, "Headset (AirPods Pro)", DS, inputs=1),
        dev(3, "Headset (AirPods Pro)", WASAPI, inputs=1, rate=16_000),
        dev(4, "Microphone (USB Audio)", WASAPI, inputs=1),
    ]
    ranked = candidates("input", windows_default="Headset (AirPods Pro)", devices=devices,
                        host_apis=host_apis(default_in={WASAPI: 4}))
    assert [d.index for d in ranked[:3]] == [3, 1, 2]  # the Windows default, WASAPI -> MME -> DS
    assert ranked[3].index == 4  # then PortAudio's default


def test_wdm_ks_is_never_picked_automatically_but_an_override_can() -> None:
    devices = [dev(9, "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0", WDMKS, inputs=1, rate=8_000)]
    apis = host_apis(default_in={WDMKS: 9})
    assert candidates("input", devices=devices, host_apis=apis) == []
    assert [d.index for d in candidates("input", wanted="9", devices=devices, host_apis=apis)] == [9]


def test_mme_truncated_names_still_match_the_windows_default() -> None:
    mme = dev(1, "Headphones (High Definition Aud", MME, outputs=2)
    assert same_device(mme, "Headphones (High Definition Audio Device)")
    assert not same_device(dev(2, "Headphones (High", MME, outputs=2), "Headphones (High Definition Audio Device)")


def test_bluetooth_hands_free_mic_pairs_with_its_hands_free_speaker() -> None:
    mic = dev(3, "Headset (TWS Buds)", WASAPI, inputs=1, rate=16_000)
    outputs = [dev(5, "Headphones (TWS Buds)", WASAPI, outputs=2), dev(6, "Headset (TWS Buds)", WASAPI, outputs=1)]
    assert bluetooth_output_for(mic, outputs).index == 6
    assert bluetooth_output_for(mic, outputs[:1]) is None  # Windows 11 unified endpoint: keep default
    assert bluetooth_output_for(dev(7, "Microphone (USB)", WASAPI, inputs=1), outputs) is None


# ---- manager with fake streams ---------------------------------------------------------------

class FakeWorld:
    """Pretend audio system: a device list that can change, a Windows default, dead devices."""

    def __init__(self, devices: list[AudioDevice], defaults: dict[str, str | None]) -> None:
        self.devices = list(devices)
        self.defaults = dict(defaults)
        self.dead: set[int] = set()  # opens, but no audio ever arrives
        self.broken: set[int] = set()  # fails to open
        self.silent: set[int] = set()  # opens, audio flows, only zeros
        self.unplugged: set[int] = set()  # was working, now its callbacks stopped
        self.reinits = 0

    def reinit(self) -> None:
        self.reinits += 1


class FakeStream:
    def __init__(self, world: FakeWorld) -> None:
        self.world = world
        self.device = None
        self.is_open = False
        self.warning = ""
        self.available = True
        self.opened: list[int] = []

    def open(self, device: AudioDevice) -> None:
        self.close()
        present = {d.index for d in self.world.devices}
        if device.index in self.world.broken or device.index not in present:
            raise AudioDeviceError(f"cannot open {device.label}")
        self.device, self.is_open = device, True
        self.opened.append(device.index)

    def close(self) -> None:
        self.is_open = False

    def probe(self):
        if self.device.index in self.world.dead:
            return False, "no audio arrived within 3 s"
        if self.device.index in self.world.silent:
            return False, "silent (only zeros)"
        return True, "ok"

    def alive(self) -> bool:
        return self.is_open and self.device.index not in self.world.unplugged

    def set_available(self, available: bool) -> None:
        self.available = available

    def read(self, timeout=None):
        return 0.0, None


def manager(world: FakeWorld, **kwargs) -> tuple[AudioManager, FakeStream, FakeStream, list[str]]:
    logs: list[str] = []
    mic, speaker = FakeStream(world), FakeStream(world)
    mgr = AudioManager(mic, speaker, log=logs.append, devices_fn=lambda: list(world.devices),
                       host_apis_fn=lambda: host_apis(), windows_default=lambda d: world.defaults.get(d),
                       reinit=world.reinit, **kwargs)
    return mgr, mic, speaker, logs


LAPTOP_MIC = [dev(0, "Microphone (Realtek)", MME, inputs=1), dev(1, "Microphone (Realtek)", WASAPI, inputs=1)]
SPEAKERS = [dev(10, "Speakers (Realtek)", WASAPI, outputs=2)]
HEADSET = [dev(20, "Headset (AirPods Pro)", WASAPI, inputs=1, rate=16_000),
           dev(21, "Headset (AirPods Pro)", MME, inputs=1, rate=16_000),
           dev(22, "Headphones (AirPods Pro)", WASAPI, outputs=2)]


def test_only_wdm_ks_devices_means_waiting_not_crashing() -> None:
    world = FakeWorld([dev(9, "Headset (bthhfenum Hands-Free)", WDMKS, inputs=1, rate=8_000), *SPEAKERS],
                      {"input": None, "output": "Speakers (Realtek)"})
    mgr, mic, speaker, logs = manager(world)
    mgr.start(monitor=False)
    assert not mic.is_open and speaker.is_open
    assert any("Microphone disconnected, waiting" in line for line in logs)
    with pytest.raises(queue.Empty):
        mgr.read(timeout=0.01)  # the pipeline just sees "no audio yet"


def test_a_dead_host_api_entry_falls_through_to_the_next_one() -> None:
    world = FakeWorld([*HEADSET, *LAPTOP_MIC, *SPEAKERS],
                      {"input": "Headset (AirPods Pro)", "output": "Speakers (Realtek)"})
    world.dead.add(20)  # WASAPI entry opens but never delivers audio
    mgr, mic, _, logs = manager(world)
    mgr.start(monitor=False)
    assert mic.opened[:2] == [20, 21]  # WASAPI tried first, then the same headset on MME
    assert mic.device.index == 21
    assert any("Microphone using: Headset (AirPods Pro) (MME)" in line for line in logs)
    assert any("skipped Headset (AirPods Pro) (WASAPI): no audio arrived" in line for line in logs)


def test_a_silent_device_is_only_used_when_nothing_else_answers() -> None:
    world = FakeWorld([*LAPTOP_MIC, *SPEAKERS], {"input": "Microphone (Realtek)", "output": "Speakers (Realtek)"})
    world.silent.update({0, 1})
    mgr, mic, _, logs = manager(world)
    mgr.start(monitor=False)
    assert mic.is_open and mic.device.index == 1
    assert any("only sends silence" in line for line in logs)


def test_unplug_then_replug_switches_away_and_back() -> None:
    world = FakeWorld([*HEADSET, *LAPTOP_MIC, *SPEAKERS],
                      {"input": "Headset (AirPods Pro)", "output": "Headphones (AirPods Pro)"})
    mic_changes = []
    mgr, mic, speaker, logs = manager(world, on_mic_change=lambda: mic_changes.append(1))
    mgr.start(monitor=False)
    assert (mic.device.index, speaker.device.index) == (20, 22)
    assert mgr.check_once() is False  # healthy: nothing to do

    # Unplug the headset: its streams stop, it disappears, Windows falls back to the laptop.
    world.unplugged.update({20, 22})
    world.devices = [*LAPTOP_MIC, *SPEAKERS]
    world.defaults = {"input": "Microphone (Realtek)", "output": "Speakers (Realtek)"}
    assert mgr.check_once() is True
    assert (mic.device.index, speaker.device.index) == (1, 10)
    assert world.reinits == 1  # PortAudio re-scanned
    assert any("Microphone switched to: Microphone (Realtek) (WASAPI)" in line for line in logs)

    # Replug: Windows makes the headset the default again -> the agent follows.
    world.unplugged.clear()
    world.devices = [*HEADSET, *LAPTOP_MIC, *SPEAKERS]
    world.defaults = {"input": "Headset (AirPods Pro)", "output": "Headphones (AirPods Pro)"}
    assert mgr.check_once() is True
    assert (mic.device.index, speaker.device.index) == (20, 22)
    assert len(mic_changes) == 3  # startup, unplug, replug: VAD/STT reset each time


def test_no_mic_at_all_then_one_is_connected() -> None:
    world = FakeWorld([*SPEAKERS], {"input": None, "output": "Speakers (Realtek)"})
    mgr, mic, _, logs = manager(world)
    mgr.start(monitor=False)
    assert not mic.is_open
    assert mgr.check_once() is True  # still waiting: re-scans
    assert logs.count("[audio] Microphone disconnected, waiting... (connect one; checking every 2 s)") == 1

    world.devices = [*HEADSET, *SPEAKERS]
    world.defaults["input"] = "Headset (AirPods Pro)"
    mgr.check_once()
    assert mic.is_open and mic.device.index == 20
    assert any("Microphone connected: Headset (AirPods Pro) (WASAPI)" in line for line in logs)


def test_speaker_lost_mid_answer_makes_playback_unavailable_instead_of_hanging() -> None:
    world = FakeWorld([*LAPTOP_MIC, *SPEAKERS], {"input": "Microphone (Realtek)", "output": "Speakers (Realtek)"})
    mgr, _, speaker, logs = manager(world)
    mgr.start(monitor=False)
    world.unplugged.add(10)
    world.devices = [*LAPTOP_MIC]
    world.defaults["output"] = None
    mgr.check_once()
    assert not speaker.is_open and speaker.available is False
    assert any("Speaker disconnected, waiting" in line for line in logs)


def test_a_failed_default_is_not_retried_every_check() -> None:
    world = FakeWorld([*HEADSET, *LAPTOP_MIC, *SPEAKERS],
                      {"input": "Headset (AirPods Pro)", "output": "Speakers (Realtek)"})
    world.dead.update({20, 21})  # the headset is the Windows default but never delivers audio
    mgr, mic, _, _ = manager(world)
    mgr.start(monitor=False)
    assert mic.device.index == 1  # fell back to the laptop mic
    assert mgr.check_once() is False  # no ping-pong: the default did not change since last time


def test_override_is_respected_and_default_changes_are_ignored_for_it() -> None:
    world = FakeWorld([*HEADSET, *LAPTOP_MIC, *SPEAKERS],
                      {"input": "Headset (AirPods Pro)", "output": "Speakers (Realtek)"})
    mgr, mic, _, _ = manager(world, mic_override="Realtek")
    mgr.start(monitor=False)
    assert mic.device.index == 1
    world.defaults["input"] = "Something Else"
    assert mgr.check_once() is False
