"""Find microphones and speakers without hardcoding a device name.

Resolution order for each direction (input / output):
1. The device the user asked for (name substring or index), if it exists.
2. Otherwise the Windows default device, taken from WASAPI when possible.
Windows lists one physical device once per host API (MME, DirectSound, WASAPI, WDM-KS);
WASAPI has the lowest latency, so it is preferred whenever a name matches several entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sounddevice as sd

Direction = Literal["input", "output"]
PREFERRED_HOST_APIS = ("Windows WASAPI", "Windows DirectSound", "MME")


class AudioDeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    def supports(self, direction: Direction) -> bool:
        channels = self.max_input_channels if direction == "input" else self.max_output_channels
        return channels > 0


def list_devices() -> list[AudioDevice]:
    host_apis = sd.query_hostapis()
    return [
        AudioDevice(
            index=index,
            name=device["name"],
            host_api=host_apis[device["hostapi"]]["name"],
            max_input_channels=device["max_input_channels"],
            max_output_channels=device["max_output_channels"],
            default_samplerate=device["default_samplerate"],
        )
        for index, device in enumerate(sd.query_devices())
    ]


def _host_api_rank(device: AudioDevice) -> int:
    if device.host_api in PREFERRED_HOST_APIS:
        return PREFERRED_HOST_APIS.index(device.host_api)
    return len(PREFERRED_HOST_APIS)


def _default_device(devices: list[AudioDevice], direction: Direction) -> AudioDevice | None:
    key = "default_input_device" if direction == "input" else "default_output_device"
    defaults = [api[key] for api in sd.query_hostapis() if api[key] >= 0]
    candidates = [d for d in devices if d.index in defaults and d.supports(direction)]
    if candidates:
        return min(candidates, key=_host_api_rank)
    return None


def _match(devices: list[AudioDevice], direction: Direction, wanted: str) -> AudioDevice | None:
    usable = [d for d in devices if d.supports(direction)]
    if wanted.strip().isdigit():
        index = int(wanted)
        return next((d for d in usable if d.index == index), None)
    matches = [d for d in usable if wanted.lower() in d.name.lower()]
    return min(matches, key=_host_api_rank) if matches else None


def resolve_device(direction: Direction, wanted: str | int | None = None) -> AudioDevice:
    """Return the device to use, falling back to the system default with a warning."""
    devices = list_devices()
    if wanted is not None and str(wanted).strip():
        device = _match(devices, direction, str(wanted))
        if device is not None:
            return device
        print(f"Warning: {direction} device '{wanted}' not found, using the default device.")
    device = _default_device(devices, direction)
    if device is None:
        kind = "microphone" if direction == "input" else "speaker"
        raise AudioDeviceError(
            f"No {kind} found. Connect one, or run 'voiceai --list-devices' to see devices."
        )
    return device


def stream_settings(device: AudioDevice) -> sd.WasapiSettings | None:
    """WASAPI shared mode only accepts the device's native rate unless auto-convert is on."""
    if device.host_api == "Windows WASAPI":
        return sd.WasapiSettings(auto_convert=True)
    return None


def format_devices(devices: list[AudioDevice]) -> str:
    lines = []
    for direction in ("input", "output"):
        lines.append("Microphones:" if direction == "input" else "Speakers:")
        usable = sorted(
            (d for d in devices if d.supports(direction)), key=lambda d: (_host_api_rank(d), d.index)
        )
        if not usable:
            lines.append("  (none found)")
        for d in usable:
            lines.append(f"  [{d.index}] {d.name} ({d.host_api})")
    return "\n".join(lines)
