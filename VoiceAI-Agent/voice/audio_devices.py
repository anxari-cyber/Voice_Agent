"""Find microphones and speakers automatically: no device name needs to be configured.

Order of preference for each direction (input / output):
1. The optional override (VOICEAI_MIC_DEVICE / VOICEAI_SPEAKER_DEVICE): name substring or index.
2. The CURRENT Windows default device (asked from Windows, see voice/windows_audio.py), so when a
   headset is connected and Windows makes it the default, the agent uses it.
3. PortAudio's own defaults, then every other device.
Windows lists one physical device once per host API. They are tried in the order
WASAPI -> MME -> DirectSound. WDM-KS is never picked automatically (its entries are often stale
driver endpoints that fail to open, e.g. a sleeping Bluetooth headset); only an explicit override
can select it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sounddevice as sd

Direction = Literal["input", "output"]
PREFERRED_HOST_APIS = ("Windows WASAPI", "MME", "Windows DirectSound")
MANUAL_ONLY_HOST_APIS = ("Windows WDM-KS",)
MME_NAME_LENGTH = 31  # MME truncates device names to 31 characters


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

    @property
    def label(self) -> str:
        return f"{self.name.splitlines()[0]} ({self.host_api.replace('Windows ', '')})"


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


def same_device(device: AudioDevice, windows_name: str) -> bool:
    """Does this PortAudio entry belong to the Windows endpoint called `windows_name`?"""
    name = device.name.splitlines()[0].strip()
    if name == windows_name:
        return True
    # MME cuts names to 31 characters: "Headphones (High Definition Aud"
    return device.host_api == "MME" and len(name) >= MME_NAME_LENGTH - 1 and windows_name.startswith(name)


def _override_matches(devices: list[AudioDevice], wanted: str) -> list[AudioDevice]:
    wanted = wanted.strip()
    if wanted.isdigit():
        return [d for d in devices if d.index == int(wanted)]
    return sorted((d for d in devices if wanted.lower() in d.name.lower()), key=_host_api_rank)


def candidates(direction: Direction, wanted: str | int | None = None, windows_default: str | None = None,
               devices: list[AudioDevice] | None = None, host_apis: list[dict] | None = None) -> list[AudioDevice]:
    """Every usable device for `direction`, best first (see the module docstring)."""
    devices = list_devices() if devices is None else devices
    host_apis = sd.query_hostapis() if host_apis is None else host_apis
    usable = [d for d in devices if d.supports(direction)]
    automatic = sorted((d for d in usable if d.host_api not in MANUAL_ONLY_HOST_APIS),
                       key=lambda d: (_host_api_rank(d), d.index))
    ordered: list[AudioDevice] = []
    if wanted is not None and str(wanted).strip():
        ordered += _override_matches(usable, str(wanted))
    if windows_default:
        ordered += [d for d in automatic if same_device(d, windows_default)]
    key = "default_input_device" if direction == "input" else "default_output_device"
    defaults = {api[key] for api in host_apis if api[key] >= 0}
    ordered += [d for d in automatic if d.index in defaults]
    ordered += automatic
    seen: set[int] = set()
    return [d for d in ordered if not (d.index in seen or seen.add(d.index))]


def resolve_device(direction: Direction, wanted: str | int | None = None,
                   windows_default: str | None = None) -> AudioDevice:
    """The best candidate (no probing). Warns when an override matches nothing."""
    options = candidates(direction, wanted, windows_default)
    if wanted is not None and str(wanted).strip() and not _override_matches(
            [d for d in list_devices() if d.supports(direction)], str(wanted)):
        print(f"Warning: {direction} device '{wanted}' not found, using the default device.")
    if not options:
        kind = "microphone" if direction == "input" else "speaker"
        raise AudioDeviceError(
            f"No {kind} found. Connect one, or run 'voiceai --list-devices' to see devices."
        )
    return options[0]


def stream_settings(device: AudioDevice) -> sd.WasapiSettings | None:
    """WASAPI shared mode only accepts the device's native rate unless auto-convert is on."""
    if device.host_api == "Windows WASAPI":
        return sd.WasapiSettings(auto_convert=True)
    return None


def bluetooth_output_for(mic: AudioDevice, outputs: list[AudioDevice]) -> AudioDevice | None:
    """A Bluetooth hands-free mic ("Headset (X)") works best with the matching hands-free
    speaker endpoint when Windows has one ("Headset (X)" as an output). Windows 11 often has a
    single unified "Headphones (X)" endpoint instead, which switches profile by itself."""
    name = mic.name.splitlines()[0].strip()
    if not name.startswith("Headset ("):
        return None
    for device in outputs:
        if device.name.splitlines()[0].strip() == name and device.supports("output"):
            return device
    return None


def format_devices(devices: list[AudioDevice], windows_defaults: dict[str, str | None] | None = None) -> str:
    lines = []
    windows_defaults = windows_defaults or {}
    for direction in ("input", "output"):
        lines.append("Microphones:" if direction == "input" else "Speakers:")
        default = windows_defaults.get(direction)
        if default:
            lines.append(f"  Windows default: {default}")
        ranked = candidates(direction, windows_default=default, devices=devices)
        manual = [d for d in devices if d.supports(direction) and d.host_api in MANUAL_ONLY_HOST_APIS]
        if not ranked and not manual:
            lines.append("  (none found)")
        for position, d in enumerate(ranked):
            marker = "*" if position == 0 else " "
            lines.append(f" {marker}[{d.index}] {d.label}")
        for d in manual:
            lines.append(f"  [{d.index}] {d.label}  (only with VOICEAI_{'MIC' if direction == 'input' else 'SPEAKER'}_DEVICE={d.index})")
    lines.append("* = what the agent tries first (it checks that audio really flows before using it)")
    return "\n".join(lines)
