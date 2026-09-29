"""Ask Windows which audio devices are the current defaults (IMMDeviceEnumerator via pycaw).

PortAudio only learns the device list when it initialises, so it can't tell that a headset was
just connected. Windows can: this returns the friendly names of the current default mic and
speaker ("Headset (AirPods Pro)", "Headphones (AirPods Pro)"), which voice/audio_devices.py
matches against PortAudio's entries. Returns None when it can't find out (not Windows, no
device, COM error); the agent then falls back to PortAudio's own defaults.
"""

from __future__ import annotations

import threading

_initialised = threading.local()


def _com_ready() -> bool:
    if getattr(_initialised, "ok", False):
        return True
    try:
        import comtypes

        comtypes.CoInitialize()  # once per thread
        _initialised.ok = True
    except Exception:  # noqa: BLE001 - not on Windows / no comtypes: no Windows defaults
        _initialised.ok = False
    return _initialised.ok


def default_device_name(direction: str) -> str | None:
    """Friendly name of the Windows default ('console' role) input or output device."""
    if not _com_ready():
        return None
    try:
        from pycaw.constants import EDataFlow, ERole
        from pycaw.utils import AudioUtilities

        flow = EDataFlow.eCapture.value if direction == "input" else EDataFlow.eRender.value
        endpoint = AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(flow, ERole.eConsole.value)
        return AudioUtilities.CreateDevice(endpoint).FriendlyName
    except Exception:  # noqa: BLE001 - no default device (e.g. nothing connected) or COM error
        return None


def default_device_names() -> dict[str, str | None]:
    return {"input": default_device_name("input"), "output": default_device_name("output")}
