"""Automatic mic/speaker selection and hot-plug.

- Follows Windows: uses the current Windows default input/output (VOICEAI_MIC_DEVICE /
  VOICEAI_SPEAKER_DEVICE are optional overrides, empty by default).
- Probes before use: a device is only used once ~200 ms of real audio has flowed. Otherwise the
  next host API (WASAPI -> MME -> DirectSound), then the next device, is tried. WDM-KS is never
  picked automatically.
- Hot-plug: every 2 s it checks that both streams are still alive and whether the Windows default
  changed. If a device disappeared, went to sleep or a new default appeared, it closes the
  streams, re-initialises PortAudio (so new devices show up), picks again and reopens. With no
  mic at all it says "Microphone disconnected, waiting..." and keeps checking.
- Bluetooth: 8/16 kHz hands-free mics are resampled (MicStream). A hands-free mic is paired
  with the matching hands-free speaker endpoint when Windows has one, and a speaker stream that
  stalls during the Headphones <-> Headset profile switch is reopened by the health check.

The audio front-end reads blocks through `read()` exactly as from a MicStream, so a device switch
is invisible to the rest of the pipeline (it only gets `on_mic_change` to reset VAD/STT state).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable

import sounddevice as sd

from voice.audio_devices import (
    AudioDevice,
    AudioDeviceError,
    bluetooth_output_for,
    candidates,
    list_devices,
)
from voice.windows_audio import default_device_name


def reinit_portaudio() -> None:
    """Forget PortAudio's device list and scan again (all streams must be closed)."""
    sd._terminate()
    sd._initialize()


class AudioManager:
    def __init__(self, mic, player, mic_override: str | None = None, speaker_override: str | None = None,
                 log: Callable[[str], None] = print, check_every_s: float = 2.0,
                 devices_fn: Callable[[], list[AudioDevice]] = list_devices,
                 host_apis_fn: Callable[[], list[dict]] = sd.query_hostapis,
                 windows_default: Callable[[str], str | None] = default_device_name,
                 reinit: Callable[[], None] = reinit_portaudio,
                 on_mic_change: Callable[[], None] | None = None) -> None:
        self.mic = mic
        self.player = player
        self.overrides = {"input": (mic_override or "").strip() or None,
                          "output": (speaker_override or "").strip() or None}
        self.log = log
        self.check_every_s = check_every_s
        self.devices_fn = devices_fn
        self.host_apis_fn = host_apis_fn
        self.windows_default = windows_default
        self.reinit = reinit
        self.on_mic_change = on_mic_change or (lambda: None)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_default: dict[str, str | None] = {"input": None, "output": None}
        self._waiting = {"input": False, "output": False}
        self.switches = 0

    # -- lifecycle -----------------------------------------------------------------------------
    def start(self, monitor: bool = True) -> None:
        """Pick devices now (never raises: with no mic it waits), then watch them."""
        self.configure("startup", reinit=False)
        if monitor:
            self.start_monitor()

    def start_monitor(self) -> None:
        """Begin the 2-second health / Windows-default checks (hot-plug)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._monitor, name="audio-devices", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            self.mic.close()
            self.player.close()

    def drain(self) -> None:
        """Forget audio captured so far (e.g. everything heard while the models were loading)."""
        self.mic.drain()

    def read(self, timeout: float | None = None):
        """Next mic block (AudioSource interface). While no mic works, waits and raises Empty."""
        if not self.mic.is_open:
            time.sleep(min(timeout or 0.1, 0.1))
            raise queue.Empty
        return self.mic.read(timeout=timeout)

    # -- watching ----------------------------------------------------------------------------------
    def _monitor(self) -> None:
        while not self._stop.wait(self.check_every_s):
            try:
                self.check_once()
            except Exception as error:  # noqa: BLE001 - never let device trouble end the agent
                self.log(f"[audio] device check failed: {error}")

    def check_once(self) -> bool:
        """One health/default check; reconfigures if needed. Returns True if it did."""
        reasons = []
        defaults = {d: self.windows_default(d) for d in ("input", "output")}
        for direction, stream, name in (("input", self.mic, "microphone"), ("output", self.player, "speaker")):
            if not stream.alive():
                reasons.append(f"{name} stopped" if stream.is_open else f"waiting for a {name}")
            elif (not self.overrides[direction] and defaults[direction]
                  and defaults[direction] != self._seen_default[direction]):
                reasons.append(f"Windows default {name} changed to {defaults[direction]}")
        if not reasons:
            return False
        self.configure("; ".join(reasons), reinit=True, defaults=defaults)
        return True

    # -- (re)configuring ---------------------------------------------------------------------------
    def configure(self, reason: str, reinit: bool = True, defaults: dict[str, str | None] | None = None) -> None:
        with self._lock:
            old_mic = self.mic.device if self.mic.is_open else None
            old_speaker = self.player.device if self.player.is_open else None
            self.mic.close()
            self.player.close()
            if reinit:
                self.reinit()
            if defaults is None:
                defaults = {d: self.windows_default(d) for d in ("input", "output")}
            self._seen_default = dict(defaults)
            devices = self.devices_fn()
            host_apis = self.host_apis_fn()

            mic = self._select("input", self.mic, devices, host_apis, defaults["input"])
            speaker_options = candidates("output", self.overrides["output"], defaults["output"], devices, host_apis)
            paired = bluetooth_output_for(mic, speaker_options) if mic and not self.overrides["output"] else None
            if paired:  # hands-free mic: use its hands-free speaker endpoint first
                speaker_options = [paired, *[d for d in speaker_options if d.index != paired.index]]
            speaker = self._select("output", self.player, devices, host_apis, defaults["output"], speaker_options)
            self.player.set_available(speaker is not None)

            self._report("input", "Microphone", mic, old_mic, reason)
            self._report("output", "Speaker", speaker, old_speaker, reason)
            if mic is not None and (old_mic is None or old_mic.index != mic.index or reinit):
                self.switches += 1
                self.on_mic_change()

    def _select(self, direction: str, stream, devices, host_apis, windows_default,
                options: list[AudioDevice] | None = None) -> AudioDevice | None:
        options = options if options is not None else candidates(
            direction, self.overrides[direction], windows_default, devices, host_apis)
        silent: AudioDevice | None = None
        skipped: list[str] = []
        for device in options:
            try:
                stream.open(device)
            except AudioDeviceError:
                skipped.append(f"{device.label}: can't be opened")
                continue
            ok, why = stream.probe()
            if ok:
                if skipped:  # not the first choice: say why, so device trouble is visible
                    self.log("[audio] skipped " + "; ".join(skipped))
                return device
            skipped.append(f"{device.label}: {why}")
            stream.close()
            if why.startswith("silent") and silent is None:
                silent = device  # works but only zeros: use it if nothing better answers
        if silent is not None:
            try:
                stream.open(silent)
                self.log(f"[audio] {silent.label} only sends silence (asleep or muted?). Using it anyway.")
                return silent
            except AudioDeviceError:
                pass
        return None

    def _report(self, direction: str, noun: str, device: AudioDevice | None, old: AudioDevice | None,
                reason: str) -> None:
        if device is None:
            if not self._waiting[direction]:
                self._waiting[direction] = True
                self.log(f"[audio] {noun} disconnected, waiting... (connect one; checking every "
                         f"{self.check_every_s:.0f} s)")
            return
        was_waiting = self._waiting[direction]
        self._waiting[direction] = False
        if old is not None and old.index == device.index and not was_waiting:
            return  # same device as before: nothing to tell
        verb = "connected" if was_waiting else ("switched to" if old is not None else "using")
        warning = getattr(self.mic, "warning", "") if direction == "input" else ""
        self.log(f"[audio] {noun} {verb}: {device.label}" + (f"\n        {warning}" if warning else ""))
