"""Can you hear the agent's voice? A short, numbered speaker test.

Usage (in the VoiceAI-Agent folder):  .venv\\Scripts\\python -m bench.speaker_check

Test 1 speaks with the microphone CLOSED. Test 2 speaks with the microphone OPEN, exactly like the
agent runs. With Bluetooth headsets that difference matters: while the mic is open the headset
is in hands-free ("Headset") mode, and on some PCs sound sent to the "Headphones" endpoint is
then silently dropped. If there are other speakers, test 3 tries each of them.
Answer y/n after each test.
"""

from __future__ import annotations

import time

import numpy as np

from config.settings import load_settings
from voice.audio_devices import candidates, list_devices
from voice.audio_io import MicStream
from voice.device_manager import AudioManager
from voice.playback import Player
from voice.tts import kokoro_from_settings
from voice.windows_audio import default_device_names


def say(tts, player: Player, text: str) -> float:
    audio = np.concatenate(list(tts.synthesize_stream(text)))
    player.reset_counter()
    player.play(audio)
    started = time.perf_counter()
    while player.is_playing and time.perf_counter() - started < 15:
        time.sleep(0.02)
    time.sleep(0.3)
    return float(np.abs(audio).max())


def ask(question: str) -> bool:
    return input(f"   {question} [y/n] ").strip().lower().startswith("y")


def main() -> None:
    settings = load_settings()
    print("Loading the voice...")
    tts = kokoro_from_settings(settings)
    tts.warm_up()
    defaults = default_device_names()
    print(f"Windows default mic: {defaults['input']}\nWindows default speaker: {defaults['output']}")

    mic, player = MicStream(), Player(sample_rate=24_000)
    manager = AudioManager(mic, player, mic_override=settings.mic_device,
                           speaker_override=settings.speaker_device, log=print)
    mic_device = None
    # Test 1: speaker only, mic closed.
    manager.configure("speaker check", reinit=False)
    mic_device = mic.device if mic.is_open else None
    mic.close()
    if not player.is_open:
        raise SystemExit("No working speaker found. Check the headset / Windows sound settings.")
    print(f"\nTest 1: {player.device.label}, microphone CLOSED")
    peak = say(tts, player, "Speaker test one. If you can hear this, the voice works.")
    print(f"   (voice level peak {peak:.2f}; played {player.played_samples} samples)")
    heard_1 = ask("Did you hear 'speaker test one'?")

    heard_2 = None
    if mic_device is not None:
        mic.open(mic_device)
        mic.probe()
        print(f"\nTest 2: {player.device.label}, microphone OPEN ({mic_device.label}), like the agent")
        say(tts, player, "Speaker test two, with the microphone open.")
        heard_2 = ask("Did you hear 'speaker test two'?")
    else:
        print("\n(No microphone connected, so test 2 is skipped.)")

    if heard_1 and heard_2 is not False:
        print("\nThe voice works. If the agent was silent before, tell me what was different then.")
    else:
        others = [d for d in candidates("output", devices=list_devices()) if d.index != player.device.index]
        print(f"\nTest 3: trying the other {len(others)} speaker entries one by one.")
        for number, device in enumerate(others, 1):
            try:
                player.open(device)
            except Exception as error:  # noqa: BLE001
                print(f"   {device.label}: can't open ({error})")
                continue
            print(f"   {number}. {device.label}")
            say(tts, player, f"Speaker number {number}.")
            if ask(f"Did you hear 'speaker number {number}'?"):
                print(f"\n=> This one works: {device.label}. Please send me this result.")
                break
    mic.close()
    player.close()
    print("\nDone. Please send me the answers (and the lines above).")


if __name__ == "__main__":
    main()
