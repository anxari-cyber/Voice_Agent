from __future__ import annotations

import argparse

from app import transcribe
from voice.audio_devices import AudioDeviceError, format_devices, list_devices


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceai", description="Local voice AI agent")
    parser.add_argument(
        "--list-devices", action="store_true", help="Show available microphones and speakers"
    )
    commands = parser.add_subparsers(dest="command")
    run_parser = commands.add_parser("run", help="Listen to one voice command and answer it")
    transcribe.add_arguments(run_parser)
    args = parser.parse_args()

    if args.list_devices:
        print(format_devices(list_devices()))
    elif args.command == "run":
        try:
            transcribe.run(args)
        except AudioDeviceError as error:
            raise SystemExit(str(error)) from None
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
