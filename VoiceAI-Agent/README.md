# VoiceAI-Agent

A Windows-first, local voice-controlled software agent.

## MVP boundaries

- CLI first
- Push-to-talk voice input
- Selected project folder only
- Read operations automatic
- File edits and terminal commands require confirmation
- Git inspection only
- English only
- Offline by default
- One command produces one task and result

## Architecture

- faster-whisper: speech-to-text
- Google Gemini API with configurable `gemini-3.8-flash`
- OpenHands SDK: agent foundation
- Piper: text-to-speech
- Custom progress and permission components

## Setup

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env`, replace `VOICEAI_GEMINI_API_KEY` with your Google AI Studio key, then set `VOICEAI_PROJECT_ROOT` to the selected test project when needed.

Run the scaffold:

```powershell
python -m app.main
```

Run tests:

```powershell
python -m pytest
```

Whisper capture uses the selected microphone, waits for speech, and stops after configurable silence.
Gemini integration is active. OpenHands integration and Piper setup will be added and tested as separate phases.

Run natural speech transcription:

```powershell
.\.venv\Scripts\python.exe -m app.transcribe
```

Use `--max-seconds`, `--silence-seconds`, `--speech-timeout`, and `--threshold` to tune recording limits. The default silence delay is 0.7 seconds. The adaptive threshold is recommended; only use `--threshold` when troubleshooting.
