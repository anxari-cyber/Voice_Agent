@echo off
rem Start the local voice agent. Double-click this file, or run  .\talk  in a terminal here.
rem Press Ctrl+C to quit.
"%~dp0VoiceAI-Agent\.venv\Scripts\voiceai.exe" run %*
if errorlevel 1 pause
