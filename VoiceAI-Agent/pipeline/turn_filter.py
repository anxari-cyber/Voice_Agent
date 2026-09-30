"""Decide whether a heard turn is worth answering, or was noise / a filler sound.

Background noise, breathing, a cough or someone else in the room can trigger the VAD, and the
speech recogniser then makes up short words from it ("Mm-hmm.", "Yeah.", "Uh"). Answering those
makes the agent talk by itself. A turn is ignored when:
- there was less than `min_speech_s` of detected speech in total, or
- after removing filler sounds, no word is left.
"""

from __future__ import annotations

import re

FILLERS = {
    "uh", "um", "umm", "uhm", "hmm", "hm", "mm", "mmm", "mhm", "mmhmm", "mm-hmm", "uh-huh",
    "ah", "oh", "eh", "er", "erm", "huh", "ha", "hah",
}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9'-]+", text.lower()) if w.strip("'-")]


def ignore_reason(text: str, speech_s: float, min_speech_s: float = 0.35) -> str | None:
    """None if the turn should be answered, otherwise a short reason to show the user."""
    if not text.strip():
        return "no words recognised"
    if speech_s < min_speech_s:
        return f"too short ({speech_s * 1000:.0f} ms of speech)"
    if all(w.strip("-") in FILLERS or w in FILLERS for w in words(text)):
        return "only filler sounds"
    return None
