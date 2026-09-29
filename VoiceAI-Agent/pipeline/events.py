"""Typed events passed from the audio front-end (and pipeline stages) to the orchestrator.

All times are time.perf_counter() values. `at` is when the thing really happened (e.g. the end of
speech inside an audio block), not when the event was noticed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechStart:
    at: float


@dataclass(frozen=True)
class SpeechEnd:
    at: float  # the user stopped speaking (VAD, before the join window)


@dataclass(frozen=True)
class PartialText:
    text: str


@dataclass(frozen=True)
class FinalText:
    """Transcript decoded right after the speech ended; may still be reopened."""

    text: str
    at: float  # when the transcript was ready
    decode_ms: float
    mode: str  # "reused" | "tail" | "window" | "full"


@dataclass(frozen=True)
class TurnReopen:
    """The user kept talking inside the join window: same turn continues."""

    at: float


@dataclass(frozen=True)
class TurnCommit:
    """The join window passed in silence: the turn is final and the agent may answer."""

    text: str
    speech_end: float
    final_ready: float
    at: float


@dataclass(frozen=True)
class LLMDelta:
    text: str


@dataclass(frozen=True)
class Clause:
    index: int
    text: str


@dataclass(frozen=True)
class AudioChunk:
    clause: int
    samples: int


@dataclass(frozen=True)
class BargeIn:
    at: float
