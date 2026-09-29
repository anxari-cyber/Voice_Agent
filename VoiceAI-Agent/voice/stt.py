"""Speech-to-text engines behind one small interface, plus live (streaming) transcription.

Engines:
    ParakeetSTT  NVIDIA Parakeet TDT 0.6B v2 via onnx-asr (CPU). Default. Gives word timings.
    WhisperSTT   faster-whisper (optional [whisper] extra), kept as a backup. Text only.

StreamingSTT (roadmap Step 1.4) keeps the final transcript fast for long speech. A full
re-decode costs ~26 ms per second of audio on this CPU (3 s: 98 ms, 20 s: 527 ms), so:
- **Decode window + LocalAgreement.** Each partial decodes only the audio after the window
  start. Words that two consecutive partials agree on are *confirmed*. Once the window is longer
  than `trim_after_s`, confirmed words move into a fixed text prefix and the window moves up.
  So a decode never covers more than a few seconds.
- **Overlap anchor.** Parakeet's timestamps mark when a word is emitted, which can lag its real
  onset, so a cut placed just before a word still leaves enough of the previous word to be
  heard twice ("check check"). Instead, the last ANCHOR_WORDS confirmed words stay inside the new
  window and the cut goes TRIM_MARGIN_S before them. Every decode aligns on the anchor and drops
  whatever comes before it (the leftover of the previous word).
- **Partial reuse.** At speech end, if the last partial already covered all the speech, its
  text becomes the final (no decode at all).
- **No waiting for stale partials.** `finish()` never waits for a running partial: it marks it
  stale and decodes at once on a separate engine instance (`final_engine`). Two instances means
  no shared state and no lock, so overlapping decodes are safe.
- Partials are scheduled by **audio time** (every `partial_every_ms` of new audio), so behaviour
  is the same live and in tests (`synchronous=True` runs partials inline).
Engines without word timings (Whisper) fall back to full re-decodes.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
TRIM_MARGIN_S = 0.3  # the window starts this far before the first anchor word's timestamp
ANCHOR_WORDS = 2  # confirmed words kept inside the window after a trim, to align decodes on
MIN_DECODE_S = 0.1  # below this there is nothing to recognise
# A word may only be confirmed once the NEXT word has started, at least this long before the end
# of the decoded audio. The encoder also looks at future frames: a word at the very end is often
# cut or lacks right context, and two partials can agree on the same wrong guess ("3.5" for
# "three point four"). Requiring a following word makes sure the word is complete.
CONFIRM_GUARD_S = 0.3


def settled_words(words: list[Word], end_s: float) -> int:
    """How many leading words are safe to confirm: each must be followed by a word that starts
    at least CONFIRM_GUARD_S before the end of the audio."""
    count = 0
    for index in range(len(words) - 1):
        if words[index + 1].start > end_s - CONFIRM_GUARD_S:
            break
        count = index + 1
    return count


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds from the start of the decoded audio (or utterance, see below)


class STTEngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...


class ParakeetSTT:
    name = "parakeet"

    def __init__(
        self,
        model_dir: Path | str = "models/parakeet-tdt-0.6b-v2",
        quantization: str | None = "int8",
        providers: list[str] | None = None,
        threads: int = 0,
    ) -> None:
        """`threads`: CPU threads for this instance (0 = onnxruntime default, all cores)."""
        import onnx_asr
        import onnxruntime as ort

        # CPU by default: measured on the RTX 5050, DirectML took 300-390 ms per utterance
        # (every utterance has a new length) while the CPU took 65-130 ms. It also keeps
        # the GPU free for the LLM and TTS.
        providers = providers or ["CPUExecutionProvider"]
        providers = [p for p in providers if p in ort.get_available_providers()]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if threads:
            options.intra_op_num_threads = threads
        if "DmlExecutionProvider" in providers:
            options.enable_mem_pattern = False  # required by DirectML
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.model = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v2",
            model_dir,
            quantization=quantization,
            providers=providers,
            sess_options=options,
        )
        self._timed = self.model.with_timestamps()
        self.providers = providers
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # warm-up: first run is slow

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < SAMPLE_RATE * MIN_DECODE_S:
            return ""
        return self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()

    def transcribe_words(self, audio: np.ndarray) -> list[Word]:
        """Words with start times (seconds from the start of `audio`)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size < SAMPLE_RATE * MIN_DECODE_S:
            return []
        result = self._timed.recognize(audio, sample_rate=SAMPLE_RATE)
        return words_from_tokens(result.tokens or [], result.timestamps or [])


def words_from_tokens(tokens: list[str], timestamps: list[float]) -> list[Word]:
    """Join sub-word tokens into words. A token starting with a space begins a new word."""
    words: list[Word] = []
    text, start = "", 0.0
    for token, stamp in zip(tokens, timestamps):
        if token.startswith(" ") or not text:
            if text.strip():
                words.append(Word(text.strip(), start))
            text, start = token, stamp
        else:
            text += token
    if text.strip():
        words.append(Word(text.strip(), start))
    return words


class WhisperSTT:
    name = "whisper"

    def __init__(self, model_name: str = "small.en", device: str = "auto", compute_type: str = "auto"):
        from voice.transcriber import Transcriber

        self.transcriber = Transcriber(model_name, device=device, compute_type=compute_type)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcriber.transcribe(np.asarray(audio, dtype=np.float32))


def create_engine(name: str, **kwargs: object) -> STTEngine:
    engines = {"parakeet": ParakeetSTT, "whisper": WhisperSTT}
    if name not in engines:
        raise ValueError(f"Unknown STT engine '{name}'. Choose one of: {', '.join(engines)}")
    return engines[name](**kwargs)


def join_words(texts: list[str]) -> str:
    """Join words into text. A "word" that starts with punctuation (".py", ",") attaches to the
    previous word without a space, like the decoder's own text output ("settings.py")."""
    out = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        attach = out and not text[0].isalnum() and text[0] not in "\"'(["
        out += text if attach or not out else " " + text
    return out


def streaming_from_settings(settings) -> StreamingSTT:
    """StreamingSTT with two engines from config.settings.Settings.

    The partial engine gets fewer CPU threads (`stt_partial_threads`) so a final decode that
    overlaps a still-running partial isn't slowed down. Measured: with both engines on all
    threads the worst-case final took 224 ms; with 3 partial threads, 73 ms.
    """
    if settings.stt_engine == "parakeet":
        model_dir = Path(settings.models_dir) / "parakeet-tdt-0.6b-v2"
        partial = ParakeetSTT(model_dir, threads=settings.stt_partial_threads)
        final = ParakeetSTT(model_dir)
    else:
        partial = create_engine(settings.stt_engine)
        final = create_engine(settings.stt_engine)
    return StreamingSTT(partial, final_engine=final, trim_after_s=settings.stt_trim_after_s)


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _similar(a: str, b: str) -> bool:
    """Same word, allowing the spelling to vary between decodes ("Arrest" / "AREST")."""
    x, y = _norm(a), _norm(b)
    if x == y:
        return True
    return min(len(x), len(y)) >= 3 and SequenceMatcher(None, x, y).ratio() >= 0.75


def _agreement(a: list[Word], b: list[Word]) -> int:
    """Length of the common prefix of two hypotheses (words compared without case/punctuation)."""
    count = 0
    for x, y in zip(a, b):
        if _norm(x.text) != _norm(y.text):
            break
        count += 1
    return count


class StreamingSTT:
    """Collects one utterance and produces partial and final transcripts."""

    def __init__(self, engine: STTEngine, final_engine: STTEngine | None = None,
                 partial_every_ms: int = 300, trim_after_s: float = 2.5,
                 synchronous: bool = False) -> None:
        self.engine = engine
        self.final_engine = final_engine or engine
        self.partial_every = SAMPLE_RATE * partial_every_ms // 1000
        self.trim_after = int(SAMPLE_RATE * trim_after_s)
        self.synchronous = synchronous
        self.timed = hasattr(engine, "transcribe_words") and hasattr(self.final_engine, "transcribe_words")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-partial")
        self._lock = threading.Lock()
        self._partial_job: Future | None = None
        self.last_final: dict = {}
        self._reset()

    # -- utterance lifecycle -------------------------------------------------------------------
    def _reset(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._generation = getattr(self, "_generation", 0) + 1
        self._window_start = 0  # samples; decodes cover audio[window_start:]
        self._prefix: list[str] = []  # confirmed words before the window (text only)
        self._confirmed_in_window = 0  # confirmed words at the start of the current window
        self._anchor: list[Word] = []  # confirmed words (with times) the window starts with
        self._previous: list[Word] = []  # last hypothesis (utterance times), window words only
        self._last_partial_start = 0  # samples of audio when the last partial was scheduled
        self._last_hypothesis_end = 0  # samples covered by the last finished partial
        self.partial_text = ""
        self.active = False

    def start(self, pre_roll: np.ndarray | None = None) -> None:
        """Begin a new utterance. `pre_roll` is audio from just before VAD fired."""
        with self._lock:
            self._reset()
            if pre_roll is not None and len(pre_roll):
                self._append(pre_roll)
            self.active = True

    def resume(self, gap: np.ndarray | None = None) -> None:
        """Continue the same utterance after a short pause instead of starting a new one.

        `gap` is the audio recorded during the pause, so the words stay in order.
        """
        with self._lock:
            if gap is not None and len(gap):
                self._append(gap)
            self.active = True

    def feed(self, block: np.ndarray) -> None:
        """Add audio. Schedules a partial decode once `partial_every_ms` of new audio arrived."""
        if not self.active:
            return
        with self._lock:
            self._append(block)
            due = self._samples - self._last_partial_start >= self.partial_every
            busy = self._partial_job is not None and not self._partial_job.done()
            if not due or busy:
                return
            self._last_partial_start = self._samples
            job = (self._generation, self._window_start, self._samples, self._audio(self._window_start))
        if self.synchronous:
            self._partial(*job)
        else:
            self._partial_job = self._executor.submit(self._partial, *job)

    def finish(self, speech_end_sample: int | None = None) -> str:
        """End the utterance and return the final transcript.

        `speech_end_sample` is where the speech really ended (VAD); audio after it is silence.
        If the last partial already covered that point, its text is reused without decoding.
        """
        started = time.perf_counter()
        with self._lock:
            self.active = False
            end = self._samples if speech_end_sample is None else min(speech_end_sample, self._samples)
            self._generation += 1  # a partial still running is now stale: its result is ignored
            reusable = self.timed and self._previous and self._last_hypothesis_end >= end
            if reusable:
                words = [w.text for w in self._previous]
                mode = "reused"
            window_start = self._window_start
            audio = None if reusable else self._audio(window_start)
            prefix = list(self._prefix)
        if not reusable:
            if self.timed:
                words = [w.text for w in self._decode_window(self.final_engine, audio, window_start)]
                mode = "tail" if window_start > 0 else "window"
            else:
                words = [self.final_engine.transcribe(audio)]
                mode = "full"
        text = join_words([*prefix, *words])
        self.partial_text = text
        self.last_final = {"mode": mode, "decoded_s": 0.0 if reusable else len(audio) / SAMPLE_RATE,
                           "ms": (time.perf_counter() - started) * 1000,
                           "utterance_s": self._samples / SAMPLE_RATE}
        return text

    def cancel(self) -> None:
        with self._lock:
            self._reset()

    @property
    def duration(self) -> float:
        with self._lock:
            return self._samples / SAMPLE_RATE

    def audio(self) -> np.ndarray:
        """Copy of the utterance audio collected so far."""
        with self._lock:
            return self._audio(0)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    # -- internals -----------------------------------------------------------------------------
    def _append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        self._chunks.append(chunk)
        self._samples += chunk.size

    def _audio(self, start: int) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]  # keep one array: cheap slicing later
        return self._chunks[0][start:].copy()

    def _decode_window(self, engine, audio: np.ndarray, window_start: int,
                       anchor: list[Word] | None = None) -> list[Word]:
        """Decode the window; return words with utterance times, aligned on the anchor."""
        offset = window_start / SAMPLE_RATE
        words = [Word(w.text, w.start + offset) for w in engine.transcribe_words(audio)]
        anchor = self._anchor if anchor is None else anchor
        if window_start == 0 or not anchor:
            return words
        return align_on_anchor(words, anchor)

    def _partial(self, generation: int, window_start: int, end: int, audio: np.ndarray) -> None:
        if not self.timed:  # no word timings: the window never moves, so this is the whole utterance
            text = self.engine.transcribe(audio)
            with self._lock:
                if generation == self._generation:
                    self.partial_text = text
                    self._last_hypothesis_end = end
            return
        words = self._decode_window(self.engine, audio, window_start)
        with self._lock:
            if generation != self._generation or window_start != self._window_start:
                return  # stale: the utterance ended/restarted, or the window moved meanwhile
            agreed = _agreement(self._previous, words)
            self._confirmed_in_window = max(self._confirmed_in_window,
                                            min(agreed, settled_words(words, end / SAMPLE_RATE)))
            self._previous = words
            self._last_hypothesis_end = end
            self.partial_text = join_words([*self._prefix, *(w.text for w in words)])
            self._maybe_trim(end)

    def _maybe_trim(self, end: int) -> None:
        """Move confirmed words out of the window once it grows past `trim_after`.

        The last ANCHOR_WORDS confirmed words stay in the window (see the module docstring).
        """
        confirmed = self._confirmed_in_window
        keep = ANCHOR_WORDS
        if end - self._window_start < self.trim_after or confirmed <= keep:
            return  # short window, or not enough confirmed words to move any out
        first_anchor = self._previous[confirmed - keep]
        cut = first_anchor.start - TRIM_MARGIN_S
        if cut * SAMPLE_RATE <= self._window_start:
            return  # would not move the window forward
        self._prefix.extend(w.text for w in self._previous[: confirmed - keep])
        self._window_start = int(cut * SAMPLE_RATE)
        self._previous = self._previous[confirmed - keep :]
        self._confirmed_in_window = keep
        self._anchor = list(self._previous[:keep])


ANCHOR_TIME_TOLERANCE_S = 0.5  # the anchor must be found close to where it was confirmed


def align_on_anchor(words: list[Word], anchor: list[Word]) -> list[Word]:
    """Line a window decode up with the anchor (the confirmed words the window starts with).

    - Found as the full word sequence near its known time: drop everything before it (the
      leftover of the previous word).
    - Not found (the decoder skipped or clipped a word at the start of the clip): keep the
      anchor as confirmed and append only the words that come after it. A single common word
      like "the" is never enough to match, and confirmed words are never lost.
    """
    size = len(anchor)
    for index in range(len(words) - size + 1):
        if (all(_similar(w.text, a.text) for w, a in zip(words[index : index + size], anchor))
                and abs(words[index].start - anchor[0].start) <= ANCHOR_TIME_TOLERANCE_S):
            return [*anchor, *words[index + size :]]  # keep the confirmed spelling
    last = anchor[-1]
    after = [w for w in words if w.start > last.start + 0.08
             and not (_similar(w.text, last.text) and w.start - last.start < ANCHOR_TIME_TOLERANCE_S)]
    return [*anchor, *after]
