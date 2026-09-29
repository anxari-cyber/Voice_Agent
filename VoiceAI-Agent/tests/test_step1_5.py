import asyncio
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

from agent.llm import Delta
from agent.memory import Memory
from agent.prompt import PromptBuilder
from pipeline.events import FinalText, SpeechEnd, SpeechStart, TurnCommit, TurnReopen
from pipeline.orchestrator import AudioFrontEnd, Orchestrator
from voice.vad import SileroVAD

SPEECH_WAV = Path(__file__).parent / "data" / "speech_16k.wav"  # 1 s silence, speech, 1 s silence
BLOCK = 320


def load_speech() -> np.ndarray:
    with wave.open(str(SPEECH_WAV), "rb") as file:
        audio = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    start, end = int(1.05 * 16_000), int(2.95 * 16_000)
    return audio[start:end]  # just the speech (VAD: ~1.12-2.91 s)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


class VirtualMic:
    """A mic on a virtual clock: block timestamps come from the sample count, so the join
    window is tested exactly without sleeping. Plays queued audio, otherwise silence."""

    def __init__(self) -> None:
        self.samples = 0
        self._audio = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.limit_s = 60.0  # stop producing after this much virtual time

    def say(self, audio: np.ndarray) -> None:
        with self._lock:
            self._audio = np.concatenate([self._audio, audio])

    def read(self, timeout=None):
        with self._lock:
            idle = self._audio.size == 0
        if idle and self.samples / 16_000 > self.limit_s:
            time.sleep(0.002)  # past the limit, virtual time only moves while there is speech
            raise queue.Empty
        with self._lock:
            block, self._audio = self._audio[:BLOCK], self._audio[BLOCK:]
        if block.size < BLOCK:
            block = np.concatenate([block, np.zeros(BLOCK - block.size, dtype=np.float32)])
        self.samples += BLOCK
        return self.samples / 16_000, block


class FakeSTT:
    """Stands in for StreamingSTT: the final text is scripted; records how it was used."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.active = False
        self.partial_text = ""
        self.last_final = {"mode": "tail", "ms": 1.0}
        self.calls: list[str] = []
        self.fed = 0

    def start(self, pre_roll=None):
        self.calls.append("start")
        self.active = True
        self.partial_text = "..."

    def resume(self, gap=None):
        self.calls.append("resume")
        self.active = True

    def feed(self, block):
        self.fed += len(block)

    def finish(self, speech_end_sample=None):
        self.calls.append("finish")
        self.active = False
        return self.texts[0] if self.texts else ""

    def cancel(self):
        self.active = False


def kinds(events: list) -> list[str]:
    return [type(e).__name__ for e in events if type(e).__name__ != "PartialText"]


def test_frontend_commits_a_turn_after_the_join_window() -> None:
    frontend_events = []
    mic = VirtualMic()
    mic.say(np.concatenate([silence(0.5), load_speech(), silence(1.5)]))
    mic.limit_s = 5.0
    stt = FakeSTT(["hello, can you check my project?"])
    frontend = AudioFrontEnd(mic, SileroVAD(), stt, emit=frontend_events.append, join_window_ms=700)
    thread = threading.Thread(target=frontend._run, daemon=True)
    thread.start()
    while mic.samples / 16_000 < 4.5:
        threading.Event().wait(0.01)
    frontend._stop.set()
    thread.join(2)

    assert kinds(frontend_events) == ["SpeechStart", "SpeechEnd", "FinalText", "TurnCommit"]
    end = next(e for e in frontend_events if isinstance(e, SpeechEnd))
    commit = next(e for e in frontend_events if isinstance(e, TurnCommit))
    assert commit.text == "hello, can you check my project?"
    assert commit.speech_end == end.at
    assert not frontend.listening.is_set()  # paused until the orchestrator answered


def test_a_pause_shorter_than_the_join_window_reopens_the_same_turn() -> None:
    speech = load_speech()
    audio = np.concatenate([silence(0.5), speech, silence(0.15), speech, silence(1.5)])
    mic = VirtualMic()
    mic.say(audio)
    mic.limit_s = len(audio) / 16_000
    events: list = []
    stt = FakeSTT(["one turn"])
    frontend = AudioFrontEnd(mic, SileroVAD(), stt, emit=events.append, join_window_ms=700)
    thread = threading.Thread(target=frontend._run, daemon=True)
    thread.start()
    while mic.samples / 16_000 < mic.limit_s:
        threading.Event().wait(0.01)
    frontend._stop.set()
    thread.join(2)

    names = kinds(events)
    assert names.count("TurnCommit") == 1
    assert "TurnReopen" in names
    assert stt.calls == ["start", "finish", "resume", "finish"]


# ---- orchestrator end to end, with fakes ----------------------------------------------------

class FakeStream:
    def __init__(self, text: str) -> None:
        self.text = text
        self.stats = {"prompt_eval_count": 50}

    def __iter__(self):
        for word in self.text.split(" "):
            yield Delta(content=word + " ")
        yield Delta(done=True, stats=self.stats)

    def cancel(self) -> None:
        pass


class FakeLLM:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.requests: list[list[dict]] = []

    def stream(self, messages, tools=None):
        self.requests.append(list(messages))
        return FakeStream(self.answers.pop(0))

    def cancel(self) -> None:
        pass


class FakeTTS:
    sample_rate = 24_000

    def synthesize_stream(self, text):
        yield np.zeros(2400, dtype=np.float32)

    def cancel(self):
        pass


class FakePlayer:
    def __init__(self) -> None:
        self.on_event = None
        self.played_samples = 0
        self.underflows = 0
        self.starved_blocks = 0
        self._started = False

    def play(self, audio):
        if not self._started and self.on_event:
            self._started = True
            self.on_event("playback_start", 0.0)
        self.played_samples += len(audio)

    def begin_utterance(self):
        self._started = False

    def end_utterance(self):
        pass

    def reset_counter(self):
        self.played_samples = 0

    def wait_until_done(self):
        pass

    def stop(self):
        pass


def test_orchestrator_answers_each_turn_and_remembers_the_conversation() -> None:
    mic = VirtualMic()
    mic.limit_s = 3.0
    stt = FakeSTT(["first question"])
    frontend = AudioFrontEnd(mic, SileroVAD(), stt, join_window_ms=700)
    llm = FakeLLM(["Sure, here is the first answer.", "And this is the second one."])
    memory = Memory()
    states: list[str] = []
    speech = load_speech()

    def on_state(name: str) -> None:
        states.append(name)
        if name == "listening" and len([s for s in states if s == "listening"]) <= 2:
            mic.say(np.concatenate([silence(0.3), speech, silence(1.2)]))
        if name == "speaking":
            mic.say(speech)  # the agent's own voice / someone talking over it: must be ignored
            stt.texts = ["second question"]

    class QuietUI:
        def state(self, name): pass
        def user_partial(self, text): pass
        def user_final(self, text): pass
        def agent_start(self): pass
        def agent_delta(self, text): pass
        def turn_done(self, stats): pass

    orchestrator = Orchestrator(frontend, llm, PromptBuilder("Be brief.", 4096, 200), memory,
                                FakeTTS(), FakePlayer(), ui=QuietUI(), on_state=on_state)
    turns = asyncio.run(asyncio.wait_for(orchestrator.run(max_turns=2), timeout=20))
    orchestrator.close()

    assert [t.user_text for t in turns] == ["first question", "second question"]
    assert [t.answer for t in turns] == ["Sure, here is the first answer.", "And this is the second one."]
    assert states[:4] == ["listening", "thinking", "speaking", "listening"]
    assert [m["role"] for m in memory.messages(10_000)] == ["user", "assistant", "user", "assistant"]
    assert llm.requests[1][1] == {"role": "user", "content": "first question"}  # history sent
    assert all(t.events.get("playback_start") is not None for t in turns)


def test_empty_turns_are_ignored() -> None:
    mic = VirtualMic()
    mic.limit_s = 20.0
    mic.say(np.concatenate([silence(0.3), load_speech(), silence(1.2)]))
    stt = FakeSTT([""])
    frontend = AudioFrontEnd(mic, SileroVAD(), stt, join_window_ms=700)
    llm = FakeLLM(["never used"])
    orchestrator = Orchestrator(frontend, llm, PromptBuilder("Be brief.", 4096, 200), Memory(),
                                FakeTTS(), FakePlayer())

    async def run_briefly():
        task = asyncio.create_task(orchestrator.run())
        while mic.samples / 16_000 < 5.0:
            await asyncio.sleep(0.01)
        orchestrator._stop.set()
        await task

    asyncio.run(asyncio.wait_for(run_briefly(), timeout=20))
    orchestrator.close()
    assert llm.requests == []
    assert frontend.listening.is_set()  # listening again after the empty turn


def test_event_types_are_plain_values() -> None:
    assert SpeechStart(1.0) == SpeechStart(1.0)
    assert FinalText("x", 1.0, 2.0, "tail").mode == "tail"
    assert TurnReopen(3.0).at == 3.0



def test_speech_end_times_stay_correct_after_listening_resumes() -> None:
    """Regression: VAD positions restart at every resume; stream time must not drift from them."""
    mic = VirtualMic()
    mic.limit_s = 3.0
    stt = FakeSTT(["a question"])
    frontend = AudioFrontEnd(mic, SileroVAD(), stt, join_window_ms=700)
    ends: list[tuple[float, float]] = []  # (reported speech end, mic time when reported)
    speech = load_speech()

    class QuietUI:
        def state(self, name): pass
        def user_partial(self, text): pass
        def user_final(self, text): pass
        def agent_start(self): pass
        def agent_delta(self, text): pass
        def turn_done(self, stats): pass

    def on_state(name: str) -> None:
        if name == "listening":
            mic.say(np.concatenate([silence(0.3), speech, silence(1.2)]))

    orchestrator = Orchestrator(frontend, FakeLLM(["One.", "Two.", "Three."]),
                                PromptBuilder("Be brief.", 4096, 200), Memory(), FakeTTS(),
                                FakePlayer(), ui=QuietUI(), on_state=on_state)
    original_emit = None

    async def run():
        nonlocal original_emit
        task = asyncio.create_task(orchestrator.run(max_turns=3))
        await asyncio.sleep(0)
        original_emit = frontend.emit

        def spy(event):
            if isinstance(event, SpeechEnd):
                ends.append((event.at, mic.samples / 16_000))
            original_emit(event)

        frontend.emit = spy
        return await task

    turns = asyncio.run(asyncio.wait_for(run(), timeout=20))
    orchestrator.close()
    assert len(turns) == 3 and len(ends) == 3
    for at, reported in ends:
        assert 0 <= reported - at < 1.0, (at, reported)
