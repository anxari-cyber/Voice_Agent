import json
import threading
import time

import httpx
import pytest

from agent.llm import Delta, LLMError, OllamaLLM
from agent.memory import Memory, estimate_tokens
from agent.prompt import PromptBuilder


def chunk(content: str = "", done: bool = False, **extra) -> bytes:
    body = {"model": "m", "message": {"role": "assistant", "content": content}, "done": done, **extra}
    return (json.dumps(body) + "\n").encode()


def llm_with(handler) -> OllamaLLM:
    return OllamaLLM(url="http://ollama.test", model="m", num_ctx=4096, max_tokens=200,
                     transport=httpx.MockTransport(handler))


# ---- LLM client -----------------------------------------------------------------------------

def test_stream_yields_deltas_and_final_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = chunk("Hel") + chunk("lo") + chunk(done=True, eval_count=2, prompt_eval_count=9)
        return httpx.Response(200, content=body)

    deltas = list(llm_with(handler).stream([{"role": "user", "content": "hi"}]))

    assert "".join(d.content for d in deltas) == "Hello"
    assert deltas[-1].done and deltas[-1].stats["eval_count"] == 2
    assert not any(d.done for d in deltas[:-1])


def test_every_request_uses_identical_options_think_false_and_keep_alive() -> None:
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=chunk("x") + chunk(done=True))

    llm = llm_with(handler)
    builder = PromptBuilder("You are a test.", num_ctx=4096, max_tokens=200)
    llm.warm_up(builder.warm_up_messages())
    list(llm.stream(builder.build(Memory(), "one")))
    list(llm.stream(builder.build(Memory(), "two"), tools=[{"type": "function"}]))

    assert len(bodies) == 3
    for body in bodies:
        assert body["options"] == {"num_ctx": 4096, "num_predict": 200, "temperature": 0.6}
        assert body["think"] is False
        assert body["keep_alive"] == -1
        assert body["stream"] is True
        assert body["messages"][0] == {"role": "system", "content": "You are a test."}
    assert "tools" not in bodies[0] and "tools" in bodies[2]


def test_tool_calls_are_parsed_into_the_delta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "abc", "function": {"index": 0, "name": "get_time", "arguments": {"tz": "PKT"}}}]}
        line = json.dumps({"message": message, "done": True, "done_reason": "stop"}) + "\n"
        return httpx.Response(200, content=line.encode())

    (delta,) = list(llm_with(handler).stream([{"role": "user", "content": "time?"}]))

    assert delta.done
    assert delta.tool_calls[0].name == "get_time"
    assert delta.tool_calls[0].arguments == {"tz": "PKT"}
    assert delta.tool_calls[0].id == "abc"


def test_http_error_and_error_chunk_raise_llm_error() -> None:
    llm = llm_with(lambda r: httpx.Response(404, text='{"error":"model not found"}'))
    with pytest.raises(LLMError, match="404"):
        list(llm.stream([]))

    llm = llm_with(lambda r: httpx.Response(200, content=b'{"error":"out of memory"}\n'))
    with pytest.raises(LLMError, match="out of memory"):
        list(llm.stream([]))


def test_connection_failure_raises_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="request failed"):
        list(llm_with(handler).stream([]))


def test_cancel_from_another_thread_stops_the_stream() -> None:
    def slow_body():
        for i in range(100):
            yield chunk(f"t{i} ")
            time.sleep(0.01)
        yield chunk(done=True)

    llm = llm_with(lambda r: httpx.Response(200, content=slow_body()))
    received: list[Delta] = []
    for delta in llm.stream([]):
        received.append(delta)
        if len(received) == 5:
            threading.Thread(target=llm.cancel).start()
    assert 5 <= len(received) < 20
    assert not any(d.done for d in received)
    llm.cancel()  # cancelling when idle is harmless


def test_empty_content_chunks_are_passed_through_as_empty() -> None:
    # TTFT is measured on the first chunk with NON-empty content, so empty chunks must stay empty.
    body = chunk("") + chunk("") + chunk("Hi") + chunk(done=True)
    deltas = list(llm_with(lambda r: httpx.Response(200, content=body)).stream([]))
    assert [d.content for d in deltas] == ["", "", "Hi", ""]


# ---- Prompt ---------------------------------------------------------------------------------

def test_system_prompt_is_byte_identical_on_every_build() -> None:
    builder = PromptBuilder("Rules. Keep it short.", num_ctx=4096, max_tokens=200)
    memory = Memory()
    first = builder.build(memory, "hello")
    memory.add_user("hello")
    memory.add_assistant("Hi there.")
    second = builder.build(memory, "how are you?")

    assert first[0] is second[0]  # the very same dict every time
    assert json.dumps(first[0]).encode() == json.dumps(second[0]).encode()
    assert builder.warm_up_messages()[0] is first[0]
    assert second[-1] == {"role": "user", "content": "how are you?"}


def test_system_prompt_file_loads(tmp_path) -> None:
    path = tmp_path / "p.md"
    path.write_text("  Be brief.\n\n", encoding="utf-8")
    builder = PromptBuilder.from_file(path, num_ctx=4096, max_tokens=200)
    assert builder.system_message["content"] == "Be brief."


def test_repo_system_prompt_fits_and_has_no_placeholders() -> None:
    builder = PromptBuilder.from_file("config/system_prompt.md", num_ctx=4096, max_tokens=200)
    text = builder.system_message["content"]
    assert "{" not in text and "}" not in text  # nothing to .format() in: it must stay static
    assert builder.system_tokens < 1000


def test_prompt_that_cannot_fit_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        PromptBuilder("x" * 12_000, num_ctx=4096, max_tokens=200)


# ---- Memory ---------------------------------------------------------------------------------

def test_memory_keeps_most_recent_turns_within_token_budget() -> None:
    memory = Memory()
    for i in range(50):
        memory.add_user(f"question number {i} " * 5)
        memory.add_assistant(f"answer number {i} " * 5)
    budget = 400
    messages = memory.messages(budget)

    assert sum(estimate_tokens(m["content"]) for m in messages) <= budget
    assert messages[-1]["content"].startswith("answer number 49")
    assert messages[0]["role"] == "user"  # never starts with an assistant turn
    assert len(messages) < 100


def test_builder_history_shrinks_when_budget_is_small() -> None:
    memory = Memory()
    for i in range(200):
        memory.add_user("tell me something interesting " * 3)
        memory.add_assistant("here is something interesting " * 3)
    builder = PromptBuilder("Be brief.", num_ctx=1024, max_tokens=200)
    messages = builder.build(memory, "next")
    total = sum(estimate_tokens(m["content"]) for m in messages)
    assert total + 200 <= 1024


def test_truncate_last_assistant_keeps_only_what_was_heard() -> None:
    memory = Memory()
    memory.add_user("list three bugs")
    memory.add_assistant("The first bug is in login. The second is in the database.")
    memory.truncate_last_assistant("The first bug is in login.")
    assert memory.messages(10_000)[-1] == {"role": "assistant", "content": "The first bug is in login."}

    memory.truncate_last_assistant("")  # nothing heard -> the answer disappears
    assert memory.messages(10_000) == [{"role": "user", "content": "list three bugs"}]


# ---- Step 1.2 fixes -------------------------------------------------------------------------

def slow_stream(tag: str, n: int = 50):
    for i in range(n):
        yield chunk(f"{tag}{i} ")
        time.sleep(0.002)
    yield chunk(done=True, eval_count=n)


def test_two_overlapping_streams_cancel_only_the_first() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        tag = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(200, content=slow_stream(tag))

    llm = llm_with(handler)
    first = llm.stream([{"role": "user", "content": "A"}])
    second = llm.stream([{"role": "user", "content": "B"}])
    first_iter, second_iter = iter(first), iter(second)
    got_a = [next(first_iter).content for _ in range(3)]
    got_b = [next(second_iter).content for _ in range(3)]

    first.cancel()
    rest_a = list(first_iter)
    rest_b = list(second_iter)

    assert got_a == ["A0 ", "A1 ", "A2 "] and got_b == ["B0 ", "B1 ", "B2 "]
    assert rest_a == [] and first.cancelled
    assert not second.cancelled
    assert rest_b[-1].done and len(rest_b) == 48  # B3..B49 + done: untouched by A's cancel
    assert second.stats["eval_count"] == 50


def test_cancel_before_iterating_sends_no_request() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=chunk(done=True))

    stream = llm_with(handler).stream([])
    stream.cancel()
    assert list(stream) == []
    assert calls == []


def test_no_first_token_within_timeout_raises_quickly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no data", request=request)

    llm = OllamaLLM(url="http://ollama.test", model="m", first_token_timeout=10.0,
                    transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="sent nothing for 10 s"):
        list(llm.stream([]))


def test_warm_up_uses_the_long_timeout_and_real_timeout_is_short() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, content=chunk("x") + chunk(done=True))

    llm = OllamaLLM(url="http://ollama.test", model="m", first_token_timeout=10.0,
                    warm_up_timeout=120.0, transport=httpx.MockTransport(handler))
    llm.warm_up([{"role": "user", "content": "hi"}])
    list(llm.stream([{"role": "user", "content": "hi"}]))
    assert seen == [120.0, 10.0]


def test_token_estimator_calibrates_but_never_under_counts() -> None:
    from agent.memory import MAX_CHARS_PER_TOKEN, TokenEstimator

    estimator = TokenEstimator()
    messages = [{"role": "user", "content": "word " * 400}]  # 2000 chars
    for _ in range(10):  # Ollama says ~4.2 chars per token
        estimator.calibrate(messages, prompt_eval_count=int(2000 / 4.2) + 5)
    assert 3.0 < estimator.chars_per_token <= MAX_CHARS_PER_TOKEN
    assert estimator.tokens("x" * 4000) >= 4000 / 4.2  # still above the real count

    crazy = TokenEstimator()
    crazy.calibrate(messages, prompt_eval_count=10)  # absurd: 200 chars per token
    assert crazy.chars_per_token == MAX_CHARS_PER_TOKEN  # safe floor holds

    tiny = TokenEstimator()
    tiny.calibrate([{"role": "user", "content": "hi"}], prompt_eval_count=500)
    assert tiny.samples == 0  # too little text to learn from


def test_builder_calibration_skips_requests_with_tools() -> None:
    from agent.memory import TokenEstimator

    estimator = TokenEstimator()
    builder = PromptBuilder("Be brief.", num_ctx=4096, max_tokens=200, estimator=estimator)
    messages = [{"role": "user", "content": "hello there " * 50}]
    builder.calibrate(messages, {"prompt_eval_count": 150}, used_tools=True)
    assert estimator.samples == 0
    builder.calibrate(messages, {"prompt_eval_count": 150})
    assert estimator.samples == 1


def test_consecutive_user_turns_are_merged() -> None:
    memory = Memory()
    memory.add_user("open the")
    memory.add_assistant("Sure, opening")
    memory.truncate_last_assistant("")  # the answer was never heard
    memory.add_user("project folder")
    assert memory.messages(10_000) == [{"role": "user", "content": "open the project folder"}]

    builder = PromptBuilder("Be brief.", num_ctx=4096, max_tokens=200)
    messages = builder.build(memory, "and run the tests")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[-1]["content"] == "open the project folder and run the tests"


def test_system_prompt_mentions_speech_recognition_errors() -> None:
    text = PromptBuilder.from_file("config/system_prompt.md", 4096, 200).system_message["content"]
    assert "speech recognition" in text and "mistakes" in text
