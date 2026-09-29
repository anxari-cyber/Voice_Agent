import json
import threading
import time

from metrics.latency import EVENTS, LatencyLog


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_mark_does_not_write_until_the_background_thread_runs_and_flush_waits(tmp_path) -> None:
    log = LatencyLog(tmp_path / "l.jsonl", run_id="r")
    log.next_turn()
    for i in range(1000):
        log.mark("stt_partial", at=float(i), n=i)
    log.flush()
    records = read(tmp_path / "l.jsonl")
    assert len(records) == 1000
    assert [r["n"] for r in records] == list(range(1000))  # order kept
    assert records[0] == {**records[0], "run": "r", "turn": 1, "event": "stt_partial", "t": 0.0}
    log.close()


def test_marks_from_many_threads_are_all_written(tmp_path) -> None:
    log = LatencyLog(tmp_path / "l.jsonl")

    def worker(k: int) -> None:
        for i in range(200):
            log.mark("playback_start", worker=k, i=i)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.close()
    assert len(read(tmp_path / "l.jsonl")) == 8 * 200


def test_close_is_idempotent_and_marks_after_close_are_dropped(tmp_path) -> None:
    log = LatencyLog(tmp_path / "l.jsonl")
    log.mark("speech_start")
    log.close()
    log.close()
    log.mark("speech_end")  # must not raise or hang
    log.flush()
    assert [r["event"] for r in read(tmp_path / "l.jsonl")] == ["speech_start"]


def test_mark_returns_quickly(tmp_path) -> None:
    # Loose bound so the test never flakes; the real target (< 50 µs) is in bench/latency_mark_bench.py.
    log = LatencyLog(tmp_path / "l.jsonl")
    started = time.perf_counter()
    for _ in range(2000):
        log.mark("stt_final")
    per_call_us = (time.perf_counter() - started) / 2000 * 1e6
    log.close()
    assert per_call_us < 500


def test_new_step_1_1_events_are_known() -> None:
    for event in ("stt_partial", "turn_soft_end", "turn_commit", "turn_reopen",
                  "llm_cancel", "tts_cancel", "speculative_start"):
        assert event in EVENTS


def test_numpy_and_unserialisable_extras_do_not_kill_the_writer(tmp_path) -> None:
    import numpy as np

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    log = LatencyLog(tmp_path / "l.jsonl")
    log.mark("stt_final", prob=np.float32(0.5), count=np.int64(3), obj=Weird())
    log.mark("stt_final", ok=1)
    log.flush(timeout=2)

    assert log._writer.is_alive()
    first, second = read(tmp_path / "l.jsonl")
    assert first["prob"] == 0.5 and isinstance(first["prob"], float)
    assert first["count"] == 3
    assert first["obj"] == "<weird>"
    assert second["ok"] == 1
    log.close()


def test_a_record_that_cannot_be_formatted_becomes_an_error_line(tmp_path) -> None:
    log = LatencyLog(tmp_path / "l.jsonl")
    log.mark("stt_final", at=float("nan"), x={1, 2})  # set -> str via default
    log._queue.put((1, "stt_final", 1.0, "not-a-timestamp", {}))  # breaks datetime
    log.mark("speech_end", at=2.0)
    log.flush(timeout=2)

    records = read(tmp_path / "l.jsonl")
    assert log._writer.is_alive()
    assert [r["event"] for r in records] == ["stt_final", "stt_final", "speech_end"]
    assert "format_error" in records[1]
    log.close()
