"""Tests for mmorch.routing.runner: model mapping, the timed streaming call, the thread pool and the 15-column CSV.

Everything runs against an in-process fake OpenAI client, so no socket is opened. The expected CallResult values
were produced by the v1.0.0 call_model (src/routing/smart_routing.py at tag v1.0.0) from the same fake streams and
scripted clock reads; the int-versus-float types matter, because the CSV writes an int 0 as '0' and a float as '0.0'.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import sys
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mmorch.data import Prompt
from mmorch.errors import ConfigError, MissingDependencyError
from mmorch.routing.classifier import Tier, classifier_prompt
from mmorch.routing.runner import (
    DEFAULT_TIMEOUT_S,
    DEFAULT_WORKERS,
    FIELDS,
    MODEL_KEYS,
    ROUTING_METHODS,
    CallResult,
    RouteRecord,
    TierModels,
    call_model,
    make_client,
    output_path,
    route_prompt,
    route_to_model,
    run_routing,
    write_rows,
)
from mmorch.settings import EndpointSettings

TIMINGS = ("latency_ms", "ttft_ms", "generation_time_ms", "tokens_per_second")
COUNTS = ("prompt_tokens", "completion_tokens", "total_tokens")


# ---------------------------------------------------------------- a fake OpenAI client


def content_chunk(text: str | None) -> SimpleNamespace:
    """A streamed chunk with one content delta and no usage, as the SDK parses it."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)


def bare_chunk(text: str) -> SimpleNamespace:
    """A streamed chunk object that has no usage attribute at all."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def usage_chunk(prompt_tokens: int | None, completion_tokens: int | None) -> SimpleNamespace:
    """The final chunk of a stream with usage: no choices, only token counts."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=None)
    return SimpleNamespace(choices=[], usage=usage)


HELLO_WORLD = (content_chunk("Hello"), content_chunk(" world"), usage_chunk(7, 2))


class FakeLLM:
    """client.chat.completions.create(**kwargs): records every call and answers it like the SDK would.

    A call with stream=True gets an iterator over stream(kwargs), by default the chunks given; any other call gets
    a completion whose message content is reply. error, when set, is raised by every call instead.
    """

    def __init__(
        self,
        chunks: Iterable[Any] = HELLO_WORLD,
        *,
        reply: str | None = "MEDIUM",
        error: Exception | None = None,
        stream: Callable[[dict[str, Any]], Iterable[Any]] | None = None,
    ) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.chunks = tuple(chunks)
        self.reply = reply
        self.error = error
        self.stream = stream
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Any:
        with self._lock:
            self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.stream(kwargs) if self.stream is not None else self.chunks)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])

    @property
    def generation_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("stream")]

    @property
    def classifier_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls if not c.get("stream")]


def generation_request(model: str, query: str, timeout: int = 90) -> dict[str, Any]:
    """The keyword arguments of the legacy generation call."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True,
        "timeout": timeout,
    }


def assert_types(result: CallResult | RouteRecord, **expected: type) -> None:
    for name, kind in expected.items():
        assert type(getattr(result, name)) is kind, name


# ---------------------------------------------------------------- tiers, model keys and served models


def test_routing_constants() -> None:
    assert ROUTING_METHODS == ("keyword", "llm")
    assert (DEFAULT_WORKERS, DEFAULT_TIMEOUT_S) == (20, 90)
    assert dict(MODEL_KEYS) == {"LOW": "llama3-small", "MEDIUM": "qwen3"}
    with pytest.raises(TypeError):
        MODEL_KEYS["HIGH"] = "x"


@pytest.mark.parametrize(
    ("tier", "key"),
    [
        (Tier.LOW, "llama3-small"),
        (Tier.MEDIUM, "qwen3"),
        (Tier.HIGH, "deepseek"),
        ("LOW", "llama3-small"),
        ("MEDIUM", "qwen3"),
        ("HIGH", "deepseek"),
        ("banana", "deepseek"),
        ("low", "deepseek"),  # the lookup is case-sensitive, as before
        ("", "deepseek"),
    ],
)
def test_route_to_model(tier: str, key: str) -> None:
    assert route_to_model(tier) == key


def test_tier_models_defaults_are_the_recorded_names() -> None:
    assert TierModels() == TierModels(low="llama3", medium="qwen3", high="deepseek-r1", classifier="llama3")
    assert TierModels.from_env({}) == TierModels()


def test_tier_models_classifier_follows_the_resolved_low_model() -> None:
    assert TierModels.from_env({"MODEL_LOW": "small-x"}) == TierModels("small-x", "qwen3", "deepseek-r1", "small-x")
    assert TierModels.from_env({"MODEL_LOW": "small-x", "MODEL_CLASSIFIER": "clf"}).classifier == "clf"
    assert TierModels.from_env({"MODEL_CLASSIFIER": "clf"}) == TierModels("llama3", "qwen3", "deepseek-r1", "clf")


def test_tier_models_reads_every_variable() -> None:
    environ = {"MODEL_LOW": "a", "MODEL_MEDIUM": "b", "MODEL_HIGH": "c", "MODEL_CLASSIFIER": "d", "OTHER": "x"}
    assert TierModels.from_env(environ) == TierModels("a", "b", "c", "d")


def test_tier_models_keeps_empty_strings() -> None:
    assert TierModels.from_env({"MODEL_LOW": "", "MODEL_MEDIUM": "", "MODEL_HIGH": ""}) == TierModels("", "", "", "")
    assert TierModels.from_env({"MODEL_LOW": "x", "MODEL_CLASSIFIER": ""}).classifier == ""


def test_tier_models_never_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_LOW", "from-the-shell")
    assert TierModels.from_env({}) == TierModels()


def test_served_maps_each_model_key_to_its_served_name() -> None:
    models = TierModels(low="small-x", medium="mid-y", high="big-z", classifier="clf")
    assert models.served("llama3-small") == "small-x"
    assert models.served("qwen3") == "mid-y"
    assert models.served("deepseek") == "big-z"
    with pytest.raises(KeyError):
        models.served("LOW")


def test_tier_models_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        TierModels().low = "x"


# ---------------------------------------------------------------- call_model


def test_call_model_times_a_stream_with_usage(fake_clock: Any) -> None:
    clock = fake_clock(1000.0, 1000.3141592, 1000.3150001, 1002.7182818, 1002.7190001)
    result = call_model(FakeLLM(HELLO_WORLD), "qwen3", "one two three", 90, clock=clock)

    assert result == CallResult(2718.28, 314.16, 2404.0, 0.83, 7, 2, 9, "Hello world", True, None)
    assert_types(result, **dict.fromkeys(TIMINGS, float), **dict.fromkeys(COUNTS, int))
    assert clock.remaining == 0


def test_call_model_reads_the_clock_in_the_legacy_order() -> None:
    events: list[str] = []
    values = iter([1000.0, 1000.001, 1000.01, 1000.1, 1001.01])

    def clock() -> float:
        events.append("clock")
        return next(values)

    def stream(kwargs: dict[str, Any]) -> Iterator[Any]:
        for chunk in (usage_chunk(None, None), *HELLO_WORLD):
            events.append("chunk")
            yield chunk
        events.append("end")

    llm = FakeLLM(stream=stream)
    original_create = llm.create

    def create(**kwargs: Any) -> Any:
        events.append("create")
        return original_create(**kwargs)

    llm.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    result = call_model(llm, "qwen3", "q", 90, clock=clock)

    # start, then the request; time to first token and the generation start right after the first content; the
    # total and the generation time after the stream ends.
    assert events == [
        "clock", "create", "chunk", "chunk", "clock", "clock", "chunk", "chunk", "end", "clock", "clock"
    ]  # fmt: skip
    assert (result.latency_ms, result.ttft_ms, result.generation_time_ms) == (100.0, 1.0, 1000.0)


def test_call_model_counts_words_without_usage(fake_clock: Any) -> None:
    # Chunks without a usage attribute, and empty or None deltas, which neither start the clock nor count.
    chunks = [
        content_chunk(None),
        bare_chunk("Hello"),
        content_chunk(None),
        content_chunk(""),
        bare_chunk(" big world"),
    ]
    clock = fake_clock(1000.0, 1000.25, 1000.5, 1001.0, 1001.875)
    result = call_model(FakeLLM(chunks), "qwen3", "one two three four", 90, clock=clock)

    assert result == CallResult(1000.0, 250.0, 1375.0, 1.45, 4, 3, 7, "Hello big world", True, None)


def test_call_model_counts_words_when_usage_reports_zero_or_none(fake_clock: Any) -> None:
    chunks = [content_chunk("a b"), usage_chunk(0, None)]
    clock = fake_clock(1000.0, 1000.25, 1000.5, 1001.0, 1001.875)
    result = call_model(FakeLLM(chunks), "qwen3", "x y", 90, clock=clock)

    assert result == CallResult(1000.0, 250.0, 1375.0, 0.73, 2, 2, 4, "a b", True, None)


def test_call_model_without_content_reads_the_clock_twice(fake_clock: Any) -> None:
    clock = fake_clock(1000.0, 1000.5)
    result = call_model(FakeLLM([usage_chunk(7, 2), content_chunk(None)]), "qwen3", "q", 90, clock=clock)

    assert result == CallResult(500.0, 0, 0, 0, 7, 2, 9, "", True, None)
    assert_types(result, latency_ms=float, ttft_ms=int, generation_time_ms=int, tokens_per_second=int)
    assert clock.reads == 2


def test_call_model_with_equal_clock_reads(fake_clock: Any) -> None:
    clock = fake_clock(1000.0, 1000.0, 1000.0, 1000.5, 1000.0)
    result = call_model(FakeLLM([content_chunk("Hi")]), "qwen3", "q", 90, clock=clock)

    # round(0.0 or 0, 2) is the int 0; round(0.0, 2) stays the float 0.0; no tokens/s without generation time.
    assert result == CallResult(500.0, 0, 0.0, 0, 1, 1, 2, "Hi", True, None)
    assert_types(result, ttft_ms=int, generation_time_ms=float, tokens_per_second=int)


def test_call_model_tests_the_generation_start_for_truthiness(fake_clock: Any) -> None:
    # A generation start read of 0.0 is falsy, so, as before, the generation time is the int 0 and is not read.
    clock = fake_clock(0.0, 0.0, 0.0, 0.5)
    result = call_model(FakeLLM([content_chunk("Hi there")]), "qwen3", "q", 90, clock=clock)

    assert result == CallResult(500.0, 0, 0, 0, 1, 2, 3, "Hi there", True, None)
    assert_types(result, generation_time_ms=int)
    assert clock.remaining == 0


def test_call_model_failure_from_the_request(fake_clock: Any) -> None:
    clock = fake_clock(1000.0)
    result = call_model(FakeLLM(error=RuntimeError("Error code: 400 - boom")), "qwen3", "q", 90, clock=clock)

    assert result == CallResult(0, 0, 0, 0, 0, 0, 0, "", False, "Error code: 400 - boom")
    assert_types(result, **dict.fromkeys(TIMINGS + COUNTS, int))
    assert result == CallResult.failure(RuntimeError("Error code: 400 - boom"))


def test_call_model_failure_in_the_middle_of_the_stream(fake_clock: Any) -> None:
    def stream(kwargs: dict[str, Any]) -> Iterator[Any]:
        yield content_chunk("partial")
        raise ConnectionError("connection reset")

    clock = fake_clock(1000.0, 1000.25, 1000.5)
    result = call_model(FakeLLM(stream=stream), "qwen3", "q", 90, clock=clock)

    assert result == CallResult(0, 0, 0, 0, 0, 0, 0, "", False, "connection reset")
    assert clock.remaining == 0


def test_call_model_sends_exactly_the_legacy_request(fake_clock: Any) -> None:
    llm = FakeLLM()
    query = 'Prove that "naïve" proofs fail.\n' + "x" * 600
    call_model(llm, "served-x", query, 45, clock=fake_clock(1000.0, 1000.1, 1000.2, 1000.3, 1000.4))

    assert llm.calls == [generation_request("served-x", query, timeout=45)]
    call = llm.calls[0]
    assert "stream_options" not in call
    # The SDK sends the timeout as the x-stainless-read-timeout header: an int must stay an int.
    assert type(call["timeout"]) is int
    assert type(call["temperature"]) is float
    assert type(call["max_tokens"]) is int


def test_call_model_keeps_the_first_500_characters_of_the_response(fake_clock: Any) -> None:
    pieces = ["a" * 300, " b" * 150, "c" * 100]
    clock = fake_clock(1000.0, 1000.01, 1000.02, 1003.33, 1003.34)
    result = call_model(FakeLLM(map(content_chunk, pieces)), "qwen3", "q", 90, clock=clock)

    text = "".join(pieces)
    assert result.response == text[:500]
    # The completion tokens are counted on the whole response, before it is cut.
    assert result == CallResult(3330.0, 10.0, 3320.0, 0.9, 1, 151, 152, text[:500], True, None)


def test_call_result_failure() -> None:
    result = CallResult.failure(TimeoutError("Request timed out."))
    assert dataclasses.astuple(result) == (0, 0, 0, 0, 0, 0, 0, "", False, "Request timed out.")
    assert_types(result, **dict.fromkeys(TIMINGS + COUNTS, int))


# ---------------------------------------------------------------- route_prompt


def test_route_prompt_keyword_keeps_200_characters_and_sends_the_whole_question(fake_clock: Any) -> None:
    llm = FakeLLM()
    prompt = Prompt("HumanEval_3", "HumanEval", "x" * 250)
    clock = fake_clock(1000.0, 1000.25, 1000.5, 1001.0, 1001.5)
    record = route_prompt(llm, prompt, "keyword", TierModels(), 90, clock=clock)

    assert record == RouteRecord(
        "HumanEval_3", "x" * 200, "MEDIUM", "keyword", "qwen3", 1000.0, 250.0, 1000.0, 2.0, 7, 2, 9, "Hello world",
        True, None,
    )  # fmt: skip
    assert llm.calls == [generation_request("qwen3", "x" * 250)]
    assert type(record.complexity) is str


@pytest.mark.parametrize(
    ("question", "complexity", "key", "served"),
    [
        ("What is the capital of France?", "LOW", "llama3-small", "small-x"),
        ("Finish the story", "MEDIUM", "qwen3", "mid-y"),
        ('Prove that "naïve" proofs fail.\nRésumé', "HIGH", "deepseek", "big-z"),
    ],
)
def test_route_prompt_keyword_sends_the_served_name_and_records_the_key(
    question: str, complexity: str, key: str, served: str
) -> None:
    llm = FakeLLM()
    models = TierModels(low="small-x", medium="mid-y", high="big-z", classifier="clf")
    record = route_prompt(llm, Prompt("HumanEval_1", "HumanEval", question), "keyword", models, 90)

    assert (record.complexity, record.routing_method, record.model) == (complexity, "keyword", key)
    assert llm.calls == [generation_request(served, question)]
    assert llm.classifier_calls == []


@pytest.mark.parametrize(
    ("reply", "complexity", "key", "served"),
    [
        ("HIGH", "HIGH", "deepseek", "big-z"),
        (" low\n", "LOW", "llama3-small", "small-x"),
        ("banana", "MEDIUM", "qwen3", "mid-y"),
    ],
)
def test_route_prompt_llm_classifies_with_the_classifier_model(
    reply: str, complexity: str, key: str, served: str
) -> None:
    llm = FakeLLM(reply=reply)
    models = TierModels(low="small-x", medium="mid-y", high="big-z", classifier="clf")
    question = "What is the capital of France? " * 30
    record = route_prompt(llm, Prompt("HumanEval_1", "HumanEval", question), "llm", models, 30)

    assert llm.calls == [
        {
            "model": "clf",
            "messages": [{"role": "user", "content": classifier_prompt(question)}],
            "max_tokens": 10,
            "temperature": 0.0,
            "timeout": 10,
        },
        generation_request(served, question, timeout=30),
    ]
    assert (record.question, record.complexity, record.routing_method, record.model) == (
        question[:200],
        complexity,
        "llm",
        key,
    )


def test_route_prompt_records_a_failed_call_as_data() -> None:
    record = route_prompt(
        FakeLLM(error=RuntimeError("down")), Prompt("ARC_1", "ARC", "Pick one"), "llm", TierModels(), 90
    )

    # The classifier call fails too, so the tier falls back to MEDIUM.
    assert record == RouteRecord("ARC_1", "Pick one", "MEDIUM", "llm", "qwen3", 0, 0, 0, 0, 0, 0, 0, "", False, "down")


# ---------------------------------------------------------------- run_routing


def prompts_of(*questions: str) -> list[Prompt]:
    return [Prompt(f"HumanEval_{i}", "HumanEval", q) for i, q in enumerate(questions)]


def test_run_routing_returns_prompt_order_when_later_prompts_finish_first() -> None:
    n = 4
    done = [threading.Event() for _ in range(n)]
    finished: list[int] = []

    def stream(kwargs: dict[str, Any]) -> Iterator[Any]:
        # Each stream waits for the next prompt's stream to finish, so they finish in reverse order.
        i = int(kwargs["messages"][0]["content"].removeprefix("q"))
        if i + 1 < n and not done[i + 1].wait(timeout=10):
            raise TimeoutError(f"stream {i + 1} never finished")
        yield content_chunk(f"answer {i}")
        finished.append(i)
        done[i].set()

    prompts = prompts_of(*(f"q{i}" for i in range(n)))
    records = run_routing(FakeLLM(stream=stream), prompts, "keyword", TierModels(), workers=n)

    assert finished == [3, 2, 1, 0]
    assert [r.qid for r in records] == [p.qid for p in prompts]
    assert [r.response for r in records] == [f"answer {i}" for i in range(n)]
    assert all(r.success for r in records)


def test_run_routing_logs_progress_every_200_completions(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command line may have configured the 'mmorch' logger not to propagate; caplog listens on the root.
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.INFO, logger="mmorch.routing.runner")

    records = run_routing(FakeLLM(), prompts_of(*["q"] * 250), "keyword", TierModels())

    assert len(records) == 250
    progress = [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "mmorch.routing.runner"]
    assert progress == [(logging.INFO, "200/250")]


def test_run_routing_logs_each_multiple_of_200(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.INFO, logger="mmorch.routing.runner")

    run_routing(FakeLLM(), prompts_of(*["q"] * 400), "keyword", TierModels())

    assert [r.getMessage() for r in caplog.records if r.name == "mmorch.routing.runner"] == ["200/400", "400/400"]


def test_run_routing_calls_the_classifier_only_for_llm_routing() -> None:
    questions = ("What is 2+2?", "Prove it", "Finish the story")

    keyword = FakeLLM(reply="HIGH")
    records = run_routing(keyword, prompts_of(*questions), "keyword", TierModels())
    assert keyword.classifier_calls == []
    assert [r.model for r in records] == ["llama3-small", "deepseek", "qwen3"]

    llm = FakeLLM(reply="HIGH")
    records = run_routing(llm, prompts_of(*questions), "llm", TierModels())
    assert sorted(c["messages"][0]["content"] for c in llm.classifier_calls) == sorted(
        map(classifier_prompt, questions)
    )
    assert [c["model"] for c in llm.generation_calls] == ["deepseek-r1"] * 3
    assert [(r.complexity, r.model, r.routing_method) for r in records] == [("HIGH", "deepseek", "llm")] * 3


def test_run_routing_passes_workers_timeout_and_clock() -> None:
    llm = FakeLLM()
    questions = [f"Question {i}" for i in range(5)]
    clock = itertools.count(1000.0, 0.5).__next__  # every read is half a second after the previous one
    records = run_routing(llm, prompts_of(*questions), "keyword", TierModels(), workers=1, timeout=7, clock=clock)

    # One worker runs the prompts one after another, in order, each reading the clock five times.
    assert llm.calls == [generation_request("qwen3", q, timeout=7) for q in questions]
    timings = [(r.latency_ms, r.ttft_ms, r.generation_time_ms, r.tokens_per_second) for r in records]
    assert timings == [(1500.0, 500.0, 1000.0, 2.0)] * 5
    assert clock() == 1000.0 + 0.5 * 25


def test_run_routing_runs_at_most_workers_prompts_at_once() -> None:
    release = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    def stream(kwargs: dict[str, Any]) -> Iterator[Any]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if not release.wait(timeout=10):
                raise TimeoutError("the test never released the streams")
            yield content_chunk("ok")
        finally:
            with lock:
                active -= 1

    records: list[RouteRecord] = []
    worker = threading.Thread(
        target=lambda: records.extend(
            run_routing(FakeLLM(stream=stream), prompts_of(*["q"] * 6), "keyword", TierModels(), workers=2)
        )
    )
    worker.start()
    try:
        deadline = time.monotonic() + 10
        while peak < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        time.sleep(0.05)  # room for a pool that ignored workers to start more streams
        assert peak == 2
    finally:
        release.set()
        worker.join(timeout=10)
    assert len(records) == 6
    assert all(r.success for r in records)


def test_run_routing_with_no_prompts() -> None:
    llm = FakeLLM()
    assert run_routing(llm, [], "llm", TierModels()) == []
    assert llm.calls == []


# ---------------------------------------------------------------- the 15-column CSV


def test_fields_are_the_15_legacy_columns() -> None:
    assert FIELDS == (
        "qid", "question", "complexity", "routing_method", "model", "latency_ms", "ttft_ms", "generation_time_ms",
        "tokens_per_second", "prompt_tokens", "completion_tokens", "total_tokens", "response", "success", "error",
    )  # fmt: skip


def test_as_row_keeps_the_field_order_and_the_python_values() -> None:
    record = RouteRecord(
        "HumanEval_1", "q", "LOW", "keyword", "llama3-small", 12.5, 0, 0.0, 0, 7, 2, 9, "r", True, None
    )
    row = record.as_row()
    assert list(row) == list(FIELDS)
    assert list(row.values()) == list(dataclasses.astuple(record))
    assert type(row["ttft_ms"]) is int
    assert type(row["generation_time_ms"]) is float


def test_write_rows_bytes(tmp_path: Path) -> None:
    ok = RouteRecord(
        "HumanEval_2", 'Prove that "naïve"\nproofs', "HIGH", "keyword", "deepseek", 2718.28, 0, 0.0, 0.83, 7, 2, 9,
        "Hello, world", True, None,
    )  # fmt: skip
    error = "Error code: 400 - {'error': {'message': 'forced', 'type': 'stub'}}"
    failed = RouteRecord(
        "HumanEval_3", "x" * 3, "MEDIUM", "llm", "qwen3", *dataclasses.astuple(CallResult.failure(RuntimeError(error)))
    )

    path = tmp_path / "live" / "keyword" / "HumanEval_keyword.csv"
    assert write_rows(path, [ok, failed]) == path

    assert path.read_bytes() == (
        b"qid,question,complexity,routing_method,model,latency_ms,ttft_ms,generation_time_ms,tokens_per_second,"
        b"prompt_tokens,completion_tokens,total_tokens,response,success,error\r\n"
        + 'HumanEval_2,"Prove that ""naïve""\nproofs",HIGH,keyword,deepseek,2718.28,0,0.0,0.83,7,2,9,"Hello, world",'
        "True,\r\n".encode()
        + b"HumanEval_3,xxx,MEDIUM,llm,qwen3,0,0,0,0,0,0,0,,False,"
        b"\"Error code: 400 - {'error': {'message': 'forced', 'type': 'stub'}}\"\r\n"
    )


def test_write_rows_with_no_records_writes_the_header(tmp_path: Path) -> None:
    path = write_rows(tmp_path / "out.csv", [])
    assert path.read_bytes() == ",".join(FIELDS).encode() + b"\r\n"


@pytest.mark.parametrize("method", ["keyword", "llm"])
def test_output_path(method: str) -> None:
    out = Path("results") / "live"
    assert output_path(out, "MMLU-Pro", method) == out / method / f"MMLU-Pro_{method}.csv"


# ---------------------------------------------------------------- make_client


@pytest.mark.parametrize("base", [None, ""])
def test_make_client_requires_a_base_url(base: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    # The base is checked before openai is imported.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ConfigError) as excinfo:
        make_client(EndpointSettings(base, "key"))
    assert str(excinfo.value) == "Set LLM_API_BASE (and LLM_API_KEY); see .env.example"


def test_make_client_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError) as excinfo:
        make_client(EndpointSettings("http://127.0.0.1:9/v1", "key"))
    assert 'pip install -e ".[live]"' in str(excinfo.value)
    assert (excinfo.value.package, excinfo.value.extra) == ("openai", "live")
    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.parametrize(("key", "sent"), [(None, "none"), ("", "none"), ("test-key", "test-key")])
def test_make_client_passes_only_the_base_url_and_key(
    key: str | None, sent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, Any]] = []

    class RecordingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = RecordingOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = make_client(EndpointSettings("http://127.0.0.1:9/v1", key))

    assert isinstance(client, RecordingOpenAI)
    assert created == [{"base_url": "http://127.0.0.1:9/v1", "api_key": sent}]


def test_make_client_builds_an_sdk_client_with_its_defaults() -> None:
    openai = pytest.importorskip("openai")
    client = make_client(EndpointSettings("http://127.0.0.1:9/v1", ""))

    assert isinstance(client, openai.OpenAI)
    assert client.api_key == "none"
    assert str(client.base_url) == "http://127.0.0.1:9/v1/"
    # Untouched SDK retries: they decide how many requests reach the endpoint on 5xx errors and timeouts.
    assert client.max_retries == openai.DEFAULT_MAX_RETRIES
