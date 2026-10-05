"""Tests for mmorch.baseline: task building, the request each task sends, the result rows and the CSV.

The request shape and the body bytes are those captured from the legacy src/baseline/strategy_baseline.py (tag
v1.0.0). Nothing here opens a socket: post is a recording fake, and the exact body bytes come from requests' own
request preparation.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
import threading
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from mmorch import baseline
from mmorch.baseline import (
    DEFAULT_BENCHMARK,
    FIELDS,
    MAX_TOKENS,
    MAX_WORKERS,
    MODELS,
    SAMPLE_QUESTIONS,
    STRATEGIES,
    TIMEOUT_S,
    QuestionSet,
    Task,
    assign_model,
    build_tasks,
    call_api,
    output_path,
    resolve_endpoint,
    run_baseline,
    write_results,
)
from mmorch.errors import ConfigError, MissingDependencyError
from mmorch.settings import EndpointSettings

# A deterministic stand-in for the builtin hash, and the model it gives each strategy.
FIXED_HASH = {"balanced": 0, "quality": 1, "speed": 2, "cost": 3, "baseline": 4}.__getitem__
FIXED_MODELS = {
    "balanced": "gemma3",
    "quality": "llama3-sdsc",
    "speed": "llama3",
    "cost": "gemma3",
    "baseline": "llama3-sdsc",
}
TASK = Task("llama3", "Write a Python function", "balanced", "HE_0")
LEGACY_BODY = (
    b'{"model": "llama3", "messages": [{"role": "user", "content": "Write a Python function"}], "max_tokens": 150}'
)


class RecordingPost:
    """Stands in for requests.post: records (url, kwargs) of every call, then answers with a status or raises."""

    def __init__(self, status: int = 200, error: BaseException | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = status
        self.error = error
        self._lock = threading.Lock()

    def __call__(self, url: str, **kwargs: Any) -> SimpleNamespace:
        with self._lock:
            self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(status_code=self.status)


def legacy_row(latency: float, success: bool, task: Task = TASK) -> dict[str, object]:
    return {
        "benchmark": "HumanEval",
        "qid": task.qid,
        "strategy": task.strategy,
        "model": task.model,
        "latency": latency,
        "success": success,
    }


@pytest.fixture
def mmorch_logs(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog capturing the 'mmorch' loggers from INFO, even after the command line's logging setup.

    mmorch.cli.configure_logging stops propagation on the 'mmorch' logger, which would hide its records from
    caplog's root handler; this restores propagation for the test.
    """
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.INFO, logger="mmorch")
    return caplog


@pytest.fixture
def tiny_benchmark(monkeypatch: pytest.MonkeyPatch) -> str:
    """Replaces the benchmark table with one benchmark, 'Tiny', of a single question: five tasks."""
    monkeypatch.setattr(baseline, "SAMPLE_QUESTIONS", MappingProxyType({"Tiny": QuestionSet("T", "Tiny question", 1)}))
    return "Tiny"


# ---------------------------------------------------------------- the task table


def test_constants_keep_the_legacy_values() -> None:
    assert MODELS == ("gemma3", "llama3-sdsc", "llama3")
    assert STRATEGIES == ("balanced", "quality", "speed", "cost", "baseline")
    assert (DEFAULT_BENCHMARK, MAX_WORKERS, MAX_TOKENS, TIMEOUT_S) == ("HumanEval", 20, 150, 180)
    assert FIELDS == ("benchmark", "qid", "strategy", "model", "latency", "success")


def test_sample_questions_keep_the_legacy_table_and_order() -> None:
    assert list(SAMPLE_QUESTIONS.items()) == [
        ("HumanEval", QuestionSet("HE", "Write a Python function", 164)),
        ("MBPP", QuestionSet("MB", "Python problem", 500)),
        ("TruthfulQA", QuestionSet("TQ", "True or false question", 790)),
        ("ARC", QuestionSet("ARC", "Science question", 1172)),
        ("GSM8K", QuestionSet("GS", "Math problem", 1319)),
        ("GPQA", QuestionSet("GP", "Graduate physics question", 1725)),
        ("MATH", QuestionSet("MATH", "Advanced math problem", 5000)),
        ("HellaSwag", QuestionSet("HS", "Complete the sentence", 10042)),
        ("MMLU-Pro", QuestionSet("MMLU", "Multiple choice question", 12032)),
    ]
    with pytest.raises(TypeError):
        SAMPLE_QUESTIONS["X"] = QuestionSet("X", "x", 1)


def test_runs_per_benchmark_match_table1() -> None:
    from mmorch.paper.constants import TABLE1

    # Table 1 covers the eight benchmarks with routing traces; GPQA stays selectable but is not in the paper.
    assert set(TABLE1) == set(SAMPLE_QUESTIONS) - {"GPQA"}
    for benchmark, (runs, *_) in TABLE1.items():
        assert len(STRATEGIES) * SAMPLE_QUESTIONS[benchmark].count == runs, benchmark


def test_build_tasks_is_question_major_then_strategy_order() -> None:
    tasks = build_tasks("HumanEval", hash_fn=FIXED_HASH)
    assert len(tasks) == 820
    assert tasks[:6] == [
        Task("gemma3", "Write a Python function", "balanced", "HE_0"),
        Task("llama3-sdsc", "Write a Python function", "quality", "HE_0"),
        Task("llama3", "Write a Python function", "speed", "HE_0"),
        Task("gemma3", "Write a Python function", "cost", "HE_0"),
        Task("llama3-sdsc", "Write a Python function", "baseline", "HE_0"),
        Task("gemma3", "Write a Python function", "balanced", "HE_1"),
    ]
    assert tasks[-1] == Task("llama3-sdsc", "Write a Python function", "baseline", "HE_163")
    assert [task.qid for task in tasks[::5]] == [f"HE_{i}" for i in range(164)]
    assert [task.strategy for task in tasks] == list(STRATEGIES) * 164
    assert all(task.model == FIXED_MODELS[task.strategy] for task in tasks)
    assert {task.query for task in tasks} == {"Write a Python function"}


def test_build_tasks_keeps_gpqa_selectable() -> None:
    tasks = build_tasks("GPQA", hash_fn=FIXED_HASH)
    assert len(tasks) == 5 * 1725
    assert (tasks[0].qid, tasks[-1].qid) == ("GP_0", "GP_1724")
    assert tasks[0].query == "Graduate physics question"


def test_build_tasks_rejects_an_unknown_benchmark() -> None:
    with pytest.raises(KeyError):
        build_tasks("GPQA-Diamond")


def test_assign_model_uses_the_builtin_hash_by_default() -> None:
    for strategy in STRATEGIES:
        assert assign_model(strategy) == MODELS[hash(strategy) % 3]
    assert [task.model for task in build_tasks("HumanEval")[:5]] == [MODELS[hash(s) % 3] for s in STRATEGIES]


@pytest.mark.parametrize(
    ("hash_value", "expected"),
    [(0, "gemma3"), (1, "llama3-sdsc"), (5, "llama3"), (-1, "llama3"), (-5, "llama3-sdsc")],
)
def test_assign_model_honours_an_injected_hash(hash_value: int, expected: str) -> None:
    # Python's % keeps a negative hash in range: -1 % 3 == 2.
    assert assign_model("speed", hash_fn=lambda strategy: hash_value) == expected


def test_assign_model_takes_any_model_list() -> None:
    assert assign_model("speed", models=["a", "b"], hash_fn=lambda strategy: 3) == "b"


# ---------------------------------------------------------------- the endpoint


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (EndpointSettings("http://h/v1", "k"), ("http://h/v1", "k")),
        (EndpointSettings("http://h/v1/", "k"), ("http://h/v1", "k")),
        (EndpointSettings("http://h/v1///", None), ("http://h/v1", "")),
        (EndpointSettings("http://h/v1", ""), ("http://h/v1", "")),
        (EndpointSettings(" http://h/v1/ ", "k"), (" http://h/v1/ ", "k")),  # only trailing '/' is stripped
    ],
)
def test_resolve_endpoint(endpoint: EndpointSettings, expected: tuple[str, str]) -> None:
    assert resolve_endpoint(endpoint) == expected


@pytest.mark.parametrize("base", [None, "", "/", "///"])
def test_resolve_endpoint_requires_a_base(base: str | None) -> None:
    with pytest.raises(ConfigError) as excinfo:
        resolve_endpoint(EndpointSettings(base, "k"))
    assert str(excinfo.value) == "Set LLM_API_BASE and LLM_API_KEY; see .env.example"


# ---------------------------------------------------------------- one request


def test_call_api_sends_the_legacy_request(fake_clock: Any) -> None:
    post = RecordingPost(200)
    clock = fake_clock(1000.0, 1000.25)
    row = call_api(TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="k", post=post, clock=clock)
    assert post.calls == [
        (
            "http://h/v1/chat/completions",
            {
                "headers": {"Authorization": "Bearer k"},
                "json": {
                    "model": "llama3",
                    "messages": [{"role": "user", "content": "Write a Python function"}],
                    "max_tokens": 150,
                },
                "timeout": 180,
            },
        )
    ]
    kwargs = post.calls[0][1]
    assert list(kwargs["json"]) == ["model", "messages", "max_tokens"]
    assert type(kwargs["timeout"]) is int
    assert row == legacy_row(250.0, True)
    assert list(row) == list(FIELDS)
    assert clock.remaining == 0


def test_call_api_with_an_empty_key_sends_a_bare_bearer(fake_clock: Any) -> None:
    post = RecordingPost(200)
    call_api(TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="", post=post, clock=fake_clock(1.0, 2.0))
    assert post.calls[0][1]["headers"] == {"Authorization": "Bearer "}


def test_call_api_sends_the_legacy_body_bytes(fake_clock: Any) -> None:
    requests = pytest.importorskip("requests")
    post = RecordingPost(200)
    call_api(TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="k", post=post, clock=fake_clock(1.0, 2.0))
    url, kwargs = post.calls[0]
    # requests.post(url, headers=..., json=...) sends exactly what this prepared request holds.
    prepared = requests.Request("POST", url, headers=kwargs["headers"], json=kwargs["json"]).prepare()
    assert prepared.url == "http://h/v1/chat/completions"
    assert prepared.body == LEGACY_BODY
    assert prepared.headers["Authorization"] == "Bearer k"
    assert prepared.headers["Content-Type"] == "application/json"


def test_call_api_keeps_the_unrounded_latency(fake_clock: Any) -> None:
    clock = fake_clock(1000.0, 1000.1234567)
    row = call_api(TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="k", post=RecordingPost(), clock=clock)
    assert row["latency"] == (1000.1234567 - 1000.0) * 1000
    assert type(row["latency"]) is float
    assert row["success"] is True


@pytest.mark.parametrize("status", [500, 404, 201])
def test_call_api_another_status_is_a_silent_failure(
    status: int, fake_clock: Any, mmorch_logs: pytest.LogCaptureFixture
) -> None:
    clock = fake_clock(1000.0, 1000.5)
    row = call_api(
        TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="k", post=RecordingPost(status), clock=clock
    )
    assert row == legacy_row(0, False)
    assert type(row["latency"]) is int
    assert clock.remaining == 0  # the latency is measured, then dropped
    assert mmorch_logs.records == []


def test_call_api_an_exception_logs_one_warning(fake_clock: Any, mmorch_logs: pytest.LogCaptureFixture) -> None:
    clock = fake_clock(1000.0)  # a failed post leaves only the start reading
    post = RecordingPost(error=RuntimeError("boom"))
    row = call_api(TASK, benchmark="HumanEval", base_url="http://h/v1", api_key="k", post=post, clock=clock)
    assert row == legacy_row(0, False)
    assert type(row["latency"]) is int
    assert [(r.name, r.levelno, r.getMessage()) for r in mmorch_logs.records] == [
        ("mmorch.baseline", logging.WARNING, "Error HE_0/balanced/llama3: boom")
    ]


# ---------------------------------------------------------------- the whole run


def test_run_baseline_runs_every_task_once(mmorch_logs: pytest.LogCaptureFixture) -> None:
    post = RecordingPost(200)
    rows = run_baseline("HumanEval", base_url="http://h/v1", api_key="k", workers=4, hash_fn=FIXED_HASH, post=post)
    assert len(rows) == len(post.calls) == 820
    assert sorted((row["qid"], row["strategy"]) for row in rows) == sorted(
        (f"HE_{i}", strategy) for i in range(164) for strategy in STRATEGIES
    )
    for row in rows:
        assert list(row) == list(FIELDS)
        assert row["benchmark"] == "HumanEval"
        assert row["model"] == FIXED_MODELS[str(row["strategy"])]
        assert row["success"] is True
        assert type(row["latency"]) is float
    assert {url for url, _ in post.calls} == {"http://h/v1/chat/completions"}
    assert ("mmorch.baseline", logging.INFO, "HumanEval: sending 820 requests with 4 workers") in [
        (r.name, r.levelno, r.getMessage()) for r in mmorch_logs.records
    ]


def test_run_baseline_returns_rows_in_completion_order(tiny_benchmark: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every task waits until all five are running, then sleeps less the later it was submitted, so the tasks finish
    # in reverse submission order. The rows must come back in that order, not sorted and not in submission order.
    all_running = threading.Barrier(len(STRATEGIES))
    delays = {"balanced": 0.24, "quality": 0.18, "speed": 0.12, "cost": 0.06, "baseline": 0.0}
    original = baseline.call_api

    def slow_call_api(task: Task, **kwargs: Any) -> dict[str, object]:
        all_running.wait(timeout=10)
        time.sleep(delays[task.strategy])
        return original(task, **kwargs)

    monkeypatch.setattr(baseline, "call_api", slow_call_api)
    rows = run_baseline(tiny_benchmark, base_url="http://h/v1", api_key="k", workers=5, post=RecordingPost(200))
    assert [row["strategy"] for row in rows] == ["baseline", "cost", "speed", "quality", "balanced"]


def test_run_baseline_posts_with_requests_by_default(tiny_benchmark: str, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = pytest.importorskip("requests")
    post = RecordingPost(200)
    monkeypatch.setattr(requests, "post", post)
    rows = run_baseline(tiny_benchmark, base_url="http://h/v1", api_key="k", hash_fn=FIXED_HASH)
    assert len(post.calls) == 5
    assert sorted((row["qid"], row["strategy"]) for row in rows) == sorted(("T_0", s) for s in STRATEGIES)
    assert {kwargs["json"]["messages"][0]["content"] for _, kwargs in post.calls} == {"Tiny question"}


def test_run_baseline_without_requests_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "requests", None)
    with pytest.raises(MissingDependencyError) as excinfo:
        run_baseline("HumanEval", base_url="http://h/v1", api_key="k")
    assert str(excinfo.value) == 'requests is required for this command: pip install -e ".[live]"'
    assert (excinfo.value.package, excinfo.value.extra) == ("requests", "live")


def test_run_baseline_records_failures_as_rows(tiny_benchmark: str, mmorch_logs: pytest.LogCaptureFixture) -> None:
    rows = run_baseline(
        tiny_benchmark, base_url="http://h/v1", api_key="k", hash_fn=FIXED_HASH, post=RecordingPost(503)
    )
    assert [(row["latency"], row["success"]) for row in rows] == [(0, False)] * 5
    assert [r for r in mmorch_logs.records if r.levelno >= logging.WARNING] == []


# ---------------------------------------------------------------- the CSV


def test_output_path() -> None:
    assert output_path(Path("out"), "MBPP") == Path("out") / "baseline" / "MBPP_baseline.csv"


def test_write_results_writes_the_legacy_csv(tmp_path: Path) -> None:
    rows = [
        legacy_row(956.0410976409912, True),
        legacy_row(0, False, Task("gemma3", "Write a Python function", "quality", "HE_0")),
    ]
    path = tmp_path / "live" / "baseline" / "HumanEval_baseline.csv"
    assert write_results(path, rows) == path
    assert path.read_bytes().split(b"\r\n") == [
        b"benchmark,qid,strategy,model,latency,success",
        b"HumanEval,HE_0,balanced,llama3,956.0410976409912,True",
        b"HumanEval,HE_0,quality,gemma3,0,False",
        b"",
    ]


def test_a_run_writes_one_csv_row_per_task(tmp_path: Path) -> None:
    rows = run_baseline("HumanEval", base_url="http://h/v1", api_key="k", workers=4, post=RecordingPost(200))
    path = write_results(output_path(tmp_path, "HumanEval"), rows)
    data = path.read_bytes()
    assert data.count(b"\r\n") == 821
    assert b"\n" not in data.replace(b"\r\n", b"")
    records = list(csv.DictReader(io.StringIO(data.decode("ascii"), newline="")))
    assert len(records) == 820
    assert {record["success"] for record in records} == {"True"}
    assert all(float(record["latency"]) >= 0 for record in records)
    assert [record["qid"] for record in records] == [row["qid"] for row in rows]
