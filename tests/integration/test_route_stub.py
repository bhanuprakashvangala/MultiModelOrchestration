"""`mmorch route` end to end against the local OpenAI-compatible stub (the llm_stub fixture of tests/conftest.py).

The tests pin what the endpoint receives (the JSON bodies, the Authorization and timeout headers and the number of
requests) and the 15-column CSV that the runner writes, for keyword and LLM-prompt routing, for the MODEL_*
overrides, for failed requests and for --limit. The prompts are those of tiny_prompts: three HumanEval prompts
(LOW, HIGH and MEDIUM by the keyword rules) and one MBPP prompt that the benchmark filter drops.

They need the 'live' extra (openai) and skip without it.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mmorch.cli import main
from mmorch.routing.classifier import classifier_prompt

pytest.importorskip("openai")

# The runner's CSV columns, in order: smart_routing.FIELDS at v1.0.0.
FIELDS = (
    "qid",
    "question",
    "complexity",
    "routing_method",
    "model",
    "latency_ms",
    "ttft_ms",
    "generation_time_ms",
    "tokens_per_second",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "response",
    "success",
    "error",
)
TIMING_COLUMNS = ("latency_ms", "ttft_ms", "generation_time_ms", "tokens_per_second")
TOKEN_COLUMNS = ("prompt_tokens", "completion_tokens", "total_tokens")

# The header in which openai-python sends a request's timeout (older releases of the SDK do not send it).
READ_TIMEOUT_HEADER = "x-stainless-read-timeout"

# The HumanEval prompts of tiny_prompts in file order: keyword tier, model key (the CSV's model column) and the
# served model name the request goes to by default.
KEYWORD_ROUTES = (
    ("HumanEval_1", "LOW", "llama3-small", "llama3"),
    ("HumanEval_2", "HIGH", "deepseek", "deepseek-r1"),
    ("HumanEval_3", "MEDIUM", "qwen3", "qwen3"),
)
HUMANEVAL_QIDS = [qid for qid, *_ in KEYWORD_ROUTES]


@dataclass(frozen=True)
class Run:
    """One `mmorch route HumanEval` run: exit status, stdout lines, stderr text and the CSV path it reports."""

    status: int
    stdout: list[str]
    stderr: str
    csv_path: Path


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty directory, so that no default path can reach the repository's data/ or results/."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def endpoint(llm_stub: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The stub, set as the endpoint with LLM_API_KEY 'test-key'."""
    monkeypatch.setenv("LLM_API_BASE", llm_stub.base_url)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    return llm_stub


@pytest.fixture
def questions(tiny_prompt_records: list[dict[str, str]]) -> dict[str, str]:
    """{qid: question} of the tiny prompts."""
    return {record["qid"]: record["question"] for record in tiny_prompt_records}


def route(capsys: pytest.CaptureFixture[str], prompts: Path, out: Path, *options: str, llm: bool = False) -> Run:
    """Run `mmorch route HumanEval` on prompts with two workers, writing under out.

    Keyword routing is the default and is not named on the command line; llm=True adds '--routing llm'.
    """
    routing = ["--routing", "llm"] if llm else []
    paths = ["--prompts", str(prompts), "--out", str(out)]
    status = main(["route", "HumanEval", *routing, *paths, "--workers", "2", *options])
    captured = capsys.readouterr()
    method = "llm" if llm else "keyword"
    return Run(status, captured.out.splitlines(), captured.err, out / method / f"HumanEval_{method}.csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def canonical(body: Any) -> str:
    """A request body as sorted-key JSON: equal for equal bodies, and still telling 0.0 from 0 and true from 1."""
    return json.dumps(body, sort_keys=True)


def generation_body(question: str, model: str) -> dict[str, Any]:
    """The legacy streaming generation request: the full question, 512 tokens, temperature 0.7, nothing else."""
    return {
        "messages": [{"role": "user", "content": question}],
        "model": model,
        "max_tokens": 512,
        "stream": True,
        "temperature": 0.7,
    }


def classifier_body(question: str, model: str) -> dict[str, Any]:
    """The legacy classification request: non-streaming, 10 tokens and the float temperature 0.0."""
    return {
        "messages": [{"role": "user", "content": classifier_prompt(question)}],
        "model": model,
        "max_tokens": 10,
        "temperature": 0.0,
    }


def is_number(cell: str) -> bool:
    try:
        return float(cell) >= 0
    except ValueError:
        return False


# ---------------------------------------------------------------- keyword routing


def test_keyword_routing_sends_one_streaming_request_per_prompt(
    endpoint: Any, tiny_prompts: Path, questions: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = route(capsys, tiny_prompts, tmp_path / "live")

    assert run.status == 0, run.stderr
    # No classifier call: the keyword rules pick the tier, then each prompt goes to its tier's served model.
    expected = [generation_body(questions[qid], served) for qid, _, _, served in KEYWORD_ROUTES]
    assert Counter(map(canonical, endpoint.json_bodies())) == Counter(map(canonical, expected))
    assert [request.path for request in endpoint.requests] == ["/v1/chat/completions"] * 3
    assert {request.headers["Authorization"] for request in endpoint.requests} == {"Bearer test-key"}


def test_keyword_routing_writes_the_15_column_csv(
    endpoint: Any, tiny_prompts: Path, questions: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "live"
    run = route(capsys, tiny_prompts, out)

    assert run.status == 0, run.stderr
    assert run.csv_path == out / "keyword" / "HumanEval_keyword.csv"
    assert [path for path in out.rglob("*") if path.is_file()] == [run.csv_path]
    assert run.csv_path.read_bytes().split(b"\r\n", 1)[0] == ",".join(FIELDS).encode()
    rows = read_rows(run.csv_path)
    assert [row["qid"] for row in rows] == HUMANEVAL_QIDS  # prompt order, whatever order the requests finished in
    for row, (qid, tier, model_key, _) in zip(rows, KEYWORD_ROUTES):
        # The second question has an embedded newline and double quotes; the third is cut from 250 characters.
        assert row["question"] == questions[qid][:200]
        assert (row["complexity"], row["routing_method"], row["model"]) == (tier, "keyword", model_key)
        assert row["response"] == "Hello world"
        assert [row[column] for column in TOKEN_COLUMNS] == ["7", "2", "9"]  # from the stream's usage chunk
        assert (row["success"], row["error"]) == ("True", "")
        assert [column for column in TIMING_COLUMNS if not is_number(row[column])] == []


def test_keyword_routing_prints_one_summary_line(
    endpoint: Any, tiny_prompts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = route(capsys, tiny_prompts, tmp_path / "live")

    assert run.status == 0, run.stderr
    assert run.stdout == [f"HumanEval [keyword]: 3/3 succeeded -> {run.csv_path}"]


# ---------------------------------------------------------------- LLM-prompt routing


@pytest.mark.parametrize(
    ("environ", "classifier_model", "generation_model"),
    [
        pytest.param({}, "llama3", "deepseek-r1", id="defaults"),
        # The classifier defaults to the low tier's served model.
        pytest.param({"MODEL_LOW": "small-x"}, "small-x", "deepseek-r1", id="MODEL_LOW"),
        # Only the served name changes; the CSV keeps the model key.
        pytest.param({"MODEL_HIGH": "big-y"}, "llama3", "big-y", id="MODEL_HIGH"),
    ],
)
def test_llm_routing_classifies_then_generates(
    endpoint: Any,
    monkeypatch: pytest.MonkeyPatch,
    tiny_prompts: Path,
    questions: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    environ: dict[str, str],
    classifier_model: str,
    generation_model: str,
) -> None:
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    endpoint.reply = "HIGH"

    run = route(capsys, tiny_prompts, tmp_path / "live", llm=True)

    assert run.status == 0, run.stderr
    bodies = endpoint.json_bodies()
    texts = [questions[qid] for qid in HUMANEVAL_QIDS]
    classify = [body for body in bodies if not body.get("stream")]
    generate = [body for body in bodies if body.get("stream")]
    assert Counter(map(canonical, classify)) == Counter(canonical(classifier_body(q, classifier_model)) for q in texts)
    assert [type(body["temperature"]) for body in classify] == [float] * 3
    assert Counter(map(canonical, generate)) == Counter(canonical(generation_body(q, generation_model)) for q in texts)
    sent = [(bool(body.get("stream")), body["messages"][0]["content"]) for body in bodies]
    for text in texts:
        assert sent.index((False, classifier_prompt(text))) < sent.index((True, text)), "generated before classified"

    rows = read_rows(run.csv_path)
    assert [row["qid"] for row in rows] == HUMANEVAL_QIDS
    assert {(row["complexity"], row["routing_method"], row["model"], row["success"]) for row in rows} == {
        ("HIGH", "llm", "deepseek", "True")
    }
    assert run.stdout == [f"HumanEval [llm]: 3/3 succeeded -> {run.csv_path}"]


@pytest.mark.parametrize(
    ("options", "generation_timeout"),
    [pytest.param((), "90", id="default"), pytest.param(("--timeout", "30"), "30", id="timeout-30")],
)
def test_requests_carry_the_legacy_timeouts(
    endpoint: Any,
    tiny_prompts: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    options: tuple[str, ...],
    generation_timeout: str,
) -> None:
    run = route(capsys, tiny_prompts, tmp_path / "live", *options, llm=True)

    assert run.status == 0, run.stderr
    requests = endpoint.requests
    if not all(READ_TIMEOUT_HEADER in request.headers for request in requests):
        pytest.skip(f"this openai version does not send the {READ_TIMEOUT_HEADER} header")
    timeouts = Counter(
        (bool(json.loads(request.body).get("stream")), request.headers[READ_TIMEOUT_HEADER]) for request in requests
    )
    # Classification keeps its fixed 10 s. Generation uses --timeout, an int as before, so the header reads '90'
    # and not '90.0'.
    assert timeouts == Counter({(False, "10"): 3, (True, generation_timeout): 3})


# ---------------------------------------------------------------- failures and --limit


def test_failed_requests_are_recorded_as_rows(
    endpoint: Any, tiny_prompts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    endpoint.status = 400

    run = route(capsys, tiny_prompts, tmp_path / "live")

    assert run.status == 0, run.stderr  # failed requests are data, not an error
    assert len(endpoint.requests) == 3  # one per prompt: the SDK does not retry a 400
    rows = read_rows(run.csv_path)
    assert [row["qid"] for row in rows] == HUMANEVAL_QIDS
    for row in rows:
        assert (row["success"], row["response"]) == ("False", "")
        assert [row[column] for column in (*TIMING_COLUMNS, *TOKEN_COLUMNS)] == ["0"] * 7
        assert row["error"].startswith("Error code: 400"), row["error"]
    assert run.stdout == [f"HumanEval [keyword]: 0/3 succeeded -> {run.csv_path}"]


@pytest.mark.parametrize(
    ("limit", "qids"),
    [
        pytest.param("1", ["HumanEval_1"], id="first"),
        pytest.param("0", [], id="none"),
        pytest.param("-1", ["HumanEval_1", "HumanEval_2"], id="all-but-last"),
    ],
)
def test_limit_slices_the_benchmark_prompts(
    endpoint: Any, tiny_prompts: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], limit: str, qids: list[str]
) -> None:
    run = route(capsys, tiny_prompts, tmp_path / "live", "--limit", limit)

    assert run.status == 0, run.stderr
    assert len(endpoint.requests) == len(qids)
    assert [row["qid"] for row in read_rows(run.csv_path)] == qids
    assert run.stdout == [f"HumanEval [keyword]: {len(qids)}/{len(qids)} succeeded -> {run.csv_path}"]
