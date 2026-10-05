"""`mmorch baseline` end to end against the local OpenAI-compatible stub (the llm_stub fixture of tests/conftest.py).

HumanEval is 164 questions x 5 strategies = 820 requests. The tests pin the raw request bodies captured from the
v1.0.0 runner, the Authorization header without a key, the strategy-to-model assignment through this process's
string hash, and the 6-column CSV, for successful and for failed requests.

They need the 'live' extra (requests) and skip without it.
"""

from __future__ import annotations

import ast
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mmorch.cli import main

pytest.importorskip("requests")

# strategy_baseline.py at v1.0.0: the served models, the strategies in task order and the HumanEval questions.
MODELS = ("gemma3", "llama3-sdsc", "llama3")
STRATEGIES = ("balanced", "quality", "speed", "cost", "baseline")
QIDS = tuple(f"HE_{i}" for i in range(164))
HEADER = b"benchmark,qid,strategy,model,latency,success"


def legacy_model(strategy: str) -> str:
    """The legacy assignment, MODELS[hash(strategy) % 3], with this process's (randomized) string hash."""
    return MODELS[hash(strategy) % len(MODELS)]


def legacy_body(model: str) -> bytes:
    """The raw request body the legacy runner sent, as captured from it (requests' json= serialisation)."""
    return (
        f'{{"model": "{model}", "messages": [{{"role": "user", "content": "Write a Python function"}}], '
        '"max_tokens": 150}'
    ).encode()


@dataclass(frozen=True)
class Run:
    """One `mmorch baseline HumanEval` run: exit status, stdout lines, stderr text and its CSV."""

    status: int
    stdout: list[str]
    stderr: str
    csv_path: Path


@pytest.fixture(autouse=True)
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty directory, so that no default path can reach the repository's results/."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def endpoint(llm_stub: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The stub as LLM_API_BASE, given with a trailing '/' that the runner strips; LLM_API_KEY stays unset."""
    monkeypatch.setenv("LLM_API_BASE", llm_stub.base_url + "/")
    return llm_stub


def run_baseline(capsys: pytest.CaptureFixture[str], out: Path) -> Run:
    status = main(["baseline", "HumanEval", "--out", str(out)])
    captured = capsys.readouterr()
    return Run(status, captured.out.splitlines(), captured.err, out / "baseline" / "HumanEval_baseline.csv")


def read_csv(path: Path) -> tuple[bytes, list[dict[str, str]]]:
    """The raw bytes of a CSV and its rows."""
    with path.open(newline="", encoding="utf-8") as f:
        return path.read_bytes(), list(csv.DictReader(f))


def test_baseline_sends_the_legacy_requests_and_writes_one_row_each(
    endpoint: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_baseline(capsys, tmp_path / "live")

    assert run.status == 0, run.stderr
    requests = endpoint.requests
    assert len(requests) == len(QIDS) * len(STRATEGIES) == 820
    assert {request.path for request in requests} == {"/v1/chat/completions"}
    # Byte for byte, and once per question for each strategy's model.
    expected = Counter(legacy_body(legacy_model(strategy)) for _ in QIDS for strategy in STRATEGIES)
    assert Counter(request.body for request in requests) == expected
    # 'Bearer ' with the unset key; the server may drop the trailing space.
    assert {request.headers["Authorization"].strip() for request in requests} == {"Bearer"}

    raw, rows = read_csv(run.csv_path)
    assert raw.split(b"\r\n", 1)[0] == HEADER
    assert raw.count(b"\r\n") == raw.count(b"\n") == 1 + 820
    assert Counter((row["qid"], row["strategy"]) for row in rows) == Counter(
        (qid, strategy) for qid in QIDS for strategy in STRATEGIES
    )
    assert {row["benchmark"] for row in rows} == {"HumanEval"}
    # Each strategy keeps one model for the whole run.
    assert {(row["strategy"], row["model"]) for row in rows} == {(s, legacy_model(s)) for s in STRATEGIES}
    assert {row["success"] for row in rows} == {"True"}
    # Unrounded float milliseconds, written as Python renders a float.
    assert [row["latency"] for row in rows if not isinstance(ast.literal_eval(row["latency"]), float)] == []
    assert run.stdout == [f"HumanEval: 820/820 succeeded -> {run.csv_path}"]


def test_baseline_records_failed_requests(endpoint: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    endpoint.status = 500

    run = run_baseline(capsys, tmp_path / "live")

    assert run.status == 0, run.stderr  # failed requests are data, not an error
    assert len(endpoint.requests) == 820  # requests does not retry
    _, rows = read_csv(run.csv_path)
    assert len(rows) == 820
    assert {(row["success"], row["latency"]) for row in rows} == {("False", "0")}
    # A status other than 200 is not an exception, so nothing is logged for it.
    assert "Error " not in run.stderr
    assert run.stdout == [f"HumanEval: 0/820 succeeded -> {run.csv_path}"]
