"""`mmorch score` end to end against the local OpenAI-compatible stub (the llm_stub fixture of tests/conftest.py).

The demo routes three queries, each with its own strategy, and runs them non-streaming. The tests pin the request
bodies, the demo report on stdout line by line as the v1.0.0 script printed it (only the measured latencies vary)
and the router's three initialisation lines, which now go to stderr through logging.

Two more runs use --config with a copy of the packaged YAML: one takes the endpoint from the YAML's api.base_url and
sends the key 'none'; the other turns on LLM complexity detection, which, faced with an unparseable reply, warns
and falls back to the keyword rules, twice per query, as before.

They need the 'live' extra (openai and pyyaml) and skip without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from mmorch.cli import main
from mmorch.scoring.scorer import complexity_prompt

pytest.importorskip("openai")
pytest.importorskip("yaml")

# The demo of multi_objective.py at v1.0.0: query and strategy, in run order.
QUERIES = (
    ("What is the capital of France?", "speed"),
    ("Explain quantum computing in detail", "quality"),
    ("What are common treatments for diabetes?", "balanced"),
)
# The completion requests: model and max_tokens follow from the scores and the keyword complexity.
DEMO_BODIES = [
    {"messages": [{"role": "user", "content": query}], "model": model, "max_tokens": max_tokens, "temperature": 0.7}
    for (query, _), model, max_tokens in zip(QUERIES, ("gemma3", "qwen3", "gemma3"), (100, 300, 100))
]

SEPARATOR = "=" * 80
LATENCY_LINE = re.compile(r"> Latency: \d+ms")
LATENCY_PLACEHOLDER = "> Latency: <n>ms"
# The legacy demo report without its three initialisation lines, which went between the banner and the first test.
DEMO_REPORT = [
    "",
    SEPARATOR,
    "PICK-AND-SPIN ROUTING SYSTEM TEST",
    SEPARATOR,
    "",
    "",
    "[Test 1] Strategy: speed",
    "Query: What is the capital of France?",
    "",
    "> Model: gemma3",
    "> Complexity: simple",
    "> Privacy: False",
    LATENCY_PLACEHOLDER,
    "> Tokens: 6",
    "> Rationale: Selected gemma3 (27B, small-tier) for speed strategy. This model offers fastest response "
    "(speed score: 0.90). Query complexity is low, small model sufficient. Alternatives considered: llama3, qwen3.",
    "",
    "[Test 2] Strategy: quality",
    "Query: Explain quantum computing in detail",
    "",
    "> Model: qwen3",
    "> Complexity: medium",
    "> Privacy: False",
    LATENCY_PLACEHOLDER,
    "> Tokens: 6",
    "> Rationale: Selected qwen3 (235B, large-tier) for quality strategy. This model offers highest quality "
    "(score: 0.90). Alternatives considered: llama3, gemma3.",
    "",
    "[Test 3] Strategy: balanced",
    "Query: What are common treatments for diabetes?",
    "",
    "> Model: gemma3",
    "> Complexity: simple",
    "> Privacy: True",
    LATENCY_PLACEHOLDER,
    "> Tokens: 6",
    "> Rationale: Selected gemma3 (27B, small-tier) for balanced strategy. This model balances quality, speed, and "
    "cost (combined: 0.83). Query complexity is low, small model sufficient. Alternatives considered: llama3, qwen3.",
    "",
    SEPARATOR,
    "Total queries: 3",
    "Model usage: {'qwen3': 1, 'gemma3': 2, 'llama3': 0}",
]
INIT_LINES = [
    "> Pick-and-Spin Router initialized",
    "> Available models: ['qwen3', 'gemma3', 'llama3']",
    "> Available strategies: ['quality', 'cost', 'speed', 'balanced', 'baseline']",
]
PACKAGED_BASE_URL = "https://your-llm-endpoint.example.org/v1"


@dataclass(frozen=True)
class Run:
    """One `mmorch score` run: exit status, stdout lines and stderr lines."""

    status: int
    stdout: list[str]
    stderr: list[str]


# ---------------------------------------------------------------- fixtures and helpers


@pytest.fixture(autouse=True)
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run from an empty directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def stub(llm_stub: Any) -> Any:
    """The stub, answering every completion with 'Paris.' (usage 5/1/6)."""
    llm_stub.reply = "Paris."
    return llm_stub


def score(capsys: pytest.CaptureFixture[str], *options: str) -> Run:
    status = main(["score", *options])
    captured = capsys.readouterr()
    return Run(status, captured.out.splitlines(), captured.err.splitlines())


def write_config(path: Path, *, base_url: str | None = None, use_llm: bool = False) -> Path:
    """Write a copy of the packaged scorer YAML, as text so that its order and values stay verbatim."""
    text = resources.files("mmorch.scoring").joinpath("config.yaml").read_text(encoding="utf-8")
    edits = {}
    if base_url is not None:
        edits[f'base_url: "{PACKAGED_BASE_URL}"'] = f'base_url: "{base_url}"'
    if use_llm:
        edits["use_llm: false"] = "use_llm: true"
    for old, new in edits.items():
        assert text.count(old) == 1, f"the packaged config.yaml no longer has {old!r}"
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return path


def canonical(bodies: list[Any]) -> list[str]:
    """Request bodies as sorted-key JSON: equal for equal bodies, and still telling 0.0 from 0."""
    return [json.dumps(body, sort_keys=True) for body in bodies]


def report(stdout: list[str]) -> list[str]:
    """stdout with each measured latency replaced by a placeholder."""
    return [LATENCY_PLACEHOLDER if LATENCY_LINE.fullmatch(line) else line for line in stdout]


# ---------------------------------------------------------------- tests


def test_score_runs_the_legacy_demo(
    stub: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_API_BASE", stub.base_url)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    run = score(capsys)

    assert run.status == 0, run.stderr
    assert canonical(stub.json_bodies()) == canonical(DEMO_BODIES)
    assert [request.path for request in stub.requests] == ["/v1/chat/completions"] * 3
    assert {request.headers["Authorization"] for request in stub.requests} == {"Bearer test-key"}
    assert report(run.stdout) == DEMO_REPORT
    assert [line for line in run.stderr if line.startswith("> ")] == INIT_LINES


def test_score_takes_the_endpoint_from_the_config_and_the_key_none(
    stub: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Neither LLM_API_BASE nor LLM_API_KEY is set.
    config = write_config(tmp_path / "scorer.yaml", base_url=stub.base_url)

    run = score(capsys, "--config", str(config))

    assert run.status == 0, run.stderr
    assert canonical(stub.json_bodies()) == canonical(DEMO_BODIES)
    assert {request.headers["Authorization"] for request in stub.requests} == {"Bearer none"}
    assert report(run.stdout) == DEMO_REPORT


def test_score_llm_complexity_falls_back_to_keywords_twice_per_query(
    stub: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LLM_API_BASE", stub.base_url)
    config = write_config(tmp_path / "scorer.yaml", use_llm=True)

    run = score(capsys, "--config", str(config))

    assert run.status == 0, run.stderr
    # select_model and then execute_query each ask for the complexity before the completion request.
    expected: list[Any] = []
    for (query, _), completion in zip(QUERIES, DEMO_BODIES):
        classify = {
            "messages": [{"role": "user", "content": complexity_prompt(query)}],
            "model": "gemma3",
            "max_tokens": 10,
            "temperature": 0.0,
        }
        expected += [classify, classify, completion]
    assert canonical(stub.json_bodies()) == canonical(expected)
    # 'Paris.' names no complexity, so each detection warns and the keyword rules decide, as without the LLM.
    warning = "  [Warning] LLM returned unparseable complexity: 'PARIS.', falling back to keywords"
    assert [line for line in run.stderr if "[Warning]" in line] == [warning] * 6
    assert report(run.stdout) == DEMO_REPORT
