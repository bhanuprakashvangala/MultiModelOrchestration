"""The released inputs satisfy the rules that `mmorch reproduce` reads them by.

The traces are read as the reproduction reads them (gzip, UTF-8, csv.DictReader, file order) and the prompts as one
JSON object per line. The checks pin what the analysis relies on: the columns, the row counts, the values of
success and complexity, numeric timings, benchmark selection by qid prefix, and a prompt for every routed qid. Two
cross-checks tie the traces to the ported runners: the routing traces' model column is route_to_model(complexity),
and the baseline trace holds exactly one row per task that the baseline runner builds.

The tests need only the base install and skip when data/ or results/traces/ is missing, as in an sdist.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mmorch import baseline
from mmorch.data import BENCHMARKS
from mmorch.routing.runner import route_to_model

ROUTING_METHODS = ("keyword", "llm")
ROUTING_COLUMNS = [
    "benchmark",
    "qid",
    "complexity",
    "model",
    "latency_ms",
    "ttft_ms",
    "generation_time_ms",
    "tokens_per_second",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "success",
    "error",
]
BASELINE_COLUMNS = ["benchmark", "qid", "strategy", "model", "latency", "success"]
PROMPT_KEYS = {"qid", "benchmark", "question"}
TIERS = {"LOW", "MEDIUM", "HIGH"}
SUCCESS_VALUES = {"0", "1"}

PROMPT_COUNT = 31_019
BASELINE_ROWS = 155_095
NON_ASCII_PROMPTS = 2_056
# Table 1 order: the benchmarks of the baseline trace.
TABLE1_BENCHMARKS = ("HumanEval", "GSM8K", "MBPP", "TruthfulQA", "ARC", "HellaSwag", "MATH", "MMLU-Pro")


@dataclass(frozen=True)
class Trace:
    """A trace file as csv.DictReader reads it: its header and its rows in file order."""

    columns: list[str]
    rows: list[dict[str, str]]


def read_trace(path: Path) -> Trace:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return Trace(list(reader.fieldnames or ()), rows)


@pytest.fixture(scope="module")
def routing_traces(repo_root: Path) -> dict[str, Trace]:
    return {m: read_trace(repo_root / "results" / "traces" / f"routing_{m}.csv.gz") for m in ROUTING_METHODS}


@pytest.fixture(scope="module")
def baseline_trace(repo_root: Path) -> Trace:
    return read_trace(repo_root / "results" / "traces" / "baseline_strategies.csv.gz")


@pytest.fixture(scope="module")
def prompt_records(repo_root: Path) -> list[Any]:
    with gzip.open(repo_root / "data" / "prompts.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def prompt_qids(prompt_records: list[Any]) -> set[str]:
    return {record["qid"] for record in prompt_records}


def is_timing(value: str) -> bool:
    """Whether a cell converts with float(), as the reproduction converts it, to a finite value of at least 0."""
    try:
        number = float(value)
    except ValueError:
        return False
    return math.isfinite(number) and number >= 0


# ---------------------------------------------------------------- routing traces


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_has_the_13_columns_and_31019_rows(routing_traces: dict[str, Trace], method: str) -> None:
    trace = routing_traces[method]
    assert trace.columns == ROUTING_COLUMNS
    assert len(trace.rows) == PROMPT_COUNT


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_success_is_0_or_1_and_complexity_a_tier(routing_traces: dict[str, Trace], method: str) -> None:
    rows = routing_traces[method].rows
    assert {row["success"] for row in rows} <= SUCCESS_VALUES
    assert {row["complexity"] for row in rows} <= TIERS


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_timings_are_numbers(routing_traces: dict[str, Trace], method: str) -> None:
    for column in ("latency_ms", "ttft_ms"):
        bad = [(row["qid"], row[column]) for row in routing_traces[method].rows if not is_timing(row[column])]
        assert bad == [], f"{column} values that are not finite numbers >= 0"


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_qids_select_exactly_their_benchmark(routing_traces: dict[str, Trace], method: str) -> None:
    # The analysis selects a benchmark's rows by qid.startswith(benchmark + '_'), not by the benchmark column.
    wrong = [
        (row["qid"], row["benchmark"])
        for row in routing_traces[method].rows
        if [b for b in BENCHMARKS if row["qid"].startswith(b + "_")] != [row["benchmark"]]
    ]
    assert wrong == []


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_routes_every_prompt_once(
    routing_traces: dict[str, Trace], prompt_qids: set[str], method: str
) -> None:
    counts = Counter(row["qid"] for row in routing_traces[method].rows)
    assert sorted(set(counts) - prompt_qids) == [], "routed qids without a prompt in data/prompts.jsonl.gz"
    assert [qid for qid, n in counts.items() if n > 1] == []
    assert set(counts) == prompt_qids


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_every_benchmark_has_a_positive_ttft(routing_traces: dict[str, Trace], method: str) -> None:
    # The TTFT figures keep only values > 0 and take per-benchmark medians and percentiles of what remains.
    positive = [row["qid"] for row in routing_traces[method].rows if float(row["ttft_ms"]) > 0]
    assert [b for b in BENCHMARKS if not any(qid.startswith(b + "_") for qid in positive)] == []


@pytest.mark.parametrize("method", ROUTING_METHODS)
def test_routing_trace_model_is_the_tier_model_key(routing_traces: dict[str, Trace], method: str) -> None:
    wrong = [
        (row["qid"], row["complexity"], row["model"])
        for row in routing_traces[method].rows
        if row["model"] != route_to_model(row["complexity"])
    ]
    assert wrong == []


# ---------------------------------------------------------------- baseline trace


def test_baseline_trace_has_the_runner_columns_and_155095_rows(baseline_trace: Trace) -> None:
    assert baseline_trace.columns == BASELINE_COLUMNS
    assert tuple(BASELINE_COLUMNS) == baseline.FIELDS
    assert len(baseline_trace.rows) == BASELINE_ROWS
    assert {row["success"] for row in baseline_trace.rows} <= SUCCESS_VALUES
    assert {row["benchmark"] for row in baseline_trace.rows} == set(TABLE1_BENCHMARKS)


def test_baseline_trace_holds_one_row_per_runner_task(baseline_trace: Trace) -> None:
    counts = Counter((row["benchmark"], row["qid"], row["strategy"]) for row in baseline_trace.rows)
    tasks = {(b, task.qid, task.strategy) for b in TABLE1_BENCHMARKS for task in baseline.build_tasks(b)}
    assert [key for key, n in counts.items() if n > 1] == []
    assert set(counts) == tasks
    assert {row["model"] for row in baseline_trace.rows} <= set(baseline.MODELS)


# ---------------------------------------------------------------- prompts


def test_prompts_have_31019_unique_qids_with_qid_benchmark_and_question(prompt_records: list[Any]) -> None:
    assert len(prompt_records) == PROMPT_COUNT
    assert [r for r in prompt_records if not isinstance(r, dict) or set(r) != PROMPT_KEYS] == []
    assert [r for r in prompt_records if not all(isinstance(r[key], str) for key in PROMPT_KEYS)] == []
    assert len({r["qid"] for r in prompt_records}) == PROMPT_COUNT
    assert {r["benchmark"] for r in prompt_records} == set(BENCHMARKS)
    assert [r["qid"] for r in prompt_records if not r["qid"].startswith(r["benchmark"] + "_")] == []


def test_prompts_hold_2056_non_ascii_questions(prompt_records: list[Any]) -> None:
    # The file is read as UTF-8; these questions would come out wrong under any other encoding.
    assert sum(not r["question"].isascii() for r in prompt_records) == NON_ASCII_PROMPTS
