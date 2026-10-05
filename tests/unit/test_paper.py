"""Tests for mmorch.paper.analysis, .constants and .reproduce, on small synthetic datasets (no repository data).

The expected values are worked out by hand in the comments. The keyword classifier is always injected, so these
tests do not depend on the routing rules. tests/golden/reproduce_stdout.txt, the console output of the original
script, pins the claim and paper text of all 88 checks, their order and the console layout.
"""

from __future__ import annotations

import csv
import functools
import gzip
import inspect
import io
import json
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mmorch.data import BENCHMARKS
from mmorch.paper import analysis, figures
from mmorch.paper.analysis import (
    LEVELS,
    METHODS,
    TABLE1_ORDER,
    BaselineRecord,
    Check,
    RoutingRecord,
    Section,
    fig4,
    fig5,
    fig6_fig8,
    fig9,
    fig10,
    fig11,
    keyword_agreement,
    load_baseline,
    load_questions,
    load_routing_runs,
    normalize,
    per_benchmark,
    prompt_count,
    scale,
    table1,
    ttft_percentile_means,
)
from mmorch.paper.constants import FIG6, FIG10, FIG10_CHANGE_PCT, FIG11, PROMPT_COUNT, TABLE1
from mmorch.paper.reproduce import TABLE_FILES, VERIFICATION_FILE, VERIFICATION_HEADER, Report, reproduce
from mmorch.routing.classifier import classify_keyword

# The column layouts of the released traces.
ROUTING_COLUMNS = (
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
)
BASELINE_COLUMNS = ("benchmark", "qid", "strategy", "model", "latency", "success")


def record(
    qid: str, complexity: str = "MEDIUM", latency_ms: float = 1000.0, ttft_ms: float = 100.0, success: bool = True
) -> RoutingRecord:
    return RoutingRecord(qid, complexity, latency_ms, ttft_ms, success)


@pytest.fixture
def mmorch_logs(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog for the mmorch loggers at INFO, also after the command line configured them in an earlier test.

    mmorch.cli.configure_logging gives the 'mmorch' logger its own handler and turns propagation off, which would
    keep records away from caplog; both are undone for the test.
    """
    logger = logging.getLogger("mmorch")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="mmorch")
    return caplog


# ---------------------------------------------------------------- the golden console output


@pytest.fixture(scope="module")
def golden_lines(golden_dir: Path) -> list[str]:
    """tests/golden/reproduce_stdout.txt, line by line (the file is CRLF; splitlines normalises that)."""
    return (golden_dir / "reproduce_stdout.txt").read_text(encoding="utf-8").splitlines()


@pytest.fixture(scope="module")
def golden_rows(golden_lines: list[str]) -> list[tuple[str, str, str, str]]:
    """The 88 (claim, paper, reproduced, match) rows of the golden console output.

    The columns are separated by at least two spaces, and no claim or value contains two spaces in a row, so the
    split does not depend on the column widths that Report.lines() is tested for.
    """
    rows = []
    for line in golden_lines[: golden_lines.index("")]:
        claim, paper, reproduced, match = re.split(r" {2,}", line)
        rows.append((claim, paper, reproduced, match))
    return rows


def test_the_golden_output_has_88_matching_rows(golden_rows: list[tuple[str, str, str, str]]) -> None:
    assert len(golden_rows) == 88
    assert {match for *_, match in golden_rows} == {"yes"}


# ---------------------------------------------------------------- checks, normalisation and constants


def test_normalize_removes_commas_and_spaces_only() -> None:
    assert normalize("6,595 / 5,924") == "6595/5924"
    assert normalize("-44.5") == "-44.5"


def test_check_matches_after_normalisation_and_renders_yes_or_no() -> None:
    same = Check("Table 1 GSM8K", "6,595 / 5,924", "6595/5924")
    assert same.matches
    assert same.row() == ("Table 1 GSM8K", "6,595 / 5,924", "6595/5924", "yes")
    # The comparison is textual: '80' is not '80.0'.
    different = Check("Table 1 HumanEval: success (%)", "80.0", "80")
    assert not different.matches
    assert different.row() == ("Table 1 HumanEval: success (%)", "80.0", "80", "NO")


def test_constants_keep_the_types_the_paper_values_print_with() -> None:
    assert f"{TABLE1['HumanEval'][3]}" == "80.0"
    assert f"{FIG6['MBPP'][0]}" == "103.0"
    assert f"{FIG10_CHANGE_PCT['TruthfulQA']}" == "-44.5"
    assert "{} / {}".format(*FIG10["MATH"]) == "15.3 / 29.0"
    assert all(type(n) is int for n in TABLE1["GSM8K"][:3])
    assert f"{TABLE1['GSM8K'][0]:,}" == "6,595"
    assert PROMPT_COUNT == "31,019"
    assert FIG11[0] == ("P50", 45.5, 56.2)


# ---------------------------------------------------------------- records and loading


def write_gzip_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> Path:
    """Write a trace in the released format: gzip, utf-8, csv.writer (CRLF)."""
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def routing_row(
    benchmark: str, qid: str, complexity: str, latency: float, ttft: float, success: str
) -> tuple[object, ...]:
    return (benchmark, qid, complexity, "qwen3", latency, ttft, 0.0, 0.0, 128, 512, 640, success, "")


def parse_row(header: Sequence[str], line: str) -> dict[str, str]:
    return next(csv.DictReader(io.StringIO(line), fieldnames=list(header)))


def test_records_convert_the_trace_strings() -> None:
    # The first data rows of routing_keyword.csv.gz and baseline_strategies.csv.gz.
    row = parse_row(ROUTING_COLUMNS, "HumanEval,HumanEval_1,MEDIUM,qwen3,30110.17,0.0,0.0,0.0,128,512,640,1,")
    assert RoutingRecord.from_row(row) == RoutingRecord("HumanEval_1", "MEDIUM", 30110.17, 0.0, True)
    # Only '1' counts as a success.
    assert not RoutingRecord.from_row({**row, "success": "True"}).success
    assert not RoutingRecord.from_row({**row, "success": "0"}).success
    baseline = parse_row(BASELINE_COLUMNS, "ARC,ARC_2,quality,llama3-sdsc,0.0,0")
    assert BaselineRecord.from_row(baseline) == BaselineRecord("ARC", False)
    assert BaselineRecord.from_row({**baseline, "success": "1"}) == BaselineRecord("ARC", True)


def test_loaders_read_the_traces_and_prompts_in_file_order(tmp_path: Path) -> None:
    write_gzip_csv(
        tmp_path / "baseline_strategies.csv.gz",
        BASELINE_COLUMNS,
        [("ARC", "ARC_2", "quality", "llama3-sdsc", 0.0, 0), ("MBPP", "MB_0", "speed", "gemma3", 1.5, 1)],
    )
    for method, complexity in (("keyword", "LOW"), ("llm", "HIGH")):
        write_gzip_csv(
            tmp_path / f"routing_{method}.csv.gz",
            ROUTING_COLUMNS,
            [
                routing_row("MBPP", "MBPP_2", complexity, 2.5, 0.0, "0"),
                routing_row("ARC", "ARC_1", "MEDIUM", 1.0, 0.5, "1"),
            ],
        )
    prompts = tmp_path / "prompts.jsonl.gz"
    lines = [
        {"qid": "ARC_1", "benchmark": "ARC", "question": "first"},
        {"qid": "MBPP_2", "benchmark": "MBPP", "question": "Größe 証明"},
        {"qid": "ARC_1", "benchmark": "ARC", "question": "second"},
    ]
    with gzip.open(prompts, "wb") as f:
        f.write("".join(json.dumps(d, ensure_ascii=False) + "\r\n" for d in lines).encode("utf-8"))

    assert load_baseline(tmp_path) == [BaselineRecord("ARC", False), BaselineRecord("MBPP", True)]
    runs = load_routing_runs(tmp_path)
    assert list(runs) == ["keyword", "llm"]
    assert runs["keyword"] == [
        RoutingRecord("MBPP_2", "LOW", 2.5, 0.0, False),
        RoutingRecord("ARC_1", "MEDIUM", 1.0, 0.5, True),
    ]
    assert runs["llm"][0].complexity == "HIGH"
    # One entry per qid, and the last duplicate wins.
    assert load_questions(prompts) == {"ARC_1": "second", "MBPP_2": "Größe 証明"}


# ---------------------------------------------------------------- Table 1, prompts and keyword agreement


def test_table1_counts_runs_and_successes_per_benchmark() -> None:
    counts = {  # benchmark: (runs, successes)
        "HumanEval": (5, 4),
        "GSM8K": (1500, 1350),
        "MBPP": (3, 2),
        "TruthfulQA": (3, 1),
        "ARC": (2, 2),
        "HellaSwag": (2, 0),
        "MATH": (4, 1),
        "MMLU-Pro": (8, 7),
        "GPQA": (1, 1),  # not in Table 1
    }
    baseline = [BaselineRecord(b, i < ok) for b, (n, ok) in counts.items() for i in range(n)]
    baseline.reverse()  # order does not matter

    section = table1(baseline)

    assert section.table is not None
    assert section.table.filename == "table1_baseline.csv"
    assert section.table.header == ("benchmark", "runs", "success", "failures", "success_pct")
    assert section.table.rows == (
        ("HumanEval", 5, 4, 1, "80.0"),
        ("GSM8K", 1500, 1350, 150, "90.0"),
        ("MBPP", 3, 2, 1, "66.7"),
        ("TruthfulQA", 3, 1, 2, "33.3"),
        ("ARC", 2, 2, 0, "100.0"),
        ("HellaSwag", 2, 0, 2, "0.0"),
        ("MATH", 4, 1, 3, "25.0"),
        ("MMLU-Pro", 8, 7, 1, "87.5"),
    )
    # Counts are ints, never bools.
    assert all(type(cell) is int for row in section.table.rows for cell in row[1:4])
    assert len(section.checks) == 16
    assert [c.row() for c in section.checks[:4]] == [
        ("Table 1 HumanEval: runs / success / failures", "820 / 656 / 164", "5 / 4 / 1", "NO"),
        ("Table 1 HumanEval: success (%)", "80.0", "80.0", "yes"),
        ("Table 1 GSM8K: runs / success / failures", "6,595 / 5,924 / 671", "1,500 / 1,350 / 150", "NO"),
        ("Table 1 GSM8K: success (%)", "89.8", "90.0", "NO"),
    ]
    assert [c.claim for c in section.checks[::2]] == [f"Table 1 {b}: runs / success / failures" for b in TABLE1_ORDER]


def test_prompt_count_formats_with_thousands_separators() -> None:
    section = prompt_count({f"q{i}": "?" for i in range(1234)})
    assert section == Section((Check("Prompts across 8 benchmarks", "31,019", "1,234"),))


def test_keyword_agreement_counts_trace_rows_with_the_injected_classifier() -> None:
    questions = {"q1": "alpha", "q2": "beta", "q3": "gamma"}
    keyword_run = [record("q1", "LOW"), record("q2", "HIGH"), record("q3", "MEDIUM"), record("q1", "MEDIUM")]
    seen: list[str] = []

    def classify(question: str) -> str:
        seen.append(question)
        return {"alpha": "LOW", "beta": "MEDIUM", "gamma": "MEDIUM"}[question]

    # q1 agrees, q2 does not, q3 agrees, and the second q1 row does not: rows are counted, not prompts.
    same = keyword_agreement(questions, keyword_run, classify)
    assert same == 2
    assert type(same) is int
    assert seen == ["alpha", "beta", "gamma", "alpha"]


def test_keyword_agreement_defaults_to_the_routing_keyword_rules() -> None:
    assert inspect.signature(keyword_agreement).parameters["classify"].default is classify_keyword


# ---------------------------------------------------------------- Figs. 4 and 5


def small_runs() -> dict[str, list[RoutingRecord]]:
    """keyword: LOW, LOW, MEDIUM (failed), HIGH; llm: LOW, MEDIUM x3 (one failed), HIGH (failed), MEDIUM."""
    return {
        "keyword": [record("a", "LOW"), record("b", "LOW"), record("c", "MEDIUM", success=False), record("d", "HIGH")],
        "llm": [
            record("a", "LOW"),
            record("b", "MEDIUM"),
            record("c", "MEDIUM", success=False),
            record("d", "HIGH", success=False),
            record("e", "MEDIUM"),
            record("f", "HIGH"),
        ],
    }


def test_fig4_counts_and_shares_per_method_and_tier() -> None:
    section = fig4(small_runs())
    assert section.table is not None
    assert section.table.filename == "fig4_complexity_distribution.csv"
    assert section.table.header == ("routing", "complexity", "queries", "pct")
    assert section.table.rows == (
        ("Keyword", "LOW", 2, "50.0"),
        ("Keyword", "MEDIUM", 1, "25.0"),
        ("Keyword", "HIGH", 1, "25.0"),
        ("LLM prompt", "LOW", 1, "16.7"),
        ("LLM prompt", "MEDIUM", 3, "50.0"),
        ("LLM prompt", "HIGH", 2, "33.3"),
    )
    assert [c.row() for c in section.checks[:2]] == [
        ("Fig. 4 Keyword LOW: count", "6,961", "2", "NO"),
        ("Fig. 4 Keyword LOW: share (%)", "22.4", "50.0", "NO"),
    ]
    assert [c.claim for c in section.checks[6:8]] == [
        "Fig. 4 LLM prompt LOW: count",
        "Fig. 4 LLM prompt LOW: share (%)",
    ]
    assert section.checks[-1].paper == "1.1"
    assert len(section.checks) == 12


def test_fig5_success_rates_and_the_overall_rate_for_fig9() -> None:
    section, success = fig5(small_runs())
    assert section.table is not None
    assert section.table.filename == "fig5_success_by_complexity.csv"
    assert section.table.header == ("routing", "complexity", "success_pct")
    assert section.table.rows == (
        ("Keyword", "LOW", "100.0"),
        ("Keyword", "MEDIUM", "0.0"),
        ("Keyword", "HIGH", "100.0"),
        ("Keyword", "Overall", "75.0"),
        ("LLM prompt", "LOW", "100.0"),
        ("LLM prompt", "MEDIUM", "66.7"),
        ("LLM prompt", "HIGH", "50.0"),
        ("LLM prompt", "Overall", "66.7"),
    )
    assert section.checks[3].row() == ("Fig. 5 Keyword success, Overall (%)", "98.0", "75.0", "NO")
    assert section.checks[4].row() == ("Fig. 5 LLM prompt success, LOW (%)", "100.0", "100.0", "yes")
    assert len(section.checks) == 8
    # 100 * sum / len, which differs in the last bit from 100 * np.mean (the Fig. 5 bars) for 4 of 6.
    assert success == {"keyword": 75.0, "llm": 66.66666666666667}
    assert 100 * np.mean([True] * 4 + [False] * 2) == 66.66666666666666


# ---------------------------------------------------------------- Figs. 6, 8, 10 and 11


def test_per_benchmark_groups_by_qid_prefix_and_converts_to_seconds() -> None:
    records = [
        record("HumanEval_1", latency_ms=30110.17, ttft_ms=0.0),
        record("HumanEval_2", latency_ms=2500.0, ttft_ms=500.0),
        record("HumanEvalPlus_1", latency_ms=9000.0, ttft_ms=900.0),  # not HumanEval: the prefix is 'HumanEval_'
        record("MBPP_1", latency_ms=0.0, ttft_ms=0.0, success=False),
        record("MBPP_2", latency_ms=3000.0, ttft_ms=-1.0),
    ]
    latency = per_benchmark(records, "latency_ms")
    assert list(latency) == list(BENCHMARKS)
    np.testing.assert_array_equal(latency["HumanEval"], [30110.17 / 1000, 2.5])
    # Division, as in the original script: multiplying by 0.001 gives another float for this value.
    assert latency["HumanEval"][0] != 30110.17 * 0.001
    np.testing.assert_array_equal(latency["MBPP"], [0.0, 3.0])  # a failed request's 0 counts
    assert all(latency[b].size == 0 for b in BENCHMARKS[2:])
    assert all(a.dtype == np.float64 for a in latency.values())

    np.testing.assert_array_equal(per_benchmark(records, "ttft_ms")["HumanEval"], [0.0, 0.5])
    positive = per_benchmark(records, "ttft_ms", positive_only=True)
    np.testing.assert_array_equal(positive["HumanEval"], [0.5])
    assert positive["MBPP"].size == 0


def by_benchmark(values: dict[str, list[float]], default: list[float]) -> dict[str, np.ndarray[Any, Any]]:
    return {b: np.array(values.get(b, default)) for b in BENCHMARKS}


def test_fig6_fig8_medians_overheads_and_their_means() -> None:
    latency = {
        "keyword": by_benchmark({"HumanEval": [40.0, 60.0, 50.0], "MBPP": [100.0]}, [10.0]),
        "llm": by_benchmark({"HumanEval": [80.0, 70.0], "MBPP": [110.0]}, [10.0]),
    }
    section = fig6_fig8(latency)
    assert section.table is not None
    assert section.table.filename == "fig6_fig8_median_latency.csv"
    assert section.table.header == (
        "benchmark",
        "keyword_median_s",
        "llm_prompt_median_s",
        "overhead_s",
        "overhead_pct",
    )
    assert section.table.rows[:3] == (
        ("HumanEval", "50.0", "75.0", "25.0", "50.0"),
        ("MBPP", "100.0", "110.0", "10.0", "10.0"),
        ("GSM8K", "10.0", "10.0", "0.0", "0.0"),
    )
    assert len(section.checks) == 26
    assert [c.row() for c in section.checks[:3]] == [
        ("Fig. 6 HumanEval median latency, keyword / LLM prompt (s)", "57.1 / 110.5", "50.0 / 75.0", "NO"),
        ("Fig. 6/8 HumanEval LLM-prompt latency overhead (%)", "93.7", "50.0", "NO"),
        ("Fig. 8 HumanEval LLM-prompt latency overhead (s)", "53.5", "25.0", "NO"),
    ]
    assert section.checks[3].paper == "103.0 / 110.4"
    # Means over the 8 benchmarks: (25 + 10) / 8 = 4.375 s and (50 + 10) / 8 = 7.5 %.
    assert [c.row() for c in section.checks[-2:]] == [
        ("Fig. 8 average absolute overhead (s)", "18.7", "4.4", "NO"),
        ("Fig. 8 average relative overhead (%)", "30.6", "7.5", "NO"),
    ]


def test_fig10_median_ttft_and_change() -> None:
    ttft = {
        "keyword": by_benchmark({"HumanEval": [2.0, 4.0], "MBPP": [10.0]}, [1.0]),
        "llm": by_benchmark({"HumanEval": [6.0], "MBPP": [5.0]}, [1.0]),
    }
    section = fig10(ttft)
    assert section.table is not None
    assert section.table.filename == "fig10_median_ttft.csv"
    assert section.table.header == ("benchmark", "keyword_s", "llm_prompt_s", "change_pct")
    assert section.table.rows[:3] == (
        ("HumanEval", "3.0", "6.0", "100.0"),
        ("MBPP", "10.0", "5.0", "-50.0"),
        ("GSM8K", "1.0", "1.0", "0.0"),
    )
    assert len(section.checks) == 16
    assert [c.row() for c in section.checks[:2]] == [
        ("Fig. 10 HumanEval median TTFT, keyword / LLM prompt (s)", "25.4 / 87.2", "3.0 / 6.0", "NO"),
        ("Fig. 10 HumanEval TTFT change (%)", "242.9", "100.0", "NO"),
    ]
    assert section.checks[6].row() == (
        "Fig. 10 MATH median TTFT, keyword / LLM prompt (s)",
        "15.3 / 29.0",
        "1.0 / 1.0",
        "NO",
    )


def test_ttft_percentile_means_average_each_percentile_over_the_benchmarks() -> None:
    # Benchmark i holds 1..5 + i (keyword) and twice that (llm): P50 = 3 + i, P95 = 4.8 + i, P99 = 4.96 + i, and
    # the means over i = 0..7 add 3.5.
    ttft = {
        "keyword": {b: np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + i for i, b in enumerate(BENCHMARKS)},
        "llm": {b: 2 * (np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + i) for i, b in enumerate(BENCHMARKS)},
    }
    pct = ttft_percentile_means(ttft)
    assert list(pct) == ["keyword", "llm"]
    assert pct["keyword"] == pytest.approx([6.5, 8.3, 8.46])
    assert pct["llm"] == pytest.approx([13.0, 16.6, 16.92])
    assert all(isinstance(v, np.float64) for v in pct["keyword"])  # numpy scalars, as before


def test_fig11_rows_and_checks() -> None:
    section = fig11({"keyword": [6.5, 8.3, 8.46], "llm": [13.0, 16.6, 16.92]})
    assert section.table is not None
    assert section.table.filename == "fig11_ttft_percentiles.csv"
    assert section.table.header == ("routing", "p50_s", "p95_s", "p99_s")
    assert section.table.rows == (("Keyword", "6.5", "8.3", "8.5"), ("LLM prompt", "13.0", "16.6", "16.9"))
    assert [c.row() for c in section.checks] == [
        ("Fig. 11 TTFT P50, keyword / LLM prompt (s)", "45.5 / 56.2", "6.5 / 13.0", "NO"),
        ("Fig. 11 TTFT P95, keyword / LLM prompt (s)", "95.4 / 111.4", "8.3 / 16.6", "NO"),
        ("Fig. 11 TTFT P99, keyword / LLM prompt (s)", "106.7 / 117.9", "8.5 / 16.9", "NO"),
        # 100 * (13.0 - 6.5) / 6.5
        ("Median TTFT increase with LLM-prompt routing (%)", "23.5", "100.0", "NO"),
    ]


# ---------------------------------------------------------------- Fig. 9


def test_scale_maps_to_0_10_inverts_and_clamps() -> None:
    assert scale(95, 90, 100) == 5.0
    assert scale(50, 40, 80, True) == 7.5
    assert scale(30, 40, 80, True) == 10  # faster than the best end of the range
    assert scale(90, 40, 80, True) == 0
    assert scale(105, 90, 100) == 10
    assert scale(85, 90, 100) == 0


def test_fig9_raw_values_and_scores() -> None:
    runs = {
        # 0, 40, 60 and 80 s: a failed request's 0 counts. Median 50, P95 60 + 0.85 * 20 = 77, mean 45.
        "keyword": [record(f"k{i}", latency_ms=ms) for i, ms in enumerate([0.0, 40000.0, 60000.0, 80000.0])],
        # 50, 70, 90 and 130 s: median 80, P95 90 + 0.85 * 40 = 124, mean 85.
        "llm": [record(f"l{i}", latency_ms=ms) for i, ms in enumerate([50000.0, 70000.0, 90000.0, 130000.0])],
    }
    section = fig9(runs, {"keyword": 95.0, "llm": 92.0})
    assert section.table is not None
    assert section.table.filename == "fig9_multi_metric.csv"
    assert section.table.header == (
        "routing",
        "success_pct",
        "median_latency_s",
        "p95_latency_s",
        "mean_latency_s",
        "score_success",
        "score_speed",
        "score_p95",
        "score_mean",
    )
    # keyword scores: (95 - 90) / 10 * 10 = 5; (80 - 50) / 40 * 10 = 7.5; (140 - 77) / 40 * 10 = 15.75 -> 10;
    # (75 - 45) / 30 * 10 = 10. llm: 2; 0; (140 - 124) / 40 * 10 = 4; (75 - 85) / 30 * 10 < 0 -> 0.
    assert section.table.rows == (
        ("Keyword", "95.0", "50.0", "77.0", "45.0", "5.0", "7.5", "10.0", "10.0"),
        ("LLM prompt", "92.0", "80.0", "124.0", "85.0", "2.0", "0.0", "4.0", "0.0"),
    )
    assert [c.row() for c in section.checks] == [
        ("Fig. 9 response speed (median latency), keyword / LLM prompt (s)", "48.9 / 65.4", "50.0 / 80.0", "NO"),
        ("Fig. 9 P95 latency, keyword / LLM prompt (s)", "117.5 / 119.8", "77.0 / 124.0", "NO"),
        ("Fig. 9 mean latency, keyword / LLM prompt (s)", "55.4 / 65.1", "45.0 / 85.0", "NO"),
        ("Fig. 9 scores success/speed/P95/mean, keyword", "8.0 / 7.8 / 5.6 / 6.5", "5.0 / 7.5 / 10.0 / 10.0", "NO"),
        ("Fig. 9 scores success/speed/P95/mean, LLM prompt", "6.0 / 3.6 / 5.0 / 3.3", "2.0 / 0.0 / 4.0 / 0.0", "NO"),
    ]


# ---------------------------------------------------------------- reproduce() and the report


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Callable[[str], str]]:
    """Traces and prompts in the released format, plus a keyword classifier that agrees with the keyword trace.

    Each routing trace has three prompts per benchmark, one per tier, with TTFT 0 for the LOW one. The baseline
    trace has 2 runs per Table 1 benchmark, one of them successful. One extra prompt is in no trace.
    """
    traces = tmp_path / "traces"
    traces.mkdir()
    write_gzip_csv(
        traces / "baseline_strategies.csv.gz",
        BASELINE_COLUMNS,
        [(b, f"{b}_{i}", "balanced", "llama3", 1.0, i) for b in TABLE1_ORDER for i in range(2)],
    )
    for m, extra in (("keyword", 0), ("llm", 10)):
        rows = [
            routing_row(
                b,
                f"{b}_{j}",
                lv,
                1000.0 * (40 + 5 * i + j + extra),
                0.0 if j == 0 else 1000.0 * (10 + i + j + extra),
                "1",
            )
            for i, b in enumerate(BENCHMARKS)
            for j, lv in enumerate(LEVELS)
        ]
        write_gzip_csv(traces / f"routing_{m}.csv.gz", ROUTING_COLUMNS, rows)
    prompts = tmp_path / "prompts.jsonl.gz"
    qids = [f"{b}_{j}" for b in BENCHMARKS for j in range(3)] + ["GPQA_0"]
    with gzip.open(prompts, "wt", encoding="utf-8") as f:
        f.writelines(
            json.dumps({"qid": q, "benchmark": q.split("_")[0], "question": f"question {q}"}) + "\n" for q in qids
        )

    def classify(question: str) -> str:
        return LEVELS[int(question.rsplit("_", 1)[1])]

    return traces, prompts, classify


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace figures.render_figures (tested in test_figures.py) with a recorder that writes empty files."""
    calls: list[dict[str, Any]] = []

    def render(runs: Any, latency: Any, ttft: Any, pct: Any, figures_dir: Path) -> list[Path]:
        calls.append({"runs": runs, "latency": latency, "ttft": ttft, "pct": pct, "figures_dir": figures_dir})
        figures_dir.mkdir(parents=True, exist_ok=True)
        paths = [figures_dir / name for name in figures.FIGURE_FILES]
        for path in paths:
            path.touch()
        return paths

    monkeypatch.setattr(figures, "render_figures", render)
    return calls


def run_reproduce(
    monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, Callable[[str], str]], out_dir: Path
) -> Report:
    traces, prompts, classify = inputs
    monkeypatch.setattr(analysis, "keyword_agreement", functools.partial(analysis.keyword_agreement, classify=classify))
    return reproduce(traces, prompts, out_dir)


def test_reproduce_checks_have_the_golden_claims_and_paper_values_in_order(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_inputs: tuple[Path, Path, Callable[[str], str]],
    fake_render: list[dict[str, Any]],
    golden_rows: list[tuple[str, str, str, str]],
    tmp_path: Path,
) -> None:
    report = run_reproduce(monkeypatch, synthetic_inputs, tmp_path / "out")
    # Claims and paper values do not depend on the data, so they must equal the golden run's, row for row.
    assert [(c.claim, c.paper) for c in report.checks] == [(claim, paper) for claim, paper, _, _ in golden_rows]


def test_reproduce_writes_tables_figures_and_verification_in_order(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_inputs: tuple[Path, Path, Callable[[str], str]],
    fake_render: list[dict[str, Any]],
    mmorch_logs: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    out = tmp_path / "nested" / "out"
    report = run_reproduce(monkeypatch, synthetic_inputs, out)

    figures_dir = out / "figures"
    assert report.out_dir == out
    assert report.written == (
        *(out / name for name in TABLE_FILES),
        *(figures_dir / name for name in figures.FIGURE_FILES),
        out / VERIFICATION_FILE,
    )
    assert sorted(p.name for p in out.iterdir()) == sorted([*TABLE_FILES, VERIFICATION_FILE, "figures"])
    # Every CSV is CRLF, as csv.writer writes it.
    for path in report.written[:7] + report.written[-1:]:
        data = path.read_bytes()
        assert data.endswith(b"\r\n")
        assert data.count(b"\n") == data.count(b"\r\n")
    with open(out / VERIFICATION_FILE, newline="", encoding="utf-8") as f:
        verification = list(csv.reader(f))
    assert verification == [list(VERIFICATION_HEADER), *(list(c.row()) for c in report.checks)]
    with open(out / "table1_baseline.csv", newline="", encoding="utf-8") as f:
        assert list(csv.reader(f))[:2] == [
            ["benchmark", "runs", "success", "failures", "success_pct"],
            ["HumanEval", "2", "1", "1", "50.0"],
        ]

    # The figures get the analysis inputs and their own directory.
    (call,) = fake_render
    assert call["figures_dir"] == figures_dir
    runs = load_routing_runs(synthetic_inputs[0])
    assert call["runs"] == runs
    assert list(call["latency"]) == list(call["ttft"]) == list(METHODS)
    for m in METHODS:
        for b in BENCHMARKS:
            np.testing.assert_array_equal(call["latency"][m][b], per_benchmark(runs[m], "latency_ms")[b])
            np.testing.assert_array_equal(call["ttft"][m][b], per_benchmark(runs[m], "ttft_ms", positive_only=True)[b])
    assert call["pct"] == ttft_percentile_means(call["ttft"])

    # 24 keyword rows, all reproduced by the injected classifier; the synthetic numbers do not match the paper.
    assert (report.keyword_same, report.keyword_total) == (24, 24)
    assert report.checks[16] == Check("Prompts across 8 benchmarks", "31,019", "25")
    assert not report.ok
    lines = report.lines()
    assert len(lines) == 92
    assert lines[-4:] == [
        "",
        f"{report.matched}/88 numbers match the paper.",
        "Keyword classifier re-run on the prompts agrees with the recorded tiers for 100.0% of prompts.",
        f"Wrote tables and figures to {out}/",
    ]

    # Progress is logged, nothing is printed.
    assert capsys.readouterr().out == ""
    assert {r.name for r in mmorch_logs.records} == {"mmorch.paper.reproduce"}
    assert all(r.levelno == logging.INFO for r in mmorch_logs.records)


def golden_report(golden_rows: list[tuple[str, str, str, str]], keyword_same: int = 31019) -> Report:
    checks = tuple(Check(claim, paper, reproduced) for claim, paper, reproduced, _ in golden_rows)
    return Report(checks, keyword_same, 31019, Path("results"), ())


def test_report_lines_equal_the_golden_console_output(
    golden_rows: list[tuple[str, str, str, str]], golden_lines: list[str]
) -> None:
    report = golden_report(golden_rows)
    assert report.lines() == golden_lines
    assert (report.matched, report.keyword_agreement, report.ok) == (88, 100.0, True)


def test_report_is_not_ok_with_one_mismatch(golden_rows: list[tuple[str, str, str, str]]) -> None:
    report = golden_report(golden_rows)
    last = report.checks[-1]
    flipped = Report(
        (*report.checks[:-1], Check(last.claim, last.paper, "6.0 / 3.6 / 5.0 / 3.4")), 31019, 31019, Path("results"), ()
    )
    assert flipped.matched == 87
    assert not flipped.ok
    lines = flipped.lines()
    assert lines[87].endswith("  NO")
    assert lines[89] == "87/88 numbers match the paper."


def test_report_is_not_ok_with_one_disagreeing_tier(golden_rows: list[tuple[str, str, str, str]]) -> None:
    report = golden_report(golden_rows, keyword_same=31018)
    assert report.matched == 88
    assert not report.ok
    # 31,018 of 31,019 still prints as 100.0%, so ok is stricter than the printed share.
    assert report.lines()[90].endswith("for 100.0% of prompts.")
