"""The reproduction's computation: pure, numpy only, with no file writes and no matplotlib.

- Loads the traces into typed records, in file order.
- One function per section returns that section's CSV table (cells already formatted) and its checks, in the
  original order. The sections are Table 1, the prompt count, keyword agreement, and Figs. 4, 5, 6/8, 10, 11 and 9.
- Every arithmetic expression, aggregation order and format spec is copied from the original reproduction
  script, because floating point is not associative and the published numbers are compared digit for digit.
  For example, unit conversion is np.array(values) / 1000 (multiplying by 0.001 gives different results) and
  the success rates of the tables are 100 * sum / len, which can differ in the last bit from the
  100 * np.mean(...) that the Fig. 5 bars use.

Numbers are kept as the types the script had: medians, percentiles and means stay numpy float64 scalars and are
only ever formatted with an explicit format spec. numpy float64 subclasses float, so where one is passed on as a
float the code uses typing.cast, which changes nothing at run time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from mmorch.data import BENCHMARKS, iter_prompts, read_csv_gz
from mmorch.paper.constants import (
    FIG4,
    FIG5,
    FIG6,
    FIG8,
    FIG8_MEAN_ABS_S,
    FIG8_MEAN_REL_PCT,
    FIG9_MEAN,
    FIG9_MEDIAN,
    FIG9_P95,
    FIG9_SCORES,
    FIG10,
    FIG10_CHANGE_PCT,
    FIG11,
    PROMPT_COUNT,
    TABLE1,
    TTFT_P50_INCREASE_PCT,
)
from mmorch.routing.classifier import classify_keyword

# The two tier classifiers, keyed as in the trace file names: keyword rules and an LLM prompt.
METHODS: Final[Mapping[str, str]] = {"keyword": "Keyword", "llm": "LLM prompt"}
LEVELS: Final = ("LOW", "MEDIUM", "HIGH")
# Table 1 lists the benchmarks in its own order; everything else uses mmorch.data.BENCHMARKS.
TABLE1_ORDER: Final = ("HumanEval", "GSM8K", "MBPP", "TruthfulQA", "ARC", "HellaSwag", "MATH", "MMLU-Pro")
TRACE_FILES: Final[Mapping[str, str]] = {
    "baseline": "baseline_strategies.csv.gz",
    "keyword": "routing_keyword.csv.gz",
    "llm": "routing_llm.csv.gz",
}


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    """The fields of one routing-trace row that the reproduction uses."""

    qid: str
    complexity: str
    latency_ms: float
    ttft_ms: float
    success: bool

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> RoutingRecord:
        """Convert a trace row: success is row['success'] == '1', the timings go through float()."""
        return cls(
            qid=row["qid"],
            complexity=row["complexity"],
            latency_ms=float(row["latency_ms"]),
            ttft_ms=float(row["ttft_ms"]),
            success=row["success"] == "1",
        )


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    """The fields of one baseline-trace row that Table 1 uses."""

    benchmark: str
    success: bool

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> BaselineRecord:
        """Convert a trace row: success is row['success'] == '1'."""
        return cls(benchmark=row["benchmark"], success=row["success"] == "1")


@dataclass(frozen=True, slots=True)
class Check:
    """One comparison of a value printed in the paper with the reproduced value, both as text."""

    claim: str
    paper: str
    reproduced: str

    @property
    def matches(self) -> bool:
        """Whether the two values are equal once commas and spaces are removed."""
        return normalize(self.paper) == normalize(self.reproduced)

    def row(self) -> tuple[str, str, str, str]:
        """Return the verification.csv row: claim, paper, reproduced and 'yes' or 'NO'."""
        return (self.claim, self.paper, self.reproduced, "yes" if self.matches else "NO")


@dataclass(frozen=True, slots=True)
class Table:
    """The CSV table of one section, with its cells already formatted."""

    filename: str
    header: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class Section:
    """The checks of one section of the paper, and its table if it has one."""

    checks: tuple[Check, ...]
    table: Table | None = None


def normalize(value: str) -> str:
    """Remove commas and spaces, so that '6,595 / 5,924' compares equal to '6595/5924'."""
    return value.replace(",", "").replace(" ", "")


# ---------------------------------------------------------------- loading


def load_baseline(traces_dir: Path) -> list[BaselineRecord]:
    """Load the five-strategy baseline trace, in file order."""
    return [BaselineRecord.from_row(row) for row in read_csv_gz(traces_dir / TRACE_FILES["baseline"])]


def load_routing_runs(traces_dir: Path) -> dict[str, list[RoutingRecord]]:
    """Load both routing traces in file order, keyed by method in METHODS order (keyword, then llm)."""
    return {m: [RoutingRecord.from_row(row) for row in read_csv_gz(traces_dir / TRACE_FILES[m])] for m in METHODS}


def load_questions(prompts_path: Path) -> dict[str, str]:
    """Return {qid: question} for the prompts file; for a repeated qid the last one wins."""
    return {p.qid: p.question for p in iter_prompts(prompts_path)}


# ---------------------------------------------------------------- Table 1 and the prompts


def table1(baseline: Sequence[BaselineRecord]) -> Section:
    """Table 1, the five-strategy baseline runs: 16 checks and table1_baseline.csv."""
    t1: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in baseline:
        t1[r.benchmark][0] += 1
        t1[r.benchmark][1] += r.success
    rows: list[tuple[object, ...]] = []
    checks: list[Check] = []
    for b in TABLE1_ORDER:
        n, ok = t1[b]
        rows.append((b, n, ok, n - ok, f"{100 * ok / n:.1f}"))
        pn, pok, pf, pp = TABLE1[b]
        checks.append(
            Check(
                f"Table 1 {b}: runs / success / failures",
                f"{pn:,} / {pok:,} / {pf:,}",
                f"{n:,} / {ok:,} / {n - ok:,}",
            )
        )
        checks.append(Check(f"Table 1 {b}: success (%)", f"{pp}", f"{100 * ok / n:.1f}"))
    header = ("benchmark", "runs", "success", "failures", "success_pct")
    return Section(tuple(checks), Table("table1_baseline.csv", header, tuple(rows)))


def prompt_count(questions: Mapping[str, str]) -> Section:
    """The number of distinct prompts: one check and no table."""
    return Section((Check("Prompts across 8 benchmarks", PROMPT_COUNT, f"{len(questions):,}"),))


def keyword_agreement(
    questions: Mapping[str, str],
    keyword_run: Sequence[RoutingRecord],
    classify: Callable[[str], str] = classify_keyword,
) -> int:
    """Count the keyword-trace rows whose recorded tier classify gives again for the row's question.

    classify defaults to the routing keyword rules, mmorch.routing.classifier.classify_keyword. The count runs
    over the trace rows, so the share is taken of the number of rows, not of distinct prompts.
    """
    return sum(classify(questions[r.qid]) == r.complexity for r in keyword_run)


# ---------------------------------------------------------------- Figs. 4 and 5


def fig4(runs: Mapping[str, Sequence[RoutingRecord]]) -> Section:
    """Fig. 4, the complexity distribution: 12 checks and fig4_complexity_distribution.csv.

    runs is keyed by method in METHODS order, as load_routing_runs returns it.
    """
    rows: list[tuple[object, ...]] = []
    checks: list[Check] = []
    for m, rs in runs.items():
        cnt = [sum(r.complexity == lv for r in rs) for lv in LEVELS]
        for lv, c, pc, pp in zip(LEVELS, cnt, *FIG4[m]):
            rows.append((METHODS[m], lv, c, f"{100 * c / len(rs):.1f}"))
            checks.append(Check(f"Fig. 4 {METHODS[m]} {lv}: count", f"{pc:,}", f"{c:,}"))
            checks.append(Check(f"Fig. 4 {METHODS[m]} {lv}: share (%)", f"{pp}", f"{100 * c / len(rs):.1f}"))
    header = ("routing", "complexity", "queries", "pct")
    return Section(tuple(checks), Table("fig4_complexity_distribution.csv", header, tuple(rows)))


def fig5(runs: Mapping[str, Sequence[RoutingRecord]]) -> tuple[Section, dict[str, float]]:
    """Fig. 5, the success rate by complexity: 8 checks and fig5_success_by_complexity.csv.

    Also returns each method's overall success (%), 100 * sum / len, which Fig. 9 reuses. runs is keyed by
    method in METHODS order, as load_routing_runs returns it.
    """
    rows: list[tuple[object, ...]] = []
    checks: list[Check] = []
    succ: dict[str, float] = {}
    for m, rs in runs.items():
        vals: list[float] = []
        for lv in LEVELS:
            sub = [r for r in rs if r.complexity == lv]
            vals.append(100 * sum(r.success for r in sub) / len(sub))
        vals.append(100 * sum(r.success for r in rs) / len(rs))
        succ[m] = vals[-1]
        for lv, v, p in zip((*LEVELS, "Overall"), vals, FIG5[m]):
            rows.append((METHODS[m], lv, f"{v:.1f}"))
            checks.append(Check(f"Fig. 5 {METHODS[m]} success, {lv} (%)", f"{p}", f"{v:.1f}"))
    header = ("routing", "complexity", "success_pct")
    return Section(tuple(checks), Table("fig5_success_by_complexity.csv", header, tuple(rows))), succ


# ---------------------------------------------------------------- Figs. 6, 8, 10 and 11


def per_benchmark(
    records: Sequence[RoutingRecord],
    field: Literal["latency_ms", "ttft_ms"],
    *,
    positive_only: bool = False,
) -> dict[str, npt.NDArray[np.float64]]:
    """Return one timing field in seconds per benchmark, in BENCHMARKS order.

    A record belongs to benchmark b when its qid starts with b + '_'. With positive_only, values that are not
    greater than zero are left out (a TTFT of 0 means no token arrived); otherwise every record counts,
    including failed requests, whose latency is recorded as 0.
    """
    out: dict[str, npt.NDArray[np.float64]] = {}
    for b in BENCHMARKS:
        v = [
            getattr(r, field)
            for r in records
            if r.qid.startswith(b + "_") and (getattr(r, field) > 0 or not positive_only)
        ]
        out[b] = np.array(v) / 1000
    return out


def fig6_fig8(latency: Mapping[str, Mapping[str, npt.NDArray[np.float64]]]) -> Section:
    """Figs. 6 and 8, the median latency and the LLM-prompt overhead: 26 checks and fig6_fig8_median_latency.csv.

    latency maps each method to per_benchmark(records, 'latency_ms').
    """
    rows: list[tuple[object, ...]] = []
    checks: list[Check] = []
    abs_over: list[np.floating[Any]] = []
    rel_over: list[np.floating[Any]] = []
    for b in BENCHMARKS:
        kw, llm = np.median(latency["keyword"][b]), np.median(latency["llm"][b])
        d, pct = llm - kw, 100 * (llm - kw) / kw
        abs_over.append(d)
        rel_over.append(pct)
        rows.append((b, f"{kw:.1f}", f"{llm:.1f}", f"{d:.1f}", f"{pct:.1f}"))
        pk, pl, pp = FIG6[b]
        checks.append(
            Check(f"Fig. 6 {b} median latency, keyword / LLM prompt (s)", f"{pk} / {pl}", f"{kw:.1f} / {llm:.1f}")
        )
        checks.append(Check(f"Fig. 6/8 {b} LLM-prompt latency overhead (%)", f"{pp}", f"{pct:.1f}"))
        checks.append(Check(f"Fig. 8 {b} LLM-prompt latency overhead (s)", f"{FIG8[b]}", f"{d:.1f}"))
    checks.append(Check("Fig. 8 average absolute overhead (s)", FIG8_MEAN_ABS_S, f"{np.mean(abs_over):.1f}"))
    checks.append(Check("Fig. 8 average relative overhead (%)", FIG8_MEAN_REL_PCT, f"{np.mean(rel_over):.1f}"))
    header = ("benchmark", "keyword_median_s", "llm_prompt_median_s", "overhead_s", "overhead_pct")
    return Section(tuple(checks), Table("fig6_fig8_median_latency.csv", header, tuple(rows)))


def fig10(ttft: Mapping[str, Mapping[str, npt.NDArray[np.float64]]]) -> Section:
    """Fig. 10, the median time to first token: 16 checks and fig10_median_ttft.csv.

    ttft maps each method to per_benchmark(records, 'ttft_ms', positive_only=True).
    """
    rows: list[tuple[object, ...]] = []
    checks: list[Check] = []
    for b in BENCHMARKS:
        kw, llm = np.median(ttft["keyword"][b]), np.median(ttft["llm"][b])
        rows.append((b, f"{kw:.1f}", f"{llm:.1f}", f"{100 * (llm - kw) / kw:.1f}"))
        checks.append(
            Check(
                f"Fig. 10 {b} median TTFT, keyword / LLM prompt (s)",
                "{} / {}".format(*FIG10[b]),
                f"{kw:.1f} / {llm:.1f}",
            )
        )
        checks.append(Check(f"Fig. 10 {b} TTFT change (%)", f"{FIG10_CHANGE_PCT[b]}", f"{100 * (llm - kw) / kw:.1f}"))
    header = ("benchmark", "keyword_s", "llm_prompt_s", "change_pct")
    return Section(tuple(checks), Table("fig10_median_ttft.csv", header, tuple(rows)))


def ttft_percentile_means(ttft: Mapping[str, Mapping[str, npt.NDArray[np.float64]]]) -> dict[str, list[float]]:
    """Return the P50, P95 and P99 TTFT (s) per method, each the mean over the benchmarks of that percentile."""
    return {
        m: [cast(float, np.mean([np.percentile(ttft[m][b], p) for b in BENCHMARKS])) for p in (50, 95, 99)]
        for m in METHODS
    }


def fig11(pct: Mapping[str, Sequence[float]]) -> Section:
    """Fig. 11, the TTFT percentiles: 4 checks (P50, P95, P99 and the median TTFT increase) and the CSV.

    pct is what ttft_percentile_means returns.
    """
    rows = tuple((METHODS[m], *(f"{v:.1f}" for v in pct[m])) for m in METHODS)
    checks = [
        Check(
            f"Fig. 11 TTFT {name}, keyword / LLM prompt (s)",
            f"{pk} / {pl}",
            f"{pct['keyword'][i]:.1f} / {pct['llm'][i]:.1f}",
        )
        for i, (name, pk, pl) in enumerate(FIG11)
    ]
    checks.append(
        Check(
            "Median TTFT increase with LLM-prompt routing (%)",
            TTFT_P50_INCREASE_PCT,
            f"{100 * (pct['llm'][0] - pct['keyword'][0]) / pct['keyword'][0]:.1f}",
        )
    )
    header = ("routing", "p50_s", "p95_s", "p99_s")
    return Section(tuple(checks), Table("fig11_ttft_percentiles.csv", header, rows))


# ---------------------------------------------------------------- Fig. 9


def scale(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """Map x from [lo, hi] onto a 0-10 score (10 at hi, or at lo when inverted), clamped to [0, 10]."""
    s = (hi - x) / (hi - lo) * 10 if invert else (x - lo) / (hi - lo) * 10
    return min(10, max(0, s))


@dataclass(frozen=True, slots=True)
class _Overall:
    """One method's raw Fig. 9 values: success (%) and the median, P95 and mean latency (s) over all rows."""

    success: float
    median: float
    p95: float
    mean: float


def _overall(records: Sequence[RoutingRecord], success: float) -> _Overall:
    """Summarise the latency (s) of every row, failed requests included, next to the given success (%)."""
    v = np.array([r.latency_ms for r in records]) / 1000
    return _Overall(success, cast(float, np.median(v)), cast(float, np.percentile(v, 95)), v.mean())


def fig9(runs: Mapping[str, Sequence[RoutingRecord]], success: Mapping[str, float]) -> Section:
    """Fig. 9, the multi-metric radar: 5 checks and fig9_multi_metric.csv with the raw values and 0-10 scores.

    success is the overall success (%) per method that fig5 returns.
    """
    ov = {m: _overall(runs[m], success[m]) for m in METHODS}
    rows: list[tuple[str, ...]] = []
    for m in METHODS:
        o = ov[m]
        scores = [
            scale(o.success, 90, 100),
            scale(o.median, 40, 80, True),
            scale(o.p95, 100, 140, True),
            scale(o.mean, 45, 75, True),
        ]
        cells = [f"{o.success:.1f}", f"{o.median:.1f}", f"{o.p95:.1f}", f"{o.mean:.1f}", *(f"{s:.1f}" for s in scores)]
        rows.append((METHODS[m], *cells))
    k, llm = ov["keyword"], ov["llm"]
    checks = (
        Check(
            "Fig. 9 response speed (median latency), keyword / LLM prompt (s)",
            FIG9_MEDIAN,
            f"{k.median:.1f} / {llm.median:.1f}",
        ),
        Check("Fig. 9 P95 latency, keyword / LLM prompt (s)", FIG9_P95, f"{k.p95:.1f} / {llm.p95:.1f}"),
        Check("Fig. 9 mean latency, keyword / LLM prompt (s)", FIG9_MEAN, f"{k.mean:.1f} / {llm.mean:.1f}"),
        Check("Fig. 9 scores success/speed/P95/mean, keyword", FIG9_SCORES["keyword"], " / ".join(rows[0][5:])),
        Check("Fig. 9 scores success/speed/P95/mean, LLM prompt", FIG9_SCORES["llm"], " / ".join(rows[1][5:])),
    )
    header = (
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
    return Section(checks, Table("fig9_multi_metric.csv", header, tuple(rows)))
