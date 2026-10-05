"""The one function `mmorch reproduce` calls, and the only place that fixes section, check and write order.

reproduce():

1. loads the traces and prompts;
2. computes Table 1 -> prompt count -> keyword agreement -> Fig. 4 -> Fig. 5 -> Figs. 6/8 -> Fig. 10 ->
   Fig. 11 -> Fig. 9;
3. writes the seven table CSVs in that order;
4. renders the five figures into <out>/figures;
5. writes verification.csv;
6. returns a Report whose lines() are the console output of the original script.

Progress goes to this module's logger at INFO; nothing is printed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from mmorch import data
from mmorch.paper import analysis, figures
from mmorch.paper.analysis import Check

log = logging.getLogger(__name__)

TABLE_FILES: Final = (
    "table1_baseline.csv",
    "fig4_complexity_distribution.csv",
    "fig5_success_by_complexity.csv",
    "fig6_fig8_median_latency.csv",
    "fig10_median_ttft.csv",
    "fig11_ttft_percentiles.csv",
    "fig9_multi_metric.csv",
)
VERIFICATION_FILE: Final = "verification.csv"
VERIFICATION_HEADER: Final = ("claim", "paper", "reproduced", "match")


@dataclass(frozen=True)
class Report:
    """The outcome of a reproduction: every check, the keyword agreement and the files written, in write order."""

    checks: tuple[Check, ...]
    keyword_same: int
    keyword_total: int
    out_dir: Path
    written: tuple[Path, ...]

    @property
    def matched(self) -> int:
        """The number of checks whose reproduced value matches the paper."""
        return sum(c.matches for c in self.checks)

    @property
    def keyword_agreement(self) -> float:
        """The share (%) of keyword-trace rows whose recorded tier the keyword rules give again."""
        return 100 * self.keyword_same / self.keyword_total

    @property
    def ok(self) -> bool:
        """Whether every check matches and the keyword rules reproduce every recorded tier.

        This is stricter than the printed share: one disagreeing row still prints 100.0% but is not ok.
        """
        return self.matched == len(self.checks) and self.keyword_same == self.keyword_total

    def lines(self) -> list[str]:
        """Return the console report: one aligned row per check, a blank line and three summary lines."""
        rows = [c.row() for c in self.checks]
        width = max(len(claim) for claim, _, _, _ in rows)
        return [
            *(f"{c:<{width}}  {p:>28}  {r:>28}  {ok}" for c, p, r, ok in rows),
            "",
            f"{self.matched}/{len(self.checks)} numbers match the paper.",
            "Keyword classifier re-run on the prompts agrees with the recorded tiers for "
            f"{self.keyword_agreement:.1f}% of prompts.",
            f"Wrote tables and figures to {self.out_dir}/",
        ]


def reproduce(traces_dir: Path, prompts_path: Path, out_dir: Path) -> Report:
    """Regenerate the tables, figures and verification.csv into out_dir and return the report.

    traces_dir holds the three trace files named in analysis.TRACE_FILES and prompts_path is the prompts file.
    out_dir is created if needed; the figures go into out_dir/figures. Existing files are overwritten.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Reading the traces in %s", traces_dir)
    baseline = analysis.load_baseline(traces_dir)
    runs = analysis.load_routing_runs(traces_dir)
    log.info("Reading the prompts in %s", prompts_path)
    questions = analysis.load_questions(prompts_path)

    table1 = analysis.table1(baseline)
    prompts = analysis.prompt_count(questions)
    keyword_same = analysis.keyword_agreement(questions, runs["keyword"])
    fig4 = analysis.fig4(runs)
    fig5, success = analysis.fig5(runs)
    latency = {m: analysis.per_benchmark(runs[m], "latency_ms") for m in analysis.METHODS}
    fig6_fig8 = analysis.fig6_fig8(latency)
    ttft = {m: analysis.per_benchmark(runs[m], "ttft_ms", positive_only=True) for m in analysis.METHODS}
    fig10 = analysis.fig10(ttft)
    pct = analysis.ttft_percentile_means(ttft)
    fig11 = analysis.fig11(pct)
    fig9 = analysis.fig9(runs, success)
    sections = (table1, prompts, fig4, fig5, fig6_fig8, fig10, fig11, fig9)

    log.info("Writing the tables and figures to %s", out_dir)
    tables = [s.table for s in sections if s.table is not None]
    written = [data.write_csv(out_dir / t.filename, t.header, t.rows) for t in tables]
    written += figures.render_figures(runs, latency, ttft, pct, out_dir / "figures")
    checks = tuple(c for s in sections for c in s.checks)
    written.append(data.write_csv(out_dir / VERIFICATION_FILE, VERIFICATION_HEADER, [c.row() for c in checks]))
    return Report(checks, keyword_same, len(runs["keyword"]), out_dir, tuple(written))
