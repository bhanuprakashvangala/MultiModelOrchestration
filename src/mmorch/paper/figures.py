"""Renders the reproduction's five PNGs with matplotlib's object-oriented API.

Each figure is a matplotlib.figure.Figure with fig.subplots(). Creation, tight_layout() and savefig(path,
dpi=200) all happen inside one `with matplotlib.style.context('default'):` block, one figure at a time, in the
original order: Figs. 4, 5, 6, 10 and 11. The style context makes the output independent of the caller's
rcParams and any matplotlibrc, and restores them afterwards. There is no pyplot, no matplotlib.use() and no other
global state, and on the pinned stack (constraints/reproduce.txt) the files are byte-identical to
results/figures. Every drawing call is the original pyplot-era call, made on the Axes in the same order.

matplotlib is imported inside render_figures; without it, figures are skipped with a warning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from mmorch.data import BENCHMARKS
from mmorch.paper.analysis import LEVELS, METHODS, RoutingRecord

if TYPE_CHECKING:
    from matplotlib.axes import Axes

log = logging.getLogger(__name__)

COLORS: Final = {"keyword": "#4C72B0", "llm": "#DD8452"}
BAR_WIDTH: Final = 0.38
DPI: Final = 200
FIGURE_FILES: Final = (
    "fig4_complexity_distribution.png",
    "fig5_success_rate.png",
    "fig6_median_latency.png",
    "fig10_median_ttft.png",
    "fig11_ttft_percentiles.png",
)


def render_figures(
    runs: Mapping[str, Sequence[RoutingRecord]],
    latency: Mapping[str, Mapping[str, npt.NDArray[np.float64]]],
    ttft: Mapping[str, Mapping[str, npt.NDArray[np.float64]]],
    pct: Mapping[str, Sequence[float]],
    figures_dir: Path,
) -> list[Path]:
    """Write the five figures into figures_dir and return their paths in FIGURE_FILES order.

    The inputs are those of the analysis: the routing records per method, per_benchmark latency and TTFT (s)
    per method, and ttft_percentile_means. figures_dir is created if needed. Without matplotlib, a warning is
    logged and nothing is written ([] is returned).
    """
    try:
        # This import checks the matplotlib package itself; `from matplotlib.figure import Figure` alone can
        # succeed from the module cache even when sys.modules blocks the package.
        import matplotlib.style
        from matplotlib.figure import Figure
    except ImportError:
        log.warning("matplotlib not installed; skipping figures")
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    panels: tuple[tuple[tuple[float, float], Callable[[Axes], None]], ...] = (
        ((6, 4), partial(_complexity_distribution, runs=runs)),
        ((6, 4), partial(_success_rate, runs=runs)),
        ((8, 4), partial(_median_latency, latency=latency)),
        ((8, 4), partial(_median_ttft, ttft=ttft)),
        ((6, 4), partial(_ttft_percentiles, pct=pct)),
    )
    paths: list[Path] = []
    with matplotlib.style.context("default"):
        for name, (figsize, draw) in zip(FIGURE_FILES, panels):
            fig = Figure(figsize=figsize)
            draw(fig.subplots())
            fig.tight_layout()
            path = figures_dir / name
            fig.savefig(path, dpi=DPI)
            log.debug("Wrote %s", path)
            paths.append(path)
    return paths


def _bars(ax: Axes, values: Mapping[str, npt.ArrayLike], fmt: str) -> None:
    """Draw one bar per benchmark and method, side by side, labelled with fmt, plus the ticks and legend."""
    x = np.arange(len(BENCHMARKS))
    for i, m in enumerate(METHODS):
        b = ax.bar(x + (i - 0.5) * BAR_WIDTH, values[m], BAR_WIDTH, label=METHODS[m], color=COLORS[m])
        ax.bar_label(b, fmt=fmt, fontsize=7)
    ax.set_xticks(x, BENCHMARKS, rotation=30, ha="right")
    ax.legend()


def _complexity_distribution(ax: Axes, *, runs: Mapping[str, Sequence[RoutingRecord]]) -> None:
    """Fig. 4: the number of queries per tier and method."""
    for i, m in enumerate(METHODS):
        cnt = [sum(r.complexity == lv for r in runs[m]) for lv in LEVELS]
        b = ax.bar(np.arange(3) + (i - 0.5) * BAR_WIDTH, cnt, BAR_WIDTH, label=METHODS[m], color=COLORS[m])
        ax.bar_label(b, fmt="{:,.0f}", fontsize=8)
    ax.set_xticks(range(3), LEVELS)
    ax.set_ylabel("Queries")
    ax.set_title("Fig. 4: complexity distribution (31,019 queries)")
    ax.legend()


def _success_rate(ax: Axes, *, runs: Mapping[str, Sequence[RoutingRecord]]) -> None:
    """Fig. 5: the success rate (%) per tier and overall, as 100 * np.mean of the success flags."""
    for i, m in enumerate(METHODS):
        vals = [100 * np.mean([r.success for r in runs[m] if r.complexity == lv]) for lv in LEVELS]
        vals.append(100 * np.mean([r.success for r in runs[m]]))
        b = ax.bar(np.arange(4) + (i - 0.5) * BAR_WIDTH, vals, BAR_WIDTH, label=METHODS[m], color=COLORS[m])
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(range(4), [*LEVELS, "Overall"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fig. 5: success rate by complexity")
    ax.legend(loc="lower right")


def _median_latency(ax: Axes, *, latency: Mapping[str, Mapping[str, npt.NDArray[np.float64]]]) -> None:
    """Fig. 6: the median latency (s) per benchmark."""
    _bars(ax, {m: [np.median(latency[m][b]) for b in BENCHMARKS] for m in METHODS}, "%.1f")
    ax.set_ylabel("Median latency (s)")
    ax.set_title("Fig. 6: median latency per benchmark")


def _median_ttft(ax: Axes, *, ttft: Mapping[str, Mapping[str, npt.NDArray[np.float64]]]) -> None:
    """Fig. 10: the median time to first token (s) per benchmark."""
    _bars(ax, {m: [np.median(ttft[m][b]) for b in BENCHMARKS] for m in METHODS}, "%.1f")
    ax.set_ylabel("Median TTFT (s)")
    ax.set_title("Fig. 10: median time to first token per benchmark")


def _ttft_percentiles(ax: Axes, *, pct: Mapping[str, Sequence[float]]) -> None:
    """Fig. 11: the P50, P95 and P99 TTFT (s), each the mean over the benchmarks."""
    for i, m in enumerate(METHODS):
        b = ax.bar(np.arange(3) + (i - 0.5) * BAR_WIDTH, pct[m], BAR_WIDTH, label=METHODS[m], color=COLORS[m])
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(range(3), ["P50", "P95", "P99"])
    ax.set_ylabel("TTFT (s), mean over 8 benchmarks")
    ax.set_title("Fig. 11: TTFT percentiles")
    ax.legend()
