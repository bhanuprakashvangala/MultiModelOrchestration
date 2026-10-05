"""Tests for mmorch.paper.figures on a small synthetic dataset (no repository data).

The PNG bytes depend on the plotting stack, so these tests check file names, the PNG signature and pixel sizes,
the isolation from the caller's matplotlib settings, and the absence of pyplot. The byte-for-byte comparison
with results/figures runs in the integration tests on the pinned stack.

The figures are rendered twice in all: once here under changed matplotlib settings, and once in a fresh process
with the defaults. Equal bytes show that the caller's settings never reach the figures.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import pytest
from matplotlib.figure import Figure

from mmorch.data import BENCHMARKS
from mmorch.paper import figures
from mmorch.paper.analysis import LEVELS, METHODS, RoutingRecord, per_benchmark, ttft_percentile_means
from mmorch.paper.figures import FIGURE_FILES, render_figures

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# (width, height) in pixels: figsize (6, 4) or (8, 4) inches at 200 dpi, as in the committed figures.
SIZES = {
    "fig4_complexity_distribution.png": (1200, 800),
    "fig5_success_rate.png": (1200, 800),
    "fig6_median_latency.png": (1600, 800),
    "fig10_median_ttft.png": (1600, 800),
    "fig11_ttft_percentiles.png": (1200, 800),
}
# Settings a caller might have; the figures must not change with them.
CALLER_SETTINGS = {
    "font.size": 20.0,
    "axes.facecolor": "black",
    "figure.dpi": 50.0,
    "axes.prop_cycle": matplotlib.cycler(color=["red"]),
}


def synthetic_runs() -> dict[str, list[RoutingRecord]]:
    """Both methods over all 8 benchmarks and 3 tiers, with some failures and TTFT > 0 for two of three prompts."""
    return {
        m: [
            RoutingRecord(
                f"{b}_{j}",
                lv,
                1000.0 * (40 + 5 * i + j + extra),
                0.0 if j == 0 else 1000.0 * (10 + i + j + extra),
                (i + j) % 4 != 1,
            )
            for i, b in enumerate(BENCHMARKS)
            for j, lv in enumerate(LEVELS)
        ]
        for m, extra in (("keyword", 0), ("llm", 10))
    }


def render(runs: Mapping[str, list[RoutingRecord]], figures_dir: Path) -> list[Path]:
    """Render the figures from routing records, with the other inputs computed as the reproduction does."""
    latency = {m: per_benchmark(runs[m], "latency_ms") for m in METHODS}
    ttft = {m: per_benchmark(runs[m], "ttft_ms", positive_only=True) for m in METHODS}
    return render_figures(runs, latency, ttft, ttft_percentile_means(ttft), figures_dir)


class Rendered(NamedTuple):
    paths: list[Path]
    figures_dir: Path
    settings_before: dict[str, Any]
    settings_after: dict[str, Any]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> Rendered:
    """The five figures rendered once under CALLER_SETTINGS, with the settings just before and after."""
    figures_dir = tmp_path_factory.mktemp("rendered") / "out" / "figures"
    with matplotlib.rc_context(CALLER_SETTINGS):
        # copy() does not resolve the backend, so it cannot import pyplot.
        before = matplotlib.rcParams.copy()
        paths = render(synthetic_runs(), figures_dir)
        after = matplotlib.rcParams.copy()
    return Rendered(paths, figures_dir, dict(before), dict(after))


def png_size(data: bytes) -> tuple[int, int]:
    """The width and height from the IHDR chunk, which follows the signature."""
    assert data[12:16] == b"IHDR"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_render_figures_writes_the_five_pngs(rendered: Rendered) -> None:
    # The directory is created with its parents, and the paths come back in FIGURE_FILES order.
    assert rendered.paths == [rendered.figures_dir / name for name in FIGURE_FILES]
    assert sorted(p.name for p in rendered.figures_dir.iterdir()) == sorted(FIGURE_FILES)
    for path in rendered.paths:
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE)
        assert png_size(data) == SIZES[path.name]


def test_the_style_context_restores_the_callers_settings(rendered: Rendered) -> None:
    assert rendered.settings_after == rendered.settings_before
    assert rendered.settings_after["font.size"] == 20.0


RENDER_IN_A_FRESH_PROCESS = textwrap.dedent(
    """
    import json
    import sys
    from pathlib import Path

    from mmorch.paper.analysis import METHODS, RoutingRecord, per_benchmark, ttft_percentile_means
    from mmorch.paper.figures import render_figures

    imported_early = "matplotlib" in sys.modules
    rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    runs = {m: [RoutingRecord(*row) for row in rows[m]] for m in METHODS}
    latency = {m: per_benchmark(runs[m], "latency_ms") for m in METHODS}
    ttft = {m: per_benchmark(runs[m], "ttft_ms", positive_only=True) for m in METHODS}
    paths = render_figures(runs, latency, ttft, ttft_percentile_means(ttft), Path(sys.argv[2]))
    print(imported_early, len(paths), "matplotlib.pyplot" in sys.modules)
    """
)


def test_a_fresh_process_renders_the_same_bytes_without_pyplot(rendered: Rendered, tmp_path: Path) -> None:
    runs = tmp_path / "runs.json"
    runs.write_text(
        json.dumps({m: [dataclasses.astuple(r) for r in rs] for m, rs in synthetic_runs().items()}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-c", RENDER_IN_A_FRESH_PROCESS, str(runs), str(tmp_path / "figures")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # matplotlib loads only once rendering starts, and pyplot never.
    assert result.stdout.split() == ["False", "5", "False"]
    # The fresh process has matplotlib's defaults, the fixture rendered under CALLER_SETTINGS.
    for expected in rendered.paths:
        assert (tmp_path / "figures" / expected.name).read_bytes() == expected.read_bytes(), expected.name


def test_without_matplotlib_figures_are_skipped_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The command line may have configured the 'mmorch' logger in an earlier test: own handler, no propagation.
    monkeypatch.setattr(logging.getLogger("mmorch"), "handlers", [])
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.WARNING, logger="mmorch")
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    assert render(synthetic_runs(), tmp_path / "figures") == []

    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("mmorch.paper.figures", logging.WARNING, "matplotlib not installed; skipping figures")
    ]
    assert not (tmp_path / "figures").exists()


# ---------------------------------------------------------------- the bars of single figures


def test_fig4_labels_counts_with_thousands_separators() -> None:
    runs = {"keyword": [RoutingRecord(f"q{i}", "LOW", 1.0, 1.0, True) for i in range(1234)], "llm": []}
    ax = Figure().subplots()
    figures._complexity_distribution(ax, runs=runs)
    assert [p.get_height() for p in ax.patches] == [1234, 0, 0, 0, 0, 0]
    assert [t.get_text() for t in ax.texts] == ["1,234", "0", "0", "0", "0", "0"]
    assert [t.get_text() for t in ax.get_xticklabels()] == list(LEVELS)


def test_fig5_bars_use_numpy_mean_unlike_the_tables() -> None:
    # MEDIUM: 2 of 3 succeed. The bar is 100 * np.mean([True, True, False]) = 66.66666666666666, while Fig. 5's
    # table and checks use 100 * 2 / 3 = 66.66666666666667.
    records = [
        RoutingRecord("a", "LOW", 1.0, 1.0, True),
        RoutingRecord("b", "MEDIUM", 1.0, 1.0, True),
        RoutingRecord("c", "MEDIUM", 1.0, 1.0, True),
        RoutingRecord("d", "MEDIUM", 1.0, 1.0, False),
        RoutingRecord("e", "HIGH", 1.0, 1.0, False),
    ]
    ax = Figure().subplots()
    figures._success_rate(ax, runs={"keyword": records, "llm": records})
    heights = [p.get_height() for p in ax.patches]
    assert heights[:4] == [100.0, 66.66666666666666, 0.0, 60.0]
    assert heights[4:] == heights[:4]
    assert [t.get_text() for t in ax.texts[:4]] == ["100.0", "66.7", "0.0", "60.0"]
    assert ax.get_ylim() == (0.0, 110.0)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["Keyword", "LLM prompt"]
