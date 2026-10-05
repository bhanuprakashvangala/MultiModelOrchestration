"""End-to-end `mmorch reproduce` on the released traces, checked against the goldens of the v1.0.0 reproduction.

One run of the command writes into a temporary directory and is shared by most tests here; it must print the
golden console output and write the eight CSVs byte for byte (tests/golden/reproduce_hashes.txt). A CSV that
differs is shown as a unified diff against its committed copy in results/. The five PNGs must exist everywhere,
but their bytes depend on the plotting stack: they are compared only on the reference stack (win32, CPython 3.12
and the pins of constraints/reproduce.txt), or anywhere with MMORCH_STRICT_FIGURES=1.

A second run uses a copy of the traces with one recorded keyword tier changed, which must fail the reproduction.

Nothing is ever written into the repository's results/. The tests need only the base install, and they skip when
data/ or results/traces/ is missing, as in an sdist.
"""

from __future__ import annotations

import contextlib
import csv
import difflib
import gzip
import hashlib
import io
import logging
import os
import platform
import shutil
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pytest

from mmorch import cli
from mmorch.paper import reproduce as paper_reproduce
from mmorch.paper.reproduce import Report

# The 13 outputs by path relative to the output directory: the eight CSVs and the five figures, each in write order.
CSV_FILES = (
    "table1_baseline.csv",
    "fig4_complexity_distribution.csv",
    "fig5_success_by_complexity.csv",
    "fig6_fig8_median_latency.csv",
    "fig10_median_ttft.csv",
    "fig11_ttft_percentiles.csv",
    "fig9_multi_metric.csv",
    "verification.csv",
)
PNG_FILES = (
    "figures/fig4_complexity_distribution.png",
    "figures/fig5_success_rate.png",
    "figures/fig6_median_latency.png",
    "figures/fig10_median_ttft.png",
    "figures/fig11_ttft_percentiles.png",
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHECK_COUNT = 88
KEYWORD_ROWS = 31_019
FULL_KEYWORD_AGREEMENT = (
    "Keyword classifier re-run on the prompts agrees with the recorded tiers for 100.0% of prompts."
)

# The packages whose versions decide the PNG bytes, compared with their pins in constraints/reproduce.txt.
REFERENCE_PACKAGES = ("numpy", "matplotlib", "pillow")
STRICT_FIGURES_VARIABLE = "MMORCH_STRICT_FIGURES"


# ---------------------------------------------------------------- running the command


@dataclass(frozen=True)
class Run:
    """The outcome of one `mmorch` command: exit status, stdout lines, stderr text and the output directory."""

    status: int
    stdout: list[str]
    stderr: str
    out_dir: Path


@contextlib.contextmanager
def restored_mmorch_logger() -> Iterator[None]:
    """Undo what main() does to the 'mmorch' logger, so that its handler and propagate=False do not leak.

    main() installs a handler on the stderr of the moment and stops propagation; left in place, that handler would
    write into a closed buffer and hide later tests' records from caplog, which listens on the root logger.
    """
    logger = logging.getLogger("mmorch")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    try:
        yield
    finally:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def run_cli(argv: Sequence[str], out_dir: Path) -> Run:
    """Run mmorch.cli.main(argv) in-process, capturing stdout and stderr (the logging handler writes to stderr)."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with restored_mmorch_logger(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = cli.main(list(argv))
    return Run(status, stdout.getvalue().splitlines(), stderr.getvalue(), out_dir)


@pytest.fixture(scope="module")
def reproduction(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    """One `mmorch --root <repo> reproduce --out results`, run from an empty temporary directory.

    The relative --out makes the last console line 'Wrote tables and figures to results/', exactly as in the golden
    output of the original script, while the files land in the temporary directory.
    """
    workdir = tmp_path_factory.mktemp("reproduce")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(workdir)
        return run_cli(["--root", str(repo_root), "reproduce", "--out", "results"], workdir / "results")


# ---------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def visible_lines(data: bytes) -> list[str]:
    """Split bytes into lines, keeping each line's terminator spelled out, so a CRLF/LF difference shows in a diff."""
    text = data.decode("utf-8", errors="replace")
    return [line.replace("\r", "\\r").replace("\n", "\\n") for line in text.splitlines(keepends=True)]


def file_diff(golden: Path, produced: Path) -> str:
    """A unified diff of a golden file against a produced one, with line terminators visible."""
    diff = difflib.unified_diff(
        visible_lines(golden.read_bytes()),
        visible_lines(produced.read_bytes()),
        fromfile=f"{golden} (golden)",
        tofile=str(produced),
        lineterm="",
    )
    return "\n".join(diff) or "(the lines are equal; the bytes differ in encoding)"


def lines_diff(expected: Sequence[str], actual: Sequence[str], expected_name: str) -> str:
    return "\n".join(difflib.unified_diff(expected, actual, fromfile=expected_name, tofile="stdout", lineterm=""))


def pinned_versions(constraints: Path) -> dict[str, str]:
    """{lowercase package name: version} from the 'name==version' lines of a pip constraints file."""
    pins: dict[str, str] = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", 1)[0].strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def reference_stack_differences(constraints: Path) -> list[str]:
    """How this interpreter and plotting stack differ from the ones that wrote the committed PNGs (empty if not)."""
    differences = []
    if sys.platform != "win32":
        differences.append(f"platform {sys.platform} (reference: win32)")
    implementation = platform.python_implementation()
    if implementation != "CPython" or sys.version_info[:2] != (3, 12):
        differences.append(f"{implementation} {platform.python_version()} (reference: CPython 3.12)")
    if not constraints.is_file():
        return [*differences, f"{constraints} not found"]
    pins = pinned_versions(constraints)
    for package in REFERENCE_PACKAGES:
        installed, pinned = installed_version(package), pins.get(package)
        if installed != pinned:
            differences.append(f"{package} {installed or 'not installed'} (reference: {pinned or 'not pinned'})")
    return differences


# ---------------------------------------------------------------- the reproduction of the released traces


def test_reproduce_prints_the_golden_console_output(reproduction: Run, golden_dir: Path) -> None:
    assert reproduction.status == 0, reproduction.stderr
    assert f"{CHECK_COUNT}/{CHECK_COUNT} numbers match the paper." in reproduction.stdout
    assert FULL_KEYWORD_AGREEMENT in reproduction.stdout
    # The golden file has CRLF line endings; splitlines() normalises them.
    golden = (golden_dir / "reproduce_stdout.txt").read_text(encoding="utf-8").splitlines()
    assert reproduction.stdout[-1] == "Wrote tables and figures to results/"
    assert reproduction.stdout == golden, lines_diff(golden, reproduction.stdout, "tests/golden/reproduce_stdout.txt")


def test_reproduce_writes_exactly_the_13_outputs(reproduction: Run, golden_hashes: dict[str, str]) -> None:
    assert sorted(golden_hashes) == sorted(CSV_FILES + PNG_FILES)
    out = reproduction.out_dir
    written = sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file())
    assert written == sorted(CSV_FILES + PNG_FILES)


@pytest.mark.parametrize("name", CSV_FILES)
def test_csv_is_byte_identical_to_the_golden(
    reproduction: Run, golden_hashes: dict[str, str], golden_csv: Callable[[str], Path], name: str
) -> None:
    produced = reproduction.out_dir / name
    # The message, and with it the committed copy to diff against, is evaluated only when the digests differ.
    assert sha256(produced) == golden_hashes[name], (
        f"{name} differs from the golden:\n{file_diff(golden_csv(name), produced)}"
    )


@pytest.mark.parametrize("name", CSV_FILES)
def test_csv_lines_end_with_crlf(reproduction: Run, name: str) -> None:
    data = (reproduction.out_dir / name).read_bytes()
    lines = data.split(b"\r\n")
    assert lines[-1] == b"", f"{name} does not end with CRLF"
    bare = [line for line in lines if b"\r" in line or b"\n" in line]
    assert bare == [], f"{name} has a line break other than CRLF"


def test_verification_has_88_matching_rows(reproduction: Run) -> None:
    with (reproduction.out_dir / "verification.csv").open(newline="", encoding="utf-8") as f:
        header, *rows = csv.reader(f)
    assert header == ["claim", "paper", "reproduced", "match"]
    assert len(rows) == CHECK_COUNT
    assert [row for row in rows if row[3] != "yes"] == []


@pytest.mark.parametrize("name", PNG_FILES)
def test_png_is_written(reproduction: Run, name: str) -> None:
    assert (reproduction.out_dir / name).read_bytes().startswith(PNG_SIGNATURE)


@pytest.mark.parametrize("name", PNG_FILES)
def test_png_is_byte_identical_to_the_golden_on_the_reference_stack(
    reproduction: Run, golden_hashes: dict[str, str], repo_root: Path, name: str
) -> None:
    if os.environ.get(STRICT_FIGURES_VARIABLE) != "1":
        differences = reference_stack_differences(repo_root / "constraints" / "reproduce.txt")
        if differences:
            pytest.skip(
                "PNG bytes are compared only on the stack that wrote results/figures "
                f"(set {STRICT_FIGURES_VARIABLE}=1 to force): {'; '.join(differences)}"
            )
    assert sha256(reproduction.out_dir / name) == golden_hashes[name], (
        f"{name} differs from the golden digest (the committed results/{name}); the PNG bytes depend on matplotlib, "
        "numpy, Pillow and its zlib-ng (constraints/reproduce.txt)"
    )


# ---------------------------------------------------------------- a trace that disagrees with the keyword rules


def change_first_tier(trace: Path) -> tuple[str, str]:
    """Rewrite a routing trace with the complexity of its first data row changed; return (old, new) tier.

    The copy is written as the released files are: gzip, CSV in the excel dialect (CRLF), UTF-8.
    """
    with gzip.open(trace, "rt", encoding="utf-8", newline="") as f:
        header, *rows = csv.reader(f)
    column = header.index("complexity")
    old = rows[0][column]
    new = {"LOW": "MEDIUM", "MEDIUM": "HIGH", "HIGH": "LOW"}[old]
    rows[0][column] = new
    with gzip.open(trace, "wt", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    return old, new


def test_one_changed_recorded_tier_fails_the_reproduction(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    traces = tmp_path / "traces"
    shutil.copytree(repo_root / "results" / "traces", traces)
    old, new = change_first_tier(traces / "routing_keyword.csv.gz")

    reports: list[Report] = []
    reproduce = paper_reproduce.reproduce

    def recording_reproduce(traces_dir: Path, prompts_path: Path, out_dir: Path) -> Report:
        reports.append(reproduce(traces_dir, prompts_path, out_dir))
        return reports[-1]

    monkeypatch.setattr(paper_reproduce, "reproduce", recording_reproduce)
    out = tmp_path / "out"
    run = run_cli(["--root", str(repo_root), "reproduce", "--traces", str(traces), "--out", str(out)], out)

    assert run.status == 1, run.stderr
    [report] = reports
    assert (report.keyword_same, report.keyword_total) == (KEYWORD_ROWS - 1, KEYWORD_ROWS)
    assert not report.ok
    assert report.matched < CHECK_COUNT
    assert f"{report.matched}/{CHECK_COUNT} numbers match the paper." in run.stdout
    # 31,018 of 31,019 still prints as 100.0%, as it did in the original script.
    assert FULL_KEYWORD_AGREEMENT in run.stdout
    assert run.stdout[-1] == f"Wrote tables and figures to {out}/"
    assert [line for line in run.stdout if line.endswith("  NO")]
    assert any(line.startswith("mmorch: error: ") for line in run.stderr.splitlines())

    with (out / "verification.csv").open(newline="", encoding="utf-8") as f:
        _, *rows = csv.reader(f)
    mismatched = {claim for claim, _, _, match in rows if match == "NO"}
    assert len(mismatched) == CHECK_COUNT - report.matched
    # The Fig. 4 counts of both tiers move by one, and the paper gives them exactly.
    assert {f"Fig. 4 Keyword {old}: count", f"Fig. 4 Keyword {new}: count"} <= mismatched
