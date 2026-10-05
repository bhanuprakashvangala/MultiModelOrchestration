"""Tests for mmorch.cli: the argparse tree, the logging setup, path resolution, the five handlers and exit codes.

The handlers' collaborators (the reproduction, the routing and baseline runners, the scorer's client and the API
server) are replaced with recording fakes, so nothing here opens a socket or reads repository data. Every test
runs in an empty working directory, so the default paths (<root>/data, <root>/results) point there, and starts
from a pristine 'mmorch' logger, which conftest's mmorch_logger restores afterwards. The live runs against a stub
endpoint and the full reproduction are in tests/integration.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
import types
from collections.abc import Callable, Sequence
from importlib.metadata import entry_points
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest

from mmorch import baseline, cli
from mmorch.baseline import SAMPLE_QUESTIONS
from mmorch.data import BENCHMARKS, Prompt
from mmorch.errors import ConfigError, DataNotFoundError, MissingDependencyError, MmorchError
from mmorch.paper import reproduce as paper_reproduce
from mmorch.paper.analysis import TRACE_FILES, Check
from mmorch.paper.reproduce import Report
from mmorch.routing import runner
from mmorch.routing.runner import FIELDS, RouteRecord, TierModels
from mmorch.scoring import router as scoring_router
from mmorch.settings import DEFAULT_CHART_DIR, EndpointSettings, MatrixSettings

COMMANDS = ("reproduce", "route", "baseline", "score", "serve")
MISSING_HINT = "(run from the repository root, pass --root DIR, or give the path explicitly)"
TIMESTAMP = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}"


class Result(NamedTuple):
    """What a command line run gave: the exit status, stdout and stderr."""

    status: object
    out: str
    err: str


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> Result:
    """Call main(argv), which must return, and return its status with what it printed."""
    status = cli.main(list(argv))
    out, err = capsys.readouterr()
    return Result(status, out, err)


def run_to_exit(capsys: pytest.CaptureFixture[str], *argv: str) -> Result:
    """Call main(argv), which must end in argparse's SystemExit, and return the exit code with what it printed."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(list(argv))
    out, err = capsys.readouterr()
    return Result(excinfo.value.code, out, err)


def parse(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


def raising(error: BaseException) -> Callable[[argparse.Namespace], int]:
    """A command handler that raises error."""

    def handler(args: argparse.Namespace) -> int:
        raise error

    return handler


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def copy_to(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def make_inputs(root: Path) -> None:
    """Empty stand-ins for the released traces and prompts under root (the reproduction itself is faked)."""
    for filename in TRACE_FILES.values():
        touch(root / "results" / "traces" / filename)
    touch(root / "data" / "prompts.jsonl.gz")


# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty working directory: the default --root, so default paths never reach the repository."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeReproduce:
    """Stands in for mmorch.paper.reproduce.reproduce: records its arguments and returns a canned Report.

    It can also log one record per level on the reproduction's logger, as the real one logs its progress.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, Path]] = []
        self.checks = (Check("Table 1 HumanEval: success (%)", "80.0", "80.0"),)
        self.keyword_same = 3
        self.keyword_total = 3
        self.log = False

    def __call__(self, traces_dir: Path, prompts_path: Path, out_dir: Path) -> Report:
        self.calls.append((traces_dir, prompts_path, out_dir))
        if self.log:
            logger = logging.getLogger("mmorch.paper.reproduce")
            logger.debug("debug detail")
            logger.info("Reading the traces in %s", traces_dir)
            logger.warning("a warning")
        return self.report(out_dir)

    def report(self, out_dir: Path) -> Report:
        return Report(self.checks, self.keyword_same, self.keyword_total, out_dir, ())


@pytest.fixture
def fake_reproduce(monkeypatch: pytest.MonkeyPatch) -> FakeReproduce:
    fake = FakeReproduce()
    monkeypatch.setattr(paper_reproduce, "reproduce", fake)
    return fake


def printed(lines: Sequence[str]) -> str:
    """What printing each line gives on stdout."""
    return "".join(f"{line}\n" for line in lines)


class FakeRouting:
    """Stands in for make_client and run_routing of mmorch.routing.runner and records how they are called.

    Every other prompt fails, so a run of n prompts reports ceil(n / 2) successes.
    """

    def __init__(self) -> None:
        self.client = object()
        self.endpoints: list[EndpointSettings] = []
        self.calls: list[SimpleNamespace] = []

    def make_client(self, endpoint: EndpointSettings) -> object:
        self.endpoints.append(endpoint)
        return self.client

    def run_routing(
        self, client: object, prompts: Sequence[Prompt], method: str, models: TierModels, *, workers: int, timeout: int
    ) -> list[RouteRecord]:
        self.calls.append(
            SimpleNamespace(
                client=client, prompts=list(prompts), method=method, models=models, workers=workers, timeout=timeout
            )
        )
        return [self.record(prompt, method, succeeded=i % 2 == 0) for i, prompt in enumerate(prompts)]

    @staticmethod
    def record(prompt: Prompt, method: str, *, succeeded: bool) -> RouteRecord:
        question = prompt.question[:200]
        head = (prompt.qid, question, "MEDIUM", method, "qwen3")
        if succeeded:
            return RouteRecord(*head, 1.5, 0.5, 1.0, 2.0, 3, 4, 7, "hi", True, None)
        return RouteRecord(*head, 0, 0, 0, 0, 0, 0, 0, "", False, "boom")


@pytest.fixture
def fake_routing(monkeypatch: pytest.MonkeyPatch) -> FakeRouting:
    fake = FakeRouting()
    monkeypatch.setattr(runner, "make_client", fake.make_client)
    monkeypatch.setattr(runner, "run_routing", fake.run_routing)
    return fake


class FakeBaseline:
    """Stands in for mmorch.baseline.run_baseline: records its arguments and returns one success and one failure."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, benchmark: str, *, base_url: str, api_key: str) -> list[dict[str, object]]:
        self.calls.append((benchmark, base_url, api_key))
        qid = f"{SAMPLE_QUESTIONS[benchmark].prefix}_0"
        row = {"benchmark": benchmark, "qid": qid}
        return [
            {**row, "strategy": "quality", "model": "llama3", "latency": 12.5, "success": True},
            {**row, "strategy": "balanced", "model": "gemma3", "latency": 0, "success": False},
        ]


@pytest.fixture
def fake_baseline(monkeypatch: pytest.MonkeyPatch) -> FakeBaseline:
    fake = FakeBaseline()
    monkeypatch.setattr(baseline, "run_baseline", fake)
    return fake


class FakeChat:
    """An OpenAI client stand-in for the scorer: every completion answers reply with 6 tokens used."""

    def __init__(self, reply: str = "Paris.") -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=6))


@pytest.fixture
def fake_scorer(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replaces mmorch.scoring.router.make_client; .client is the fake client, .calls the make_client arguments."""
    pytest.importorskip("yaml")
    fake = SimpleNamespace(client=FakeChat(), calls=[])

    def make_client(endpoint: EndpointSettings, default_base_url: str) -> FakeChat:
        fake.calls.append((endpoint, default_base_url))
        return fake.client

    monkeypatch.setattr(scoring_router, "make_client", make_client)
    return fake


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[MatrixSettings]:
    """The settings passed to serve() of a stand-in mmorch.matrix.api (the real module needs the 'matrix' extra)."""
    calls: list[MatrixSettings] = []
    api = types.ModuleType("mmorch.matrix.api")
    api.serve = calls.append
    monkeypatch.setitem(sys.modules, "mmorch.matrix.api", api)
    return calls


# ---------------------------------------------------------------- the parser and its legacy defaults


def test_global_options_default_to_the_current_directory_and_info() -> None:
    args = parse("reproduce")
    assert args.root == Path(".")
    assert args.verbosity == 0
    assert args.handler is cli._reproduce


@pytest.mark.parametrize(
    ("flags", "verbosity"),
    [([], 0), (["-v"], 1), (["--verbose"], 1), (["-vv"], 2), (["-q"], -1), (["--quiet"], -1)],
)
def test_verbosity_flags(flags: list[str], verbosity: int) -> None:
    assert parse(*flags, "score").verbosity == verbosity


def test_root_is_a_path() -> None:
    assert parse("--root", "checkout", "score").root == Path("checkout")


def test_route_keeps_the_legacy_defaults() -> None:
    args = parse("route", "HumanEval")
    legacy = ("HumanEval", "keyword", 20, 90, None)
    assert (args.benchmark, args.routing, args.workers, args.timeout, args.limit) == legacy
    assert type(args.workers) is int
    assert type(args.timeout) is int
    assert (args.prompts, args.out) == (None, None)
    assert args.handler is cli._route


def test_route_takes_the_legacy_flags() -> None:
    args = parse("route", "MMLU-Pro", "--routing", "llm", "--workers", "50", "--limit", "100", "--timeout", "30")
    assert (args.benchmark, args.routing, args.workers, args.limit, args.timeout) == ("MMLU-Pro", "llm", 50, 100, 30)


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_route_accepts_every_benchmark_of_the_traces(benchmark: str) -> None:
    assert parse("route", benchmark).benchmark == benchmark


def test_route_takes_path_overrides() -> None:
    args = parse("route", "ARC", "--prompts", "p.jsonl.gz", "--out", "live")
    assert (args.prompts, args.out) == (Path("p.jsonl.gz"), Path("live"))


def test_baseline_defaults_to_humaneval() -> None:
    args = parse("baseline")
    assert args.benchmark == "HumanEval"
    assert args.out is None
    assert args.handler is cli._baseline


@pytest.mark.parametrize("benchmark", list(SAMPLE_QUESTIONS))
def test_baseline_accepts_every_sample_benchmark_including_gpqa(benchmark: str) -> None:
    assert parse("baseline", benchmark).benchmark == benchmark


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("reproduce", ("traces", "prompts", "out")),
        ("score", ("config",)),
        ("serve", ("host", "port", "namespace", "chart_dir")),
    ],
)
def test_options_default_to_none(command: str, options: tuple[str, ...]) -> None:
    args = parse(command)
    assert {name: getattr(args, name) for name in options} == dict.fromkeys(options)


def test_options_take_their_types() -> None:
    reproduce = parse("reproduce", "--traces", "t", "--prompts", "p", "--out", "o")
    assert (reproduce.traces, reproduce.prompts, reproduce.out) == (Path("t"), Path("p"), Path("o"))
    assert parse("score", "--config", "c.yaml").config == Path("c.yaml")
    serve = parse("serve", "--host", "0.0.0.0", "--port", "9000", "--namespace", "ns", "--chart-dir", "./charts")
    assert (serve.host, serve.port, serve.namespace, serve.chart_dir) == ("0.0.0.0", 9000, "ns", "./charts")
    assert type(serve.chart_dir) is str  # the helm argv is built from the raw string


# ---------------------------------------------------------------- help, version and usage errors


def test_no_command_prints_the_help_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    result = run(capsys)
    assert (result.status, result.out) == (2, "")
    assert result.err.startswith("usage: mmorch ")
    assert all(command in result.err for command in COMMANDS)


def test_global_options_alone_are_no_command(capsys: pytest.CaptureFixture[str]) -> None:
    result = run(capsys, "-v", "--root", "checkout")
    assert (result.status, result.out) == (2, "")
    assert result.err.startswith("usage: mmorch ")


def test_main_reads_sys_argv_by_default(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mmorch", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "mmorch 2.0.0\n"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_to_exit(capsys, "--version") == Result(0, "mmorch 2.0.0\n", "")


def test_the_console_script_runs_main() -> None:
    [script] = entry_points(group="console_scripts", name="mmorch")
    assert script.value == "mmorch.cli:main"
    assert script.load() is cli.main


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_has_help(capsys: pytest.CaptureFixture[str], command: str) -> None:
    result = run_to_exit(capsys, command, "--help")
    assert (result.status, result.err) == (0, "")
    assert re.match(rf"usage: mmorch {command}\s", result.out)


def test_help_shows_the_global_options_and_lists_the_commands(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COLUMNS", "120")  # argparse wraps the usage line to the terminal width
    result = run_to_exit(capsys, "--help")
    assert result.status == 0
    assert result.out.startswith("usage: mmorch [-h] [--root DIR] [-v | -q] [--version] COMMAND ...\n")
    assert all(command in result.out for command in COMMANDS)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["route", "GPQA"], "invalid choice: 'GPQA'"),  # GPQA has no routing trace
        (["route", "humaneval"], "invalid choice: 'humaneval'"),
        (["route"], "the following arguments are required: BENCHMARK"),
        (["route", "HumanEval", "--routing", "random"], "invalid choice: 'random'"),
        (["route", "HumanEval", "--workers", "many"], "invalid int value: 'many'"),
        (["baseline", "Unknown"], "invalid choice: 'Unknown'"),  # was a KeyError traceback
        (["baseline", "HumanEval", "extra"], "unrecognized arguments: extra"),  # was silently ignored
        (["serve", "--port", "http"], "invalid int value: 'http'"),
        (["deploy"], "invalid choice: 'deploy'"),
        (["reproduce", "-v"], "unrecognized arguments: -v"),  # global options come before the command
        (["-v", "-q", "reproduce"], "not allowed with argument"),
    ],
)
def test_usage_errors_exit_2(capsys: pytest.CaptureFixture[str], argv: list[str], message: str) -> None:
    result = run_to_exit(capsys, *argv)
    assert (result.status, result.out) == (2, "")
    assert result.err.startswith("usage: mmorch")
    assert message in result.err


# ---------------------------------------------------------------- exit codes and error reports


@pytest.mark.parametrize(
    "error",
    [
        MmorchError("something failed"),
        ConfigError("Set LLM_API_BASE (and LLM_API_KEY); see .env.example"),
        DataNotFoundError("prompts file not found: data/prompts.jsonl.gz"),
        MissingDependencyError("openai", "live"),
    ],
    ids=["MmorchError", "ConfigError", "DataNotFoundError", "MissingDependencyError"],
)
def test_mmorch_errors_exit_1_with_one_line(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, error: MmorchError
) -> None:
    monkeypatch.setattr(cli, "_score", raising(error))
    assert run(capsys, "score") == Result(1, "", f"mmorch: error: {error}\n")


def test_verbose_errors_log_the_traceback_at_debug(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_score", raising(ConfigError("bad value")))
    result = run(capsys, "-v", "score")

    lines = result.err.splitlines()
    assert (result.status, result.out) == (1, "")
    assert re.fullmatch(rf"{TIMESTAMP} DEBUG mmorch\.cli: The command failed:", lines[0])
    assert lines[1] == "Traceback (most recent call last):"
    assert lines[-2:] == ["mmorch.errors.ConfigError: bad value", "mmorch: error: bad value"]


def test_ctrl_c_exits_130(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_route", raising(KeyboardInterrupt()))
    assert run(capsys, "route", "HumanEval") == Result(130, "", "")


def test_other_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_serve", raising(RuntimeError("a bug")))
    with pytest.raises(RuntimeError, match="a bug"):
        cli.main(["serve"])


def test_main_returns_the_status_of_the_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_serve", lambda args: 3)
    assert run(capsys, "serve").status == 3


# ---------------------------------------------------------------- logging


def format_record(handler: logging.Handler, level: int) -> str:
    record = logging.LogRecord("mmorch.paper.reproduce", level, __file__, 1, "Reading %s", ("traces",), None)
    return handler.format(record)


@pytest.mark.parametrize(
    ("verbosity", "level", "pattern"),
    [
        (-1, logging.WARNING, "Reading traces"),
        (0, logging.INFO, "Reading traces"),
        (1, logging.DEBUG, rf"{TIMESTAMP} INFO mmorch\.paper\.reproduce: Reading traces"),
        (2, logging.DEBUG, rf"{TIMESTAMP} INFO mmorch\.paper\.reproduce: Reading traces"),
    ],
)
def test_configure_logging_sends_the_mmorch_logger_to_stderr(
    mmorch_logger: logging.Logger, verbosity: int, level: int, pattern: str
) -> None:
    cli.configure_logging(verbosity)

    [handler] = mmorch_logger.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr
    assert re.fullmatch(pattern, format_record(handler, logging.INFO))
    assert mmorch_logger.level == level
    assert mmorch_logger.propagate is False


def test_configure_logging_twice_keeps_one_handler(mmorch_logger: logging.Logger) -> None:
    other = logging.NullHandler()
    mmorch_logger.addHandler(other)

    cli.configure_logging(-1)
    [first] = [h for h in mmorch_logger.handlers if h is not other]
    cli.configure_logging(1)

    [second] = [h for h in mmorch_logger.handlers if h is not other]
    assert second is not first
    assert other in mmorch_logger.handlers  # a handler it did not install stays
    assert len(mmorch_logger.handlers) == 2
    assert mmorch_logger.level == logging.DEBUG
    assert mmorch_logger.propagate is False


def test_configure_logging_leaves_the_root_logger_alone(mmorch_logger: logging.Logger) -> None:
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    cli.configure_logging(1)
    assert (list(root.handlers), root.level) == before


def test_logs_go_to_stderr_and_results_to_stdout(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    workdir: Path,
    fake_reproduce: FakeReproduce,
) -> None:
    make_inputs(workdir)
    fake_reproduce.log = True

    result = run(capsys, "reproduce")

    assert result.status == 0
    assert result.out == printed(fake_reproduce.report(Path("results")).lines())
    assert result.err == f"Reading the traces in {Path('results', 'traces')}\na warning\n"
    assert caplog.records == []  # nothing reaches the root logger


def test_quiet_logs_only_warnings(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce
) -> None:
    make_inputs(workdir)
    fake_reproduce.log = True
    result = run(capsys, "-q", "reproduce")
    assert (result.status, result.err) == (0, "a warning\n")


def test_verbose_logs_debug_details_with_timestamps(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce
) -> None:
    make_inputs(workdir)
    fake_reproduce.log = True
    result = run(capsys, "-v", "reproduce")

    lines = result.err.splitlines()
    assert result.status == 0
    assert len(lines) == 3
    assert re.fullmatch(rf"{TIMESTAMP} DEBUG mmorch\.paper\.reproduce: debug detail", lines[0])
    assert re.fullmatch(rf"{TIMESTAMP} INFO mmorch\.paper\.reproduce: Reading the traces in .+", lines[1])
    assert re.fullmatch(rf"{TIMESTAMP} WARNING mmorch\.paper\.reproduce: a warning", lines[2])


# ---------------------------------------------------------------- reproduce


def test_reproduce_resolves_the_defaults_under_the_working_directory(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce
) -> None:
    make_inputs(workdir)
    result = run(capsys, "reproduce")

    assert fake_reproduce.calls == [(Path("results", "traces"), Path("data", "prompts.jsonl.gz"), Path("results"))]
    assert result.status == 0
    # As in tests/golden/reproduce_stdout.txt.
    assert result.out.splitlines()[-1] == "Wrote tables and figures to results/"


def test_reproduce_resolves_the_defaults_under_the_root(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, fake_reproduce: FakeReproduce
) -> None:
    root = tmp_path / "checkout"
    make_inputs(root)
    result = run(capsys, "--root", str(root), "reproduce")

    assert fake_reproduce.calls == [
        (root / "results" / "traces", root / "data" / "prompts.jsonl.gz", root / "results"),
    ]
    assert result.status == 0
    assert result.out.splitlines()[-1] == f"Wrote tables and figures to {root / 'results'}/"


def test_reproduce_uses_explicit_paths_as_typed(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce
) -> None:
    for filename in TRACE_FILES.values():
        touch(workdir / "t" / filename)
    touch(workdir / "p.jsonl.gz")

    # The root holds nothing: explicit paths are relative to the working directory, not to --root.
    result = run(capsys, "--root", "elsewhere", "reproduce", "--traces", "t", "--prompts", "p.jsonl.gz", "--out", "o")

    assert fake_reproduce.calls == [(Path("t"), Path("p.jsonl.gz"), Path("o"))]
    assert result.status == 0
    assert result.out.splitlines()[-1] == f"Wrote tables and figures to {Path('o')}/"


def test_reproduce_prints_the_report_and_exits_0(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce
) -> None:
    make_inputs(workdir)
    fake_reproduce.checks = (
        Check("Table 1 HumanEval: runs / success / failures", "820 / 656 / 164", "820 / 656 / 164"),
        Check("Prompts across 8 benchmarks", "31,019", "31,019"),
    )
    result = run(capsys, "reproduce")

    assert result == Result(0, printed(fake_reproduce.report(Path("results")).lines()), "")
    assert "2/2 numbers match the paper." in result.out.splitlines()


@pytest.mark.parametrize(
    ("checks", "keyword_same", "keyword_total", "summary"),
    [
        (
            (Check("Claim A", "80.0", "80.0"), Check("Claim B", "89.8", "89.7")),
            3,
            3,
            "1/2 numbers match the paper and the keyword classifier agrees with 3/3 recorded tiers",
        ),
        (
            # One disagreeing tier still prints 100.0%, but the reproduction is not exact.
            (Check("Claim A", "80.0", "80.0"),),
            31018,
            31019,
            "1/1 numbers match the paper and the keyword classifier agrees with 31018/31019 recorded tiers",
        ),
    ],
    ids=["number", "tier"],
)
def test_reproduce_mismatch_exits_1(
    capsys: pytest.CaptureFixture[str],
    workdir: Path,
    fake_reproduce: FakeReproduce,
    checks: tuple[Check, ...],
    keyword_same: int,
    keyword_total: int,
    summary: str,
) -> None:
    make_inputs(workdir)
    fake_reproduce.checks = checks
    fake_reproduce.keyword_same, fake_reproduce.keyword_total = keyword_same, keyword_total

    result = run(capsys, "reproduce")

    assert result.status == 1
    assert result.out == printed(fake_reproduce.report(Path("results")).lines())
    assert result.err == f"mmorch: error: reproduction mismatch: {summary}\n"


def test_reproduce_in_an_empty_root_reports_the_missing_input(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, fake_reproduce: FakeReproduce
) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    result = run(capsys, "--root", str(root), "reproduce")

    missing = root / "results" / "traces" / "baseline_strategies.csv.gz"
    assert result == Result(1, "", f"mmorch: error: baseline trace not found: {missing} {MISSING_HINT}\n")
    assert "not found" in result.err
    assert fake_reproduce.calls == []
    assert list(root.iterdir()) == []  # nothing written


@pytest.mark.parametrize(
    ("missing", "what"),
    [
        (Path("results", "traces", "baseline_strategies.csv.gz"), "baseline trace"),
        (Path("results", "traces", "routing_keyword.csv.gz"), "keyword trace"),
        (Path("results", "traces", "routing_llm.csv.gz"), "llm trace"),
        (Path("data", "prompts.jsonl.gz"), "prompts file"),
    ],
)
def test_reproduce_checks_every_input_before_any_work(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_reproduce: FakeReproduce, missing: Path, what: str
) -> None:
    make_inputs(workdir)
    (workdir / missing).unlink()

    result = run(capsys, "reproduce")

    assert result == Result(1, "", f"mmorch: error: {what} not found: {missing} {MISSING_HINT}\n")
    assert fake_reproduce.calls == []


# ---------------------------------------------------------------- route


@pytest.mark.parametrize("base", [None, ""], ids=["unset", "empty"])
def test_route_without_an_endpoint_exits_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, base: str | None
) -> None:
    if base is not None:
        monkeypatch.setenv("LLM_API_BASE", base)
    # As before, the endpoint is checked first, even without a prompts file.
    assert run(capsys, "route", "HumanEval") == Result(
        1, "", "mmorch: error: Set LLM_API_BASE (and LLM_API_KEY); see .env.example\n"
    )


def test_route_runs_the_benchmark_with_the_environment(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    tiny_prompts: Path,
    fake_routing: FakeRouting,
) -> None:
    copy_to(tiny_prompts, workdir / "data" / "prompts.jsonl.gz")
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL_LOW", "small-x")
    monkeypatch.setenv("MODEL_HIGH", "big-y")

    result = run(capsys, "route", "HumanEval", "--routing", "llm", "--workers", "3", "--timeout", "7")

    out = Path("results", "live", "llm", "HumanEval_llm.csv")
    assert result == Result(0, f"HumanEval [llm]: 2/3 succeeded -> {out}\n", "")
    assert fake_routing.endpoints == [EndpointSettings("http://llm.test/v1", "sk-test")]
    [call] = fake_routing.calls
    assert call.client is fake_routing.client
    assert [p.qid for p in call.prompts] == ["HumanEval_1", "HumanEval_2", "HumanEval_3"]
    assert (call.method, call.workers, call.timeout) == ("llm", 3, 7)
    assert call.models == TierModels(low="small-x", medium="qwen3", high="big-y", classifier="small-x")
    with (workdir / out).open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(FIELDS)
    assert [(row[0], row[13]) for row in rows[1:]] == [
        ("HumanEval_1", "True"),
        ("HumanEval_2", "False"),
        ("HumanEval_3", "True"),
    ]


def test_route_uses_the_legacy_defaults(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    tiny_prompts: Path,
    fake_routing: FakeRouting,
) -> None:
    copy_to(tiny_prompts, workdir / "data" / "prompts.jsonl.gz")
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")

    assert run(capsys, "route", "HumanEval").status == 0

    assert fake_routing.endpoints == [EndpointSettings("http://llm.test/v1", None)]
    [call] = fake_routing.calls
    assert (call.method, call.workers, call.timeout, call.models) == ("keyword", 20, 90, TierModels())
    assert (workdir / "results" / "live" / "keyword" / "HumanEval_keyword.csv").is_file()


@pytest.mark.parametrize(
    ("limit", "qids"),
    [
        (None, ["HumanEval_1", "HumanEval_2", "HumanEval_3"]),
        ("2", ["HumanEval_1", "HumanEval_2"]),
        ("0", []),
        ("-1", ["HumanEval_1", "HumanEval_2"]),  # a negative limit drops from the end, as before
    ],
)
def test_route_limit_slices_after_the_benchmark_filter(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tiny_prompts: Path,
    fake_routing: FakeRouting,
    limit: str | None,
    qids: list[str],
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")
    argv = ["route", "HumanEval", "--prompts", str(tiny_prompts), "--out", "live"]
    if limit is not None:
        argv += ["--limit", limit]

    result = run(capsys, *argv)

    assert [p.qid for p in fake_routing.calls[0].prompts] == qids
    out = Path("live", "keyword", "HumanEval_keyword.csv")
    assert result.out == f"HumanEval [keyword]: {(len(qids) + 1) // 2}/{len(qids)} succeeded -> {out}\n"


def test_route_resolves_the_defaults_under_the_root(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_prompts: Path,
    fake_routing: FakeRouting,
) -> None:
    root = tmp_path / "checkout"
    copy_to(tiny_prompts, root / "data" / "prompts.jsonl.gz")
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")

    result = run(capsys, "--root", str(root), "route", "MBPP")

    out = root / "results" / "live" / "keyword" / "MBPP_keyword.csv"
    assert result == Result(0, f"MBPP [keyword]: 1/1 succeeded -> {out}\n", "")
    assert [p.qid for p in fake_routing.calls[0].prompts] == ["MBPP_1"]
    assert out.is_file()


def test_route_uses_explicit_paths_as_typed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    tiny_prompts: Path,
    fake_routing: FakeRouting,
) -> None:
    copy_to(tiny_prompts, workdir / "my-prompts.jsonl.gz")
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")

    result = run(capsys, "--root", "elsewhere", "route", "MBPP", "--prompts", "my-prompts.jsonl.gz", "--out", "o")

    out = Path("o", "keyword", "MBPP_keyword.csv")
    assert result == Result(0, f"MBPP [keyword]: 1/1 succeeded -> {out}\n", "")
    assert (workdir / out).is_file()
    assert not (workdir / "elsewhere").exists()


def test_route_without_the_prompts_file_exits_1_before_any_request(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, workdir: Path, fake_routing: FakeRouting
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")

    result = run(capsys, "route", "HumanEval")

    missing = Path("data", "prompts.jsonl.gz")
    assert result == Result(1, "", f"mmorch: error: prompts file not found: {missing} {MISSING_HINT}\n")
    assert fake_routing.calls == []
    assert list(workdir.iterdir()) == []


# ---------------------------------------------------------------- baseline


@pytest.mark.parametrize("base", [None, "", "/"], ids=["unset", "empty", "slash"])
def test_baseline_without_an_endpoint_exits_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, base: str | None
) -> None:
    if base is not None:
        monkeypatch.setenv("LLM_API_BASE", base)
    assert run(capsys, "baseline") == Result(
        1, "", "mmorch: error: Set LLM_API_BASE and LLM_API_KEY; see .env.example\n"
    )


def test_baseline_runs_the_default_benchmark(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, workdir: Path, fake_baseline: FakeBaseline
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1//")

    result = run(capsys, "baseline")

    out = Path("results", "live", "baseline", "HumanEval_baseline.csv")
    assert result == Result(0, f"HumanEval: 1/2 succeeded -> {out}\n", "")
    # The base loses its trailing slashes and an unset key is sent as ''.
    assert fake_baseline.calls == [("HumanEval", "http://llm.test/v1", "")]
    assert (workdir / out).read_bytes() == (
        b"benchmark,qid,strategy,model,latency,success\r\n"
        b"HumanEval,HE_0,quality,llama3,12.5,True\r\n"
        b"HumanEval,HE_0,balanced,gemma3,0,False\r\n"
    )


def test_baseline_resolves_the_defaults_under_the_root(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_baseline: FakeBaseline
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    root = tmp_path / "checkout"

    result = run(capsys, "--root", str(root), "baseline", "GPQA")

    out = root / "results" / "live" / "baseline" / "GPQA_baseline.csv"
    assert result == Result(0, f"GPQA: 1/2 succeeded -> {out}\n", "")
    assert fake_baseline.calls == [("GPQA", "http://llm.test/v1", "k")]
    assert out.is_file()


def test_baseline_uses_an_explicit_out_as_typed(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, workdir: Path, fake_baseline: FakeBaseline
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")

    result = run(capsys, "--root", "elsewhere", "baseline", "MBPP", "--out", "o")

    out = Path("o", "baseline", "MBPP_baseline.csv")
    assert result == Result(0, f"MBPP: 1/2 succeeded -> {out}\n", "")
    assert (workdir / out).is_file()
    assert not (workdir / "elsewhere").exists()


# ---------------------------------------------------------------- score


def test_score_runs_the_demo(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, fake_scorer: SimpleNamespace
) -> None:
    monkeypatch.setenv("LLM_API_BASE", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "")

    result = run(capsys, "score")

    assert result.status == 0
    # The client gets the raw environment values and the YAML's base URL as the fallback.
    assert fake_scorer.calls == [
        (EndpointSettings("http://llm.test/v1", ""), "https://your-llm-endpoint.example.org/v1")
    ]
    assert [(r["model"], r["max_tokens"]) for r in fake_scorer.client.requests] == [
        ("gemma3", 100),
        ("qwen3", 300),
        ("gemma3", 100),
    ]
    # The demo's report goes to stdout, as before, without the router's three initialisation lines.
    assert result.out.startswith("\n" + "=" * 80 + "\nPICK-AND-SPIN ROUTING SYSTEM TEST\n" + "=" * 80 + "\n\n")
    lines = result.out.splitlines()
    assert "[Test 1] Strategy: speed" in lines
    assert "> Model: gemma3" in lines
    assert "> Tokens: 6" in lines
    assert lines[-2:] == ["Total queries: 3", "Model usage: {'qwen3': 1, 'gemma3': 2, 'llama3': 0}"]
    assert result.err.splitlines() == [
        "> Pick-and-Spin Router initialized",
        "> Available models: ['qwen3', 'gemma3', 'llama3']",
        "> Available strategies: ['quality', 'cost', 'speed', 'balanced', 'baseline']",
    ]


def test_quiet_score_hides_the_initialisation_lines(
    capsys: pytest.CaptureFixture[str], fake_scorer: SimpleNamespace
) -> None:
    result = run(capsys, "-q", "score")
    assert (result.status, result.err) == (0, "")
    assert "PICK-AND-SPIN ROUTING SYSTEM TEST" in result.out


def test_score_reads_an_explicit_config_as_typed(
    capsys: pytest.CaptureFixture[str], workdir: Path, fake_scorer: SimpleNamespace
) -> None:
    packaged = files("mmorch.scoring").joinpath("config.yaml").read_text(encoding="utf-8")
    custom = packaged.replace("https://your-llm-endpoint.example.org/v1", "http://from-yaml.test/v1")
    assert custom != packaged
    (workdir / "scorer.yaml").write_text(custom, encoding="utf-8")

    result = run(capsys, "--root", "elsewhere", "score", "--config", "scorer.yaml")

    assert result.status == 0
    assert fake_scorer.calls == [(EndpointSettings(None, None), "http://from-yaml.test/v1")]


def test_score_with_a_missing_config_exits_1(capsys: pytest.CaptureFixture[str], fake_scorer: SimpleNamespace) -> None:
    result = run(capsys, "score", "--config", "missing.yaml")
    assert (result.status, result.out) == (1, "")
    assert result.err.startswith("mmorch: error: scorer config not found: missing.yaml ")
    assert fake_scorer.calls == []


def test_score_with_a_malformed_config_exits_1(
    capsys: pytest.CaptureFixture[str], fake_scorer: SimpleNamespace, tmp_path: Path
) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("api: {}\n", encoding="utf-8")
    result = run(capsys, "score", "--config", str(path))
    assert (result.status, result.out) == (1, "")
    assert result.err == f"mmorch: error: {path}: missing required key 'base_url'\n"
    assert fake_scorer.calls == []


def test_score_without_yaml_names_the_extra(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    assert run(capsys, "score") == Result(
        1, "", 'mmorch: error: pyyaml is required for this command: pip install -e ".[live]"\n'
    )


def test_score_without_openai_names_the_extra(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("yaml")
    monkeypatch.setitem(sys.modules, "openai", None)
    assert run(capsys, "score") == Result(
        1, "", 'mmorch: error: openai is required for this command: pip install -e ".[live]"\n'
    )


# ---------------------------------------------------------------- serve


def test_serve_defaults(capsys: pytest.CaptureFixture[str], served: list[MatrixSettings]) -> None:
    assert run(capsys, "serve") == Result(0, "", "")
    assert served == [MatrixSettings("default", "./deploy/helm/pick-and-spin-umbrella", "localhost", 8080)]


ENVIRONMENT = {
    "KUBERNETES_NAMESPACE": "env-ns",
    "MATRIX_CHART_DIR": "./env-charts",
    "API_HOST": "0.0.0.0",
    "API_PORT": "9000",
}


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], MatrixSettings("env-ns", "./env-charts", "0.0.0.0", 9000)),
        (["--port", "9100"], MatrixSettings("env-ns", "./env-charts", "0.0.0.0", 9100)),
        (
            ["--host", "127.0.0.1", "--port", "9100", "--namespace", "ns", "--chart-dir", "./charts"],
            MatrixSettings("ns", "./charts", "127.0.0.1", 9100),
        ),
    ],
    ids=["environment", "one-flag", "all-flags"],
)
def test_serve_flags_win_over_the_environment(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    served: list[MatrixSettings],
    flags: list[str],
    expected: MatrixSettings,
) -> None:
    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert run(capsys, "serve", *flags).status == 0
    assert served == [expected]


def test_serve_with_an_invalid_port_variable_exits_1(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, served: list[MatrixSettings]
) -> None:
    monkeypatch.setenv("API_PORT", "abc")
    assert run(capsys, "serve") == Result(1, "", "mmorch: error: API_PORT must be an integer, got 'abc'\n")
    assert served == []


def test_serve_port_flag_skips_the_port_variable(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, served: list[MatrixSettings]
) -> None:
    monkeypatch.setenv("API_PORT", "abc")
    assert run(capsys, "serve", "--port", "9100").status == 0
    assert served[0].port == 9100


def test_serve_ignores_the_root(capsys: pytest.CaptureFixture[str], served: list[MatrixSettings]) -> None:
    assert run(capsys, "--root", "checkout", "serve").status == 0
    assert served[0].chart_dir == DEFAULT_CHART_DIR


def test_serve_without_fastapi_names_the_extra(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pydantic")
    uvicorn = pytest.importorskip("uvicorn")

    def run_server(*args: object, **kwargs: object) -> None:
        pytest.fail("uvicorn.run must not be reached without fastapi")

    monkeypatch.setattr(uvicorn, "run", run_server)
    monkeypatch.setitem(sys.modules, "fastapi", None)

    assert run(capsys, "serve") == Result(
        1, "", 'mmorch: error: fastapi is required for this command: pip install -e ".[matrix]"\n'
    )
