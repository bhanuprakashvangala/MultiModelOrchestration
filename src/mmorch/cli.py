"""The `mmorch` command line, also reachable as `python -m mmorch`.

    mmorch [--root DIR] [-v | -q] [--version] COMMAND ...

- reproduce: Table 1 and Figs. 4-11 from the released traces (base install)
- route: Pick tier routing of one benchmark's prompts on a live endpoint (extra 'live')
- baseline: the five-strategy Table 1 requests for one benchmark (extra 'live')
- score: the multi-objective scorer demo (extra 'live')
- serve: the model x backend prototype API (extra 'matrix'; 'matrix-ml' adds the DistilBERT domain classifier)

This is the only module that reads os.environ, which it does when a command starts, and the only one that prints.
It builds the argparse tree, sends the 'mmorch' logger (the parent of every module's logger) to stderr, and
imports each command's implementation inside its handler, so building the parser or printing help loads no
optional dependency and not even numpy.
stdout carries only command results; progress and diagnostics are logged to stderr.

Default inputs and outputs resolve under --root (default: the current directory); a path given to a flag is used
as typed. Exit status: 0 on success; 1 for a MmorchError (missing input, missing extra, missing or invalid
environment value) or a reproduction that differs from the paper, reported as 'mmorch: error: <message>' on
stderr; 2 for a usage error or a missing command; 130 on Ctrl-C.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import ChainMap
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, TypeAlias

from mmorch import __version__
from mmorch.baseline import DEFAULT_BENCHMARK, SAMPLE_QUESTIONS
from mmorch.data import BENCHMARKS
from mmorch.errors import MmorchError
from mmorch.routing.runner import DEFAULT_TIMEOUT_S, DEFAULT_WORKERS, ROUTING_METHODS
from mmorch.settings import DEFAULT_CHART_DIR, EndpointSettings, MatrixSettings, Paths, require_file

log = logging.getLogger(__name__)

Command: TypeAlias = Callable[[argparse.Namespace], int]

# configure_logging names its handler, so that a second call replaces it and leaves any other handler alone.
_HANDLER_NAME: Final = "mmorch.cli"
_MESSAGE_FORMAT: Final = "%(message)s"
_DEBUG_FORMAT: Final = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree: the global options, then one subcommand per command with its legacy defaults."""
    parser = argparse.ArgumentParser(
        prog="mmorch",
        description="Multi-model orchestration for self-hosted LLMs (DAI Workshop at AAAI 2026).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="directory that holds data/ and results/; default paths resolve under it (default: the current directory)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="count", dest="verbosity", help="log debug details with timestamps"
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_const", const=-1, dest="verbosity", help="log warnings and errors only"
    )
    parser.add_argument("--version", action="version", version=f"mmorch {__version__}")
    # No command leaves handler None, and main() prints the help.
    parser.set_defaults(handler=None, verbosity=0)
    commands = parser.add_subparsers(title="commands", metavar="COMMAND")

    reproduce = commands.add_parser(
        "reproduce",
        help="regenerate Table 1 and Figs. 4-11 from the released traces",
        description="Regenerate Table 1 and Figs. 4-11 from the released traces, compare every reproduced number "
        "with the paper and write the tables, the figures and verification.csv. Exits 1 unless every number "
        "matches and the keyword rules reproduce every recorded tier. Needs only the base install.",
    )
    reproduce.add_argument(
        "--traces", type=Path, metavar="DIR", help="trace directory (default: <root>/results/traces)"
    )
    reproduce.add_argument(
        "--prompts", type=Path, metavar="FILE", help="prompts file (default: <root>/data/prompts.jsonl.gz)"
    )
    reproduce.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results)")
    reproduce.set_defaults(handler=_reproduce)

    route = commands.add_parser(
        "route",
        help="route one benchmark's prompts to three model tiers on a live endpoint",
        description="Classify each prompt of a benchmark as LOW, MEDIUM or HIGH and send it to that tier's model "
        "over an OpenAI-compatible streaming API (LLM_API_BASE, LLM_API_KEY; served model names from MODEL_LOW, "
        "MODEL_MEDIUM, MODEL_HIGH and MODEL_CLASSIFIER). Writes OUT/<routing>/<BENCHMARK>_<routing>.csv. "
        "Needs the 'live' extra.",
    )
    route.add_argument("benchmark", choices=BENCHMARKS, metavar="BENCHMARK", help=f"one of {', '.join(BENCHMARKS)}")
    route.add_argument(
        "--routing", choices=ROUTING_METHODS, default="keyword", help="tier classifier (default: %(default)s)"
    )
    route.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N", help="concurrent requests (default: %(default)s)"
    )
    route.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_S,
        metavar="S",
        help="per-request timeout in seconds (default: %(default)s)",
    )
    route.add_argument(
        "--limit", type=int, metavar="N", help="run only the first N prompts of the benchmark (default: all)"
    )
    route.add_argument(
        "--prompts", type=Path, metavar="FILE", help="prompts file (default: <root>/data/prompts.jsonl.gz)"
    )
    route.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/live)")
    route.set_defaults(handler=_route)

    baseline = commands.add_parser(
        "baseline",
        help="run the five-strategy Table 1 requests for one benchmark",
        description="Send one request per question and strategy of a benchmark to an OpenAI-compatible endpoint "
        "(LLM_API_BASE, LLM_API_KEY). Each strategy's model is chosen by Python's string hash, so it changes from "
        "run to run unless PYTHONHASHSEED is set. Writes OUT/baseline/<BENCHMARK>_baseline.csv. Needs the 'live' "
        "extra.",
    )
    baseline.add_argument(
        "benchmark",
        nargs="?",
        default=DEFAULT_BENCHMARK,
        choices=tuple(SAMPLE_QUESTIONS),
        metavar="BENCHMARK",
        help=f"one of {', '.join(SAMPLE_QUESTIONS)} (default: %(default)s)",
    )
    baseline.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/live)")
    baseline.set_defaults(handler=_baseline)

    score = commands.add_parser(
        "score",
        help="run the multi-objective scorer demo on a live endpoint",
        description="Route three demo queries with the multi-objective scorer and run them on an OpenAI-compatible "
        "endpoint (LLM_API_BASE, else the YAML api.base_url; LLM_API_KEY). Needs the 'live' extra.",
    )
    score.add_argument(
        "--config", type=Path, metavar="FILE", help="scorer YAML (default: the packaged mmorch/scoring/config.yaml)"
    )
    score.set_defaults(handler=_score)

    serve = commands.add_parser(
        "serve",
        help="serve the model x backend prototype API",
        description="Serve the 3x3 domain model x backend prototype API with uvicorn. Each option falls back to "
        "its environment variable, then to its default. Needs the 'matrix' extra; 'matrix-ml' adds the DistilBERT "
        "domain classifier, without which domains are classified by keywords.",
    )
    serve.add_argument("--host", metavar="H", help="bind address (default: $API_HOST, else localhost)")
    serve.add_argument("--port", type=int, metavar="P", help="port (default: $API_PORT, else 8080)")
    serve.add_argument(
        "--namespace", metavar="NS", help="Kubernetes namespace (default: $KUBERNETES_NAMESPACE, else default)"
    )
    serve.add_argument(
        "--chart-dir",
        metavar="DIR",
        help=f"umbrella Helm chart directory (default: $MATRIX_CHART_DIR, else {DEFAULT_CHART_DIR})",
    )
    serve.set_defaults(handler=_serve)
    return parser


def configure_logging(verbosity: int) -> None:
    """Send the 'mmorch' logger to stderr: WARNING for -1, INFO for 0 and DEBUG with timestamps for 1 or more.

    The one handler goes on the 'mmorch' logger, which stops propagating, so the root logger and the loggers of
    other libraries (httpx, openai) are left alone. A second call replaces the handler of the first.
    """
    logger = logging.getLogger("mmorch")
    for installed in [h for h in logger.handlers if h.get_name() == _HANDLER_NAME]:
        logger.removeHandler(installed)
        installed.close()
    if verbosity < 0:
        level, fmt = logging.WARNING, _MESSAGE_FORMAT
    elif verbosity == 0:
        level, fmt = logging.INFO, _MESSAGE_FORMAT
    else:
        level, fmt = logging.DEBUG, _DEBUG_FORMAT
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line on argv (default: sys.argv[1:]) and return the exit status.

    Usage errors, --help and --version end in argparse's own SystemExit, with status 2 or 0.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    command: Command | None = args.handler
    if command is None:
        parser.print_help(sys.stderr)
        return 2
    configure_logging(args.verbosity)
    try:
        return command(args)
    except MmorchError as exc:
        log.debug("The command failed:", exc_info=True)
        return _error(exc)
    except KeyboardInterrupt:
        return 130


def _error(message: object) -> int:
    """Report a failed command on stderr as 'mmorch: error: <message>' and return exit status 1."""
    print(f"mmorch: error: {message}", file=sys.stderr)
    return 1


def _reproduce(args: argparse.Namespace) -> int:
    """`mmorch reproduce`: check the inputs exist, reproduce the paper's numbers and print the report.

    Exits 0 only if every number matches the paper and the keyword rules reproduce every recorded tier.
    """
    from mmorch.paper.analysis import TRACE_FILES
    from mmorch.paper.reproduce import reproduce

    paths = Paths(args.root)
    traces = args.traces if args.traces is not None else paths.traces
    for name, filename in TRACE_FILES.items():
        require_file(traces / filename, f"{name} trace")
    prompts = require_file(args.prompts if args.prompts is not None else paths.prompts, "prompts file")
    out = args.out if args.out is not None else paths.results

    report = reproduce(traces, prompts, out)
    for line in report.lines():
        print(line)
    if report.ok:
        return 0
    # One disagreeing tier still prints 100.0% above, so say what failed.
    return _error(
        f"reproduction mismatch: {report.matched}/{len(report.checks)} numbers match the paper and the keyword "
        f"classifier agrees with {report.keyword_same}/{report.keyword_total} recorded tiers"
    )


def _route(args: argparse.Namespace) -> int:
    """`mmorch route`: route one benchmark's prompts on the live endpoint and write the 15-column CSV.

    Failed requests are rows with success False, so they do not change the exit status.
    """
    from mmorch.data import load_prompts
    from mmorch.routing.runner import TierModels, make_client, output_path, run_routing, write_rows

    paths = Paths(args.root)
    client = make_client(EndpointSettings.from_env(os.environ))
    models = TierModels.from_env(os.environ)
    prompts_path = require_file(args.prompts if args.prompts is not None else paths.prompts, "prompts file")
    prompts = load_prompts(prompts_path, args.benchmark)[: args.limit]
    out = output_path(args.out if args.out is not None else paths.live, args.benchmark, args.routing)
    # As before, the output directory exists before the first request, so a bad --out fails before a long run.
    out.parent.mkdir(parents=True, exist_ok=True)

    records = run_routing(client, prompts, args.routing, models, workers=args.workers, timeout=args.timeout)
    write_rows(out, records)
    ok = sum(record.success for record in records)
    print(f"{args.benchmark} [{args.routing}]: {ok}/{len(records)} succeeded -> {out}")
    return 0


def _baseline(args: argparse.Namespace) -> int:
    """`mmorch baseline`: run the five-strategy requests for one benchmark and write the 6-column CSV.

    Failed requests are rows with success False, so they do not change the exit status.
    """
    from mmorch.baseline import output_path, resolve_endpoint, run_baseline, write_results

    base_url, api_key = resolve_endpoint(EndpointSettings.from_env(os.environ))
    rows = run_baseline(args.benchmark, base_url=base_url, api_key=api_key)
    out = write_results(output_path(args.out if args.out is not None else Paths(args.root).live, args.benchmark), rows)
    ok = sum(1 for row in rows if row["success"])
    print(f"{args.benchmark}: {ok}/{len(rows)} succeeded -> {out}")
    return 0


def _score(args: argparse.Namespace) -> int:
    """`mmorch score`: run the scorer demo and print its report.

    The router's initialisation lines and warnings are logged to stderr; failed queries are part of the report.
    """
    from mmorch.scoring.config import load_config
    from mmorch.scoring.router import MultiObjectiveRouter, make_client, run_demo

    config = load_config(args.config)
    client = make_client(EndpointSettings.from_env(os.environ), config.api_base_url)
    run_demo(MultiObjectiveRouter(config, client), print)
    return 0


def _serve(args: argparse.Namespace) -> int:
    """`mmorch serve`: serve the prototype API until it is stopped.

    Each flag wins over its environment variable, which wins over the default; an environment variable is not
    read at all when its flag is given.
    """
    flags = {
        "API_HOST": args.host,
        "API_PORT": args.port,
        "KUBERNETES_NAMESPACE": args.namespace,
        "MATRIX_CHART_DIR": args.chart_dir,
    }
    given = {name: str(value) for name, value in flags.items() if value is not None}
    settings = MatrixSettings.from_env(ChainMap(given, os.environ))

    from mmorch.matrix.api import serve

    serve(settings)
    return 0
