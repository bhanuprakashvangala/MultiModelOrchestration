"""Import hygiene: what the command line and the library modules load when imported, and three source conventions.

The import checks run in fresh interpreters (sys.executable), all started at once, because this test process has
imported everything long ago. They also run in the base-install CI job, where no extra is installed; there, a
module-level import of an optional package fails its check with an ImportError instead of showing up in
sys.modules.

- Building the parser, `mmorch --help` and every `mmorch <command> --help` load neither numpy nor matplotlib nor
  any optional dependency.
- `python -m mmorch` is the command line: --version prints the version, and the status main() returns becomes
  the exit status.
- The library modules load no optional dependency and no matplotlib when imported; only mmorch.paper may load
  numpy. Optional packages are imported inside the functions that need them.
- A scan of the package source pins three conventions: only mmorch.cli reads the environment, no module calls
  logging.basicConfig, and nothing imports matplotlib.pyplot.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import mmorch

# The plotting stack of `mmorch reproduce`, then the packages of the extras live, matrix and matrix-ml.
UNWANTED = (
    "numpy",
    "matplotlib",
    "openai",
    "requests",
    "yaml",
    "fastapi",
    "uvicorn",
    "pydantic",
    "aiohttp",
    "torch",
    "transformers",
)
COMMANDS = ("reproduce", "route", "baseline", "score", "serve")
# What the command line runs before any command does work: each must leave every UNWANTED package unloaded.
CLI_PROBES = {
    "build_parser()": "import mmorch, mmorch.cli; mmorch.cli.build_parser()",
    "mmorch --help": "from mmorch.cli import main; main(['--help'])",
    **{
        f"mmorch {command} --help": f"from mmorch.cli import main; main([{command!r}, '--help'])"
        for command in COMMANDS
    },
}
LIBRARY_MODULES = (
    "mmorch.paper.reproduce",
    "mmorch.routing",
    "mmorch.routing.runner",
    "mmorch.baseline",
    "mmorch.scoring",
    "mmorch.scoring.router",
    "mmorch.matrix",
    "mmorch.matrix.backend_manager",
    "mmorch.matrix.orchestrator",
    "mmorch.matrix.domain",
)
# The reproduction is numpy code; everything else is plain Python until a command needs an extra.
ALLOWED = {"mmorch.paper.reproduce": ("numpy",)}

# Runs one statement in a fresh interpreter and prints a JSON report on one line: the SystemExit code the
# statement ended with (None if it returned), what it printed, and which of the watched top-level packages are
# loaded afterwards.
PROBE = textwrap.dedent(
    """
    import contextlib, io, json, sys

    statement, watched = sys.argv[1], json.loads(sys.argv[2])
    status = None
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        try:
            exec(statement)
        except SystemExit as exc:
            status = exc.code
    loaded = sorted(name for name in watched if name in sys.modules)
    print(json.dumps({"status": status, "stdout": printed.getvalue(), "loaded": loaded}))
    """
)

PACKAGE_DIR = Path(mmorch.__file__).resolve().parent


# ---------------------------------------------------------------- fresh interpreters


def run_probe(statement: str, watched: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run statement in a fresh interpreter, in an empty working directory so that nothing there shadows a module."""
    return subprocess.run(
        [sys.executable, "-c", PROBE, statement, json.dumps(list(watched))],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def report(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """The probe's JSON report; fails the test with the interpreter's stderr if the statement raised."""
    assert result.returncode == 0, f"the probe failed:\n{result.stderr}"
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


@pytest.fixture(scope="module")
def probes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, subprocess.CompletedProcess[str]]:
    """Every probe's result by name, the CLI probes and one per library module, run in parallel."""
    cwd = tmp_path_factory.mktemp("probes")
    statements = {**CLI_PROBES, **{module: f"import {module}" for module in LIBRARY_MODULES}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {name: pool.submit(run_probe, statement, UNWANTED, cwd) for name, statement in statements.items()}
        return {name: future.result() for name, future in futures.items()}


@pytest.mark.parametrize("name", CLI_PROBES)
def test_the_command_line_loads_no_heavy_or_optional_package(
    probes: dict[str, subprocess.CompletedProcess[str]], name: str
) -> None:
    outcome = report(probes[name])

    assert outcome["loaded"] == [], f"{name} loaded {', '.join(outcome['loaded'])}"
    if name == "build_parser()":
        assert (outcome["status"], outcome["stdout"]) == (None, "")
    else:
        # The help went to stdout and argparse exited 0: 'usage: mmorch ...' or 'usage: mmorch route ...'.
        assert outcome["status"] == 0
        assert outcome["stdout"].startswith(f"usage: {name.removesuffix(' --help')} ")


@pytest.mark.parametrize("module", LIBRARY_MODULES)
def test_importing_a_library_module_loads_no_optional_package(
    probes: dict[str, subprocess.CompletedProcess[str]], module: str
) -> None:
    outcome = report(probes[module])

    unwanted = [name for name in outcome["loaded"] if name not in ALLOWED.get(module, ())]
    assert unwanted == [], f"importing {module} loaded {', '.join(unwanted)}"


def python_m_mmorch(*argv: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mmorch", *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_python_m_mmorch_prints_the_version(tmp_path: Path) -> None:
    result = python_m_mmorch("--version", cwd=tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (0, f"mmorch {mmorch.__version__}\n", "")


def test_python_m_mmorch_exits_with_the_status_main_returns(tmp_path: Path) -> None:
    # Without a command main() prints the help to stderr and returns 2, which must become the exit status.
    result = python_m_mmorch(cwd=tmp_path)
    assert (result.returncode, result.stdout) == (2, "")
    assert result.stderr.startswith("usage: mmorch ")


# ---------------------------------------------------------------- source conventions

ENVIRONMENT_NAMES = frozenset({"environ", "environb", "getenv", "getenvb"})
Scan = Callable[[ast.Module], list[int]]


def environment_reads(tree: ast.Module) -> list[int]:
    """Lines that read the environment through the os module: os.environ, os.getenv, or a from-import of them."""
    os_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_names.add(alias.asname or "os")
                elif alias.name.startswith("os.") and alias.asname is None:
                    os_names.add("os")  # `import os.path` binds os as well
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in ENVIRONMENT_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id in os_names
        ) or (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name in ENVIRONMENT_NAMES for alias in node.names)
        ):
            lines.append(node.lineno)
    return sorted(lines)


def basic_config_calls(tree: ast.Module) -> list[int]:
    """Lines that use logging.basicConfig, as an attribute or through a from-import."""
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr == "basicConfig")
        or (
            isinstance(node, ast.ImportFrom)
            and node.module == "logging"
            and any(alias.name == "basicConfig" for alias in node.names)
        )
    )


def _is_pyplot(name: str) -> bool:
    return name == "matplotlib.pyplot" or name.startswith("matplotlib.pyplot.")


def _imports_pyplot(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_pyplot(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
        return _is_pyplot(node.module) or (
            node.module == "matplotlib" and any(alias.name == "pyplot" for alias in node.names)
        )
    if isinstance(node, ast.Call) and node.args:
        func, first = node.func, node.args[0]
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return (
            called in ("import_module", "__import__")
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and _is_pyplot(first.value)
        )
    return False


def pyplot_imports(tree: ast.Module) -> list[int]:
    """Lines that import matplotlib.pyplot: import statements, or import_module / __import__ with a literal name."""
    return sorted(node.lineno for node in ast.walk(tree) if _imports_pyplot(node))


@pytest.fixture(scope="module")
def sources() -> dict[str, ast.Module]:
    """Every module of the installed package, parsed, by path relative to the package directory."""
    return {
        path.relative_to(PACKAGE_DIR).as_posix(): ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
    }


def findings(sources: dict[str, ast.Module], scan: Scan) -> dict[str, list[int]]:
    """{module path: lines} for every module where scan finds something."""
    return {name: lines for name, tree in sources.items() if (lines := scan(tree))}


def test_the_scan_covers_the_whole_package(sources: dict[str, ast.Module]) -> None:
    assert {"__init__.py", "cli.py", "paper/figures.py", "matrix/api.py", "scoring/router.py"} <= set(sources)


def test_only_the_command_line_reads_the_environment(sources: dict[str, ast.Module]) -> None:
    assert set(findings(sources, environment_reads)) == {"cli.py"}, findings(sources, environment_reads)


def test_no_module_calls_logging_basic_config(sources: dict[str, ast.Module]) -> None:
    assert findings(sources, basic_config_calls) == {}


def test_nothing_imports_pyplot(sources: dict[str, ast.Module]) -> None:
    assert findings(sources, pyplot_imports) == {}


@pytest.mark.parametrize(
    ("source", "scan"),
    [
        ("import os\nos.environ['LLM_API_BASE']", environment_reads),
        ("import os\nos.getenv('LLM_API_BASE')", environment_reads),
        ("import os as system\nsystem.environ.get('LLM_API_BASE')", environment_reads),
        ("import os.path\nos.environ", environment_reads),
        ("from os import environ", environment_reads),
        ("from os import getenv as read", environment_reads),
        ("import logging\nlogging.basicConfig(level=logging.INFO)", basic_config_calls),
        ("from logging import basicConfig", basic_config_calls),
        ("import matplotlib.pyplot as plt", pyplot_imports),
        ("from matplotlib import pyplot", pyplot_imports),
        ("from matplotlib.pyplot import subplots", pyplot_imports),
        ("def f():\n    import matplotlib.pyplot", pyplot_imports),
        ("import importlib\nimportlib.import_module('matplotlib.pyplot')", pyplot_imports),
    ],
)
def test_the_scans_find_each_form(source: str, scan: Scan) -> None:
    assert scan(ast.parse(source)) != []


@pytest.mark.parametrize(
    ("source", "scan"),
    [
        ('"""Only mmorch.cli reads os.environ."""\nimport os\nos.path.join("a", "b")', environment_reads),
        ("environ = {}\nenviron.get('LLM_API_BASE')", environment_reads),
        ("import logging\nlog = logging.getLogger(__name__)", basic_config_calls),
        ('"""No matplotlib.pyplot here."""\nfrom matplotlib.figure import Figure', pyplot_imports),
        ("import matplotlib\nmatplotlib.style.context('default')", pyplot_imports),
    ],
)
def test_the_scans_ignore_lookalikes(source: str, scan: Scan) -> None:
    assert scan(ast.parse(source)) == []
