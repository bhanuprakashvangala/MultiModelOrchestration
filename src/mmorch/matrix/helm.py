"""The exact helm argv lists that both managers use, and the single synchronous subprocess seam.

BackendManager installs and uninstalls one release per endpoint from the subchart under <chart_dir>/charts/.
Orchestrator switches a release on or off in the umbrella chart, installed as the release 'multi-llm', through a
'<model>_<backend>.enabled' value.

Chart paths are built with os.path.join on the raw chart directory string, never with pathlib, so the argv stays
byte-identical (pathlib would drop the leading './'). helm runs synchronously, through subprocess.run, exactly as
before; the managers call it through an injectable runner so that tests never execute helm.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from typing import Final, TypeAlias

UMBRELLA_RELEASE: Final = "multi-llm"

HelmRunner: TypeAlias = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a helm command synchronously, capturing its output as text."""
    return subprocess.run(list(cmd), capture_output=True, text=True)


def subchart_path(chart_dir: str, release: str) -> str:
    """Return the path of a release's subchart: chart_dir/charts/release."""
    return os.path.join(chart_dir, "charts", release)


def install_cmd(release: str, chart_path: str, namespace: str) -> list[str]:
    """Return the argv that installs or upgrades one release from its own chart."""
    return [
        "helm",
        "upgrade",
        "--install",
        release,
        chart_path,
        "--namespace",
        namespace,
        "--create-namespace",
        "--wait",
        "--timeout",
        "5m",
    ]


def uninstall_cmd(release: str, namespace: str) -> list[str]:
    """Return the argv that uninstalls one release."""
    return ["helm", "uninstall", release, "--namespace", namespace]


def values_key(release: str) -> str:
    """Return the umbrella chart's values key for a release ('-' becomes '_')."""
    return release.replace("-", "_")


def umbrella_enable_cmd(chart_dir: str, namespace: str, release: str) -> list[str]:
    """Return the argv that enables one release in the umbrella chart."""
    return [
        "helm",
        "upgrade",
        "--install",
        UMBRELLA_RELEASE,
        chart_dir,
        "--namespace",
        namespace,
        "--create-namespace",
        "--set",
        f"{values_key(release)}.enabled=true",
        "--wait",
        "--timeout",
        "5m",
    ]


def umbrella_disable_cmd(chart_dir: str, namespace: str, release: str) -> list[str]:
    """Return the argv that disables one release in the umbrella chart."""
    return [
        "helm",
        "upgrade",
        UMBRELLA_RELEASE,
        chart_dir,
        "--namespace",
        namespace,
        "--set",
        f"{values_key(release)}.enabled=false",
        "--wait",
        "--timeout",
        "2m",
    ]
