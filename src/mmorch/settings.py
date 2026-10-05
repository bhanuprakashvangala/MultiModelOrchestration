"""Repository layout and the environment-driven settings that more than one command uses.

Nothing here reads os.environ. mmorch.cli passes the environment to the from_env() constructors when a command
starts, and tests build the dataclasses directly. Default paths are relative to a root directory: the --root
option, by default the current directory, which is the repository root in normal use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from mmorch.errors import ConfigError, DataNotFoundError

# A str on purpose: the helm argv is built from it with os.path.join and must stay byte-identical, and pathlib
# would drop the leading './'.
DEFAULT_CHART_DIR: Final[str] = "./deploy/helm/pick-and-spin-umbrella"


@dataclass(frozen=True, slots=True)
class Paths:
    """The default input and output locations under a root directory."""

    root: Path = Path(".")

    @property
    def prompts(self) -> Path:
        """The released prompts: <root>/data/prompts.jsonl.gz."""
        return self.root / "data" / "prompts.jsonl.gz"

    @property
    def traces(self) -> Path:
        """The released traces: <root>/results/traces."""
        return self.root / "results" / "traces"

    @property
    def results(self) -> Path:
        """Where `mmorch reproduce` writes its tables and figures: <root>/results."""
        return self.root / "results"

    @property
    def live(self) -> Path:
        """Where the live runners write their CSVs: <root>/results/live (git-ignored)."""
        return self.root / "results" / "live"


def require_file(path: Path, what: str) -> Path:
    """Return path if it is an existing file, else raise DataNotFoundError naming what is missing and where."""
    if path.is_file():
        return path
    raise DataNotFoundError(
        f"{what} not found: {path} (run from the repository root, pass --root DIR, or give the path explicitly)"
    )


@dataclass(frozen=True, slots=True)
class EndpointSettings:
    """The OpenAI-compatible endpoint given by LLM_API_BASE and LLM_API_KEY, as raw values.

    A value is None when its variable is unset; an empty string is kept and nothing is stripped. Each consumer
    applies its own legacy rule: mmorch.routing.runner.make_client, mmorch.baseline.resolve_endpoint and
    mmorch.scoring.router.make_client. The key never appears in repr().
    """

    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> EndpointSettings:
        """Read LLM_API_BASE and LLM_API_KEY from environ, the mapping the command line passes in."""
        return cls(environ.get("LLM_API_BASE"), environ.get("LLM_API_KEY"))


@dataclass(frozen=True, slots=True)
class MatrixSettings:
    """Settings of the model x backend prototype (`mmorch serve`): namespace, Helm chart directory and address."""

    namespace: str = "default"
    chart_dir: str = DEFAULT_CHART_DIR
    host: str = "localhost"
    port: int = 8080

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> MatrixSettings:
        """Read KUBERNETES_NAMESPACE, MATRIX_CHART_DIR, API_HOST and API_PORT from environ.

        Unset variables take the defaults and empty strings are kept. A non-integer API_PORT raises ConfigError.
        """
        value = environ.get("API_PORT", "8080")
        try:
            port = int(value)
        except ValueError as exc:
            raise ConfigError(f"API_PORT must be an integer, got {value!r}") from exc
        return cls(
            namespace=environ.get("KUBERNETES_NAMESPACE", "default"),
            chart_dir=environ.get("MATRIX_CHART_DIR", DEFAULT_CHART_DIR),
            host=environ.get("API_HOST", "localhost"),
            port=port,
        )
