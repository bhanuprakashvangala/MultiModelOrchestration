"""The errors that the command line reports as one line with exit status 1.

Library code never calls sys.exit, print or SystemExit. It raises one of these errors instead, and mmorch.cli
prints 'mmorch: error: <message>' on stderr, logs the traceback at DEBUG and returns 1. Each specific error also
derives from the matching built-in exception, so callers can still catch ValueError, FileNotFoundError or
ImportError.

Optional dependencies are imported inside the function that needs them, never at module level. There is no
import helper, so the imported names keep their static types:

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingDependencyError("openai", "live") from exc

The extra that provides each package:

- live: openai, requests and yaml (reported as 'pyyaml')
- matrix: fastapi, uvicorn, aiohttp and pydantic
- matrix-ml: torch and transformers (mmorch.matrix.domain falls back to keywords instead of raising)
"""

from __future__ import annotations


class MmorchError(Exception):
    """Base class of the errors that the command line turns into exit status 1."""


class ConfigError(MmorchError, ValueError):
    """A required environment variable or option value is missing or invalid."""


class DataNotFoundError(MmorchError, FileNotFoundError):
    """A required input file does not exist."""


class MissingDependencyError(MmorchError, ImportError):
    """An optional dependency is not installed; the message names the extra that provides it.

    package is the name to install (for example 'pyyaml') and extra the mmorch extra that brings it in.
    """

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(f'{package} is required for this command: pip install -e ".[{extra}]"')
        self.package = package
        self.extra = extra

    def __reduce__(self) -> tuple[type[MissingDependencyError], tuple[str, str]]:
        # The constructor takes (package, extra) rather than the message, so pickling passes those back.
        return type(self), (self.package, self.extra)
