"""mmorch: multi-model orchestration for self-hosted LLMs (DAI Workshop at AAAI 2026).

The package follows the paper:

- mmorch.routing: Pick's LOW / MEDIUM / HIGH tier routing, the runner behind the routing traces (Figs. 4-11)
- mmorch.scoring: the Eq. 2 multi-objective scorer with its operator profiles
- mmorch.baseline: the five-strategy runs behind Table 1
- mmorch.paper: the reproduction of Table 1 and Figs. 4-11 from the released traces
- mmorch.matrix: the earlier model x backend prototype (extras 'matrix' and 'matrix-ml')
- mmorch.cli: the `mmorch` command

Importing mmorch imports nothing else, so it never loads numpy, matplotlib or an optional dependency.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__: str = version("mmorch")
except PackageNotFoundError:
    __version__ = "0+unknown"
