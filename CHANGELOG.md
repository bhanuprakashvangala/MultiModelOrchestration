# Changelog

## 2.0.0

The scripts of the first release are now the installable Python package `mmorch`, with one `mmorch` command. No
numeric change (enforced by `tests/golden` and CI): `mmorch reproduce` prints the same report as
`python scripts/reproduce.py` and writes the same tables, `verification.csv` and figures, byte for byte, and the
live runners send the same requests and write the same CSV columns.

### Commands

| 1.0.0 | 2.0.0 |
|---|---|
| `pip install -r requirements.txt` | `pip install -e .` (reproduce) or `pip install -e ".[live]"` (route, baseline, score) |
| `pip install -r requirements-matrix.txt` | `pip install -e ".[matrix,matrix-ml]"` (serve) |
| | `pip install -e ".[dev]"` (development) |
| `python scripts/reproduce.py` | `mmorch reproduce` |
| `python src/routing/smart_routing.py BENCHMARK --routing R --workers N` | `mmorch route BENCHMARK --routing R --workers N` |
| `python src/baseline/strategy_baseline.py BENCHMARK` | `mmorch baseline BENCHMARK` |
| `python src/routing/multi_objective.py` | `mmorch score` |
| `python src/matrix/api_server.py` | `mmorch serve` |

`mmorch route` keeps the arguments and defaults of `smart_routing.py`, including `--timeout` and `--limit`. Apart
from the global `-v`, `-q` and `--version`, the new options are paths (`--root` for every command, `--traces`,
`--prompts`, `--out` and `--config`) and the `mmorch serve` options `--host`, `--port`, `--namespace` and
`--chart-dir`, which mirror existing environment variables. The original scripts are kept at tag `v1.0.0`; there are
no compatibility wrappers in `scripts/` or `src/`.

### Changed

- Python 3.11 or newer is required.
- Logging. Progress and diagnostics go to stderr with their old text: the `n/total` progress of `mmorch route`, the
  `Error <qid>/<strategy>/<model>: ...` lines of `mmorch baseline`, the scorer's `> ...` start-up lines and its
  `[Warning]` lines, and `matplotlib not installed; skipping figures`. Two commands also log new progress lines
  there: `mmorch reproduce` writes three (the traces and prompts it reads, and where it writes), and
  `mmorch baseline` writes one (`<benchmark>: sending N requests with 20 workers`). stdout carries only results:
  the reproduction report, which is byte-identical to the old output when run from the repository root with the
  default paths, the route and baseline summaries and the scorer demo. `-v` adds timestamped debug output and `-q`
  keeps only warnings and errors, which also hides the progress lines.
- Errors and exit codes. A missing input file (the prompts, a trace or a `mmorch score --config` file), a missing
  extra and a missing or invalid environment variable print one line, `mmorch: error: <message>`, and exit 1; the
  two `Set LLM_API_BASE ...` messages keep their text, and a missing extra names the `pip install` command for it.
  `mmorch reproduce` still exits 1 when a number or a tier differs, and now says so on stderr. Usage errors exit 2,
  and Ctrl-C exits 130.
- `mmorch baseline` checks its argument: an unknown benchmark or an extra argument is a usage error. Before, the first
  stopped with a `KeyError` traceback and the second was ignored. GPQA can still be selected, and HumanEval is still
  the default.
- Paths. Default inputs and outputs resolve against `--root` (default: the current directory) instead of the
  script's location, so run the commands from the repository root or pass `--root DIR`. Summary lines print the
  output path as built from `--root` or `--out` (a relative path by default) instead of an absolute one.
- `mmorch serve` starts. `python src/matrix/api_server.py` stopped with `RuntimeError: no running event loop`; the
  orchestrator and its idle-shutdown task are now created inside the running server. `mmorch serve` reads
  `KUBERNETES_NAMESPACE`, `MATRIX_CHART_DIR`, `API_HOST` and `API_PORT`.
  `uvicorn --factory mmorch.matrix.api:create_app` also serves the API but, unlike `uvicorn api_server:app` before,
  does not read `KUBERNETES_NAMESPACE` or `MATRIX_CHART_DIR`: it uses the default namespace and chart directory.
- Figures are drawn with matplotlib's object-oriented API, without pyplot and without switching the global backend.
  On the plotting stack pinned in `constraints/reproduce.txt` the PNGs are byte-identical to the committed ones.
- Dependencies. `requirements.txt` and `requirements-matrix.txt` are replaced by `pyproject.toml`: the base install
  (numpy, matplotlib) and the extras `live`, `matrix`, `matrix-ml` and `dev`. The matrix extras drop the unused
  `requests` and `numpy` entries of `requirements-matrix.txt`.
- The scorer configuration moved from `src/routing/config.yaml` to `src/mmorch/scoring/config.yaml` and ships with the
  package; `mmorch score --config FILE` reads another file.
- `deploy/Dockerfile` builds the image that `deploy/jobs/smart-routing-job.yaml` runs.
- The README no longer says that the live runner writes the format of the traces: the runner writes 15 columns, and
  the released traces are a reduced 13-column export that records a failed request's error as `timeout` instead of
  the exception message.

The runners, the scorer and the model x backend prototype are otherwise ported as they were, including known quirks
of the prototype; those will be fixed in separate releases.

## 1.0.0

The code, experiment traces and reproduction script for the DAI Workshop at AAAI 2026 paper, as flat scripts
(tag `v1.0.0`).
