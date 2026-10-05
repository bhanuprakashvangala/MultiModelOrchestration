"""Five-strategy completion runs (Table 1), behind `mmorch baseline`.

    mmorch baseline HumanEval

This runner produced results/traces/baseline_strategies.csv.gz; credentials come from the environment. For every
question id of a benchmark it sends one request per strategy (balanced, quality, speed, cost, baseline):

- the prompt is a fixed short text per benchmark (SAMPLE_QUESTIONS), e.g. "Math problem" for GSM8K;
- the model is MODELS[hash(strategy) % 3]; Python randomizes string hashes per process, so the strategy-to-model
  assignment changes from run to run unless PYTHONHASHSEED is set;
- success means HTTP 200 within 180 s, with max_tokens=150.

Tasks run question-major, then in strategy order, with one requests.post per task (no session, so no connection
reuse) on 20 workers. Rows come back in completion order and are written to <out>/baseline/<benchmark>_baseline.csv
with the columns benchmark, qid, strategy, model, latency (ms) and success. requests is imported lazily, once, in
run_baseline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from mmorch.data import write_dicts
from mmorch.errors import ConfigError, MissingDependencyError
from mmorch.settings import EndpointSettings

log = logging.getLogger(__name__)

MODELS: Final = ("gemma3", "llama3-sdsc", "llama3")  # served model names at the time
STRATEGIES: Final = ("balanced", "quality", "speed", "cost", "baseline")


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """The fixed prompt text of one benchmark and its question ids, f'{prefix}_{i}' for i in range(count)."""

    prefix: str
    text: str
    count: int


# Legacy key order; GPQA has no routing trace but stays selectable.
SAMPLE_QUESTIONS: Final[Mapping[str, QuestionSet]] = MappingProxyType(
    {
        "HumanEval": QuestionSet("HE", "Write a Python function", 164),
        "MBPP": QuestionSet("MB", "Python problem", 500),
        "TruthfulQA": QuestionSet("TQ", "True or false question", 790),
        "ARC": QuestionSet("ARC", "Science question", 1172),
        "GSM8K": QuestionSet("GS", "Math problem", 1319),
        "GPQA": QuestionSet("GP", "Graduate physics question", 1725),
        "MATH": QuestionSet("MATH", "Advanced math problem", 5000),
        "HellaSwag": QuestionSet("HS", "Complete the sentence", 10042),
        "MMLU-Pro": QuestionSet("MMLU", "Multiple choice question", 12032),
    }
)

DEFAULT_BENCHMARK: Final = "HumanEval"
MAX_WORKERS: Final = 20
MAX_TOKENS: Final = 150
TIMEOUT_S: Final = 180

FIELDS: Final = ("benchmark", "qid", "strategy", "model", "latency", "success")


@dataclass(frozen=True, slots=True)
class Task:
    """One request: the served model, the prompt text, the strategy and the question id."""

    model: str
    query: str
    strategy: str
    qid: str


def assign_model(strategy: str, models: Sequence[str] = MODELS, hash_fn: Callable[[str], int] = hash) -> str:
    """Return models[hash_fn(strategy) % len(models)]; the builtin hash is randomized per process."""
    return models[hash_fn(strategy) % len(models)]


def build_tasks(benchmark: str, *, hash_fn: Callable[[str], int] = hash) -> list[Task]:
    """Return one task per question and strategy, question-major and then in STRATEGIES order.

    An unknown benchmark raises KeyError.
    """
    questions = SAMPLE_QUESTIONS[benchmark]
    return [
        Task(assign_model(strategy, hash_fn=hash_fn), questions.text, strategy, f"{questions.prefix}_{i}")
        for i in range(questions.count)
        for strategy in STRATEGIES
    ]


def resolve_endpoint(endpoint: EndpointSettings) -> tuple[str, str]:
    """Return (base URL without trailing '/', key or ''); ConfigError when the base is unset or empty."""
    base_url = (endpoint.api_base or "").rstrip("/")
    if not base_url:
        raise ConfigError("Set LLM_API_BASE and LLM_API_KEY; see .env.example")
    return base_url, endpoint.api_key or ""


def call_api(
    task: Task,
    *,
    benchmark: str,
    base_url: str,
    api_key: str,
    post: Callable[..., Any],
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Send one chat completion request and return its result row; any failure gives latency 0 and success False.

    The latency is the unrounded wall time in ms, kept only for HTTP 200. Another status is a failure without a
    log record; an exception logs a warning naming the question, strategy and model.
    """
    start = clock()
    try:
        response = post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": task.model, "messages": [{"role": "user", "content": task.query}], "max_tokens": MAX_TOKENS},
            timeout=TIMEOUT_S,
        )
        latency = (clock() - start) * 1000
        if response.status_code == 200:
            return _row(task, benchmark, latency, True)
    except Exception as exc:
        log.warning("Error %s/%s/%s: %s", task.qid, task.strategy, task.model, exc)
    return _row(task, benchmark, 0, False)


def _row(task: Task, benchmark: str, latency: float, success: bool) -> dict[str, object]:
    return {
        "benchmark": benchmark,
        "qid": task.qid,
        "strategy": task.strategy,
        "model": task.model,
        "latency": latency,
        "success": success,
    }


def run_baseline(
    benchmark: str,
    *,
    base_url: str,
    api_key: str,
    workers: int = MAX_WORKERS,
    hash_fn: Callable[[str], int] = hash,
    post: Callable[..., Any] | None = None,
) -> list[dict[str, object]]:
    """Run every task of a benchmark on a thread pool and return the rows in completion order, unsorted.

    post defaults to requests.post, imported here; without the 'live' extra this raises MissingDependencyError.
    """
    if post is None:
        try:
            import requests
        except ImportError as exc:
            raise MissingDependencyError("requests", "live") from exc
        post = requests.post

    tasks = build_tasks(benchmark, hash_fn=hash_fn)
    log.info("%s: sending %d requests with %d workers", benchmark, len(tasks), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(call_api, task, benchmark=benchmark, base_url=base_url, api_key=api_key, post=post)
            for task in tasks
        ]
        # As before, as_completed yields the futures that already finished while tasks were being submitted first,
        # in no particular order, and then the others as they finish; the rows are never sorted.
        return [future.result() for future in as_completed(futures)]


def output_path(out_dir: Path, benchmark: str) -> Path:
    """Return out_dir/baseline/<benchmark>_baseline.csv."""
    return out_dir / "baseline" / f"{benchmark}_baseline.csv"


def write_results(path: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    """Write the result rows as the 6-column CSV and return path."""
    return write_dicts(path, FIELDS, rows)
