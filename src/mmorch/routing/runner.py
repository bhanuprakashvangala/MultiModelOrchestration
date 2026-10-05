"""The live routing runner behind `mmorch route`.

- Maps a tier to a model key and a served model name.
- Makes the streaming call with the legacy time.time() bookkeeping.
- Fans out on a ThreadPoolExecutor that shares one client.
- Logs 'n/total' every 200 completions.
- Returns rows in prompt order and writes the 15-column CSV.

The released traces in results/traces are a reduced 13-column export of such rows, not this CSV: they add
benchmark, drop question, routing_method and response, write success as 1/0 and the timings as floats, and
record a failed request's error as 'timeout' where this runner writes str(exc).

openai is imported only inside make_client; the annotations import it under TYPE_CHECKING.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from mmorch.data import Prompt, write_dicts
from mmorch.errors import ConfigError, MissingDependencyError
from mmorch.routing.classifier import classify_keyword, classify_llm
from mmorch.settings import EndpointSettings

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

ROUTING_METHODS: Final = ("keyword", "llm")
RoutingMethod: TypeAlias = Literal["keyword", "llm"]

# Tier -> model key, the value of the CSV's model column.
MODEL_KEYS: Final[Mapping[str, str]] = MappingProxyType({"LOW": "llama3-small", "MEDIUM": "qwen3"})
FALLBACK_MODEL_KEY: Final = "deepseek"

GEN_MAX_TOKENS: Final = 512
GEN_TEMPERATURE: Final = 0.7
QUESTION_CHARS: Final = 200
RESPONSE_CHARS: Final = 500
PROGRESS_EVERY: Final = 200
DEFAULT_WORKERS: Final = 20
DEFAULT_TIMEOUT_S: Final = 90


def route_to_model(tier: str) -> str:
    """Return the model key for a tier: LOW -> llama3-small, MEDIUM -> qwen3, anything else -> deepseek."""
    return MODEL_KEYS.get(tier, FALLBACK_MODEL_KEY)


@dataclass(frozen=True, slots=True)
class TierModels:
    """The served model name behind each model key, and the model that classifies for --routing llm."""

    low: str = "llama3"
    medium: str = "qwen3"
    high: str = "deepseek-r1"
    classifier: str = "llama3"

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> TierModels:
        """Read MODEL_LOW, MODEL_MEDIUM, MODEL_HIGH and MODEL_CLASSIFIER (default: the resolved low model).

        Unset variables take the names used in the recorded runs; empty strings are kept.
        """
        low = environ.get("MODEL_LOW", "llama3")
        return cls(
            low=low,
            medium=environ.get("MODEL_MEDIUM", "qwen3"),
            high=environ.get("MODEL_HIGH", "deepseek-r1"),
            classifier=environ.get("MODEL_CLASSIFIER", low),
        )

    def served(self, model_key: str) -> str:
        """Return the served model name for a model key: llama3-small, qwen3 or deepseek."""
        return {"llama3-small": self.low, "qwen3": self.medium, "deepseek": self.high}[model_key]


@dataclass(frozen=True, slots=True)
class CallResult:
    """Timing, token counts and response of one streaming call, stored exactly as computed and never coerced.

    A timing is an int 0 rather than a float where the legacy arithmetic gave one, so the CSV writes '0' there.
    """

    latency_ms: float
    ttft_ms: float
    generation_time_ms: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: str
    success: bool
    error: str | None

    @classmethod
    def failure(cls, exc: BaseException) -> CallResult:
        """The result of a failed call: int zeros, an empty response, success False and the error text."""
        return cls(0, 0, 0, 0, 0, 0, 0, "", False, str(exc))


@dataclass(frozen=True, slots=True)
class RouteRecord:
    """One row of the runner's CSV; the field order is the column order."""

    qid: str
    question: str
    complexity: str
    routing_method: str
    model: str
    latency_ms: float
    ttft_ms: float
    generation_time_ms: float
    tokens_per_second: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response: str
    success: bool
    error: str | None

    def as_row(self) -> dict[str, object]:
        """Return the record as a CSV row keyed by field name."""
        return dataclasses.asdict(self)


FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in dataclasses.fields(RouteRecord))


def make_client(endpoint: EndpointSettings) -> OpenAI:
    """Create the OpenAI client that all workers share; ConfigError when LLM_API_BASE is unset or empty.

    An unset or empty LLM_API_KEY is sent as 'none'. The SDK defaults stay untouched (two retries, its own HTTP
    client), because the retry count decides how many requests reach the endpoint on 5xx errors and timeouts.
    """
    if not endpoint.api_base:
        raise ConfigError("Set LLM_API_BASE (and LLM_API_KEY); see .env.example")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingDependencyError("openai", "live") from exc
    return OpenAI(base_url=endpoint.api_base, api_key=endpoint.api_key or "none")


def call_model(
    client: OpenAI,
    model: str,
    query: str,
    timeout: int,
    *,
    clock: Callable[[], float] = time.time,
) -> CallResult:
    """Stream one completion of query from the served model and time it, statement for statement as before.

    The clock is read at the start, at the first content (time to first token, then the generation start), at
    the end of the stream and, if a token arrived, once more for the generation time. Token counts come from a
    usage chunk when the server sends one, otherwise from whitespace splits of the query and the response. Any
    exception gives CallResult.failure.
    """
    start = clock()
    ttft: float | None = None
    gen_start: float | None = None
    chunks, text, p_tok, c_tok = 0, "", 0, 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
            stream=True,
            timeout=timeout,
        )
        for chunk in stream:
            # getattr, as before: a chunk object without a usage attribute counts as no usage.
            usage = getattr(chunk, "usage", None)
            if usage:
                p_tok = usage.prompt_tokens or 0
                c_tok = usage.completion_tokens or 0
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                if content:
                    if ttft is None:
                        ttft = (clock() - start) * 1000
                        gen_start = clock()
                    text += content
                    chunks += 1
        total = (clock() - start) * 1000
        # Truthiness, not 'is not None', exactly as before.
        gen = (clock() - gen_start) * 1000 if gen_start else 0
        p_tok = p_tok or len(query.split())
        c_tok = c_tok or len(text.split())
        return CallResult(
            latency_ms=round(total, 2),
            ttft_ms=round(ttft or 0, 2),
            generation_time_ms=round(gen, 2),
            tokens_per_second=round(chunks / (gen / 1000), 2) if gen > 0 else 0,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=p_tok + c_tok,
            response=text[:RESPONSE_CHARS],
            success=True,
            error=None,
        )
    except Exception as exc:
        return CallResult.failure(exc)


def route_prompt(
    client: OpenAI,
    prompt: Prompt,
    method: RoutingMethod,
    models: TierModels,
    timeout: int,
    *,
    clock: Callable[[], float] = time.time,
) -> RouteRecord:
    """Classify one prompt, pick its model and run it; the record keeps the first 200 characters of the question.

    The model column holds the model key (llama3-small, qwen3 or deepseek); the request goes to its served name.
    """
    q = prompt.question
    tier = classify_keyword(q) if method == "keyword" else classify_llm(client, q, model=models.classifier)
    key = route_to_model(tier)
    result = call_model(client, models.served(key), q, timeout, clock=clock)
    return RouteRecord(prompt.qid, q[:QUESTION_CHARS], str(tier), method, key, **dataclasses.asdict(result))


def run_routing(
    client: OpenAI,
    prompts: Sequence[Prompt],
    method: RoutingMethod,
    models: TierModels,
    *,
    workers: int = DEFAULT_WORKERS,
    timeout: int = DEFAULT_TIMEOUT_S,
    clock: Callable[[], float] = time.time,
) -> list[RouteRecord]:
    """Route prompts on a thread pool, logging progress every 200 completions; records come back in prompt order.

    All workers share client. A failed request is a record with success False, not an exception.
    """
    rows: dict[int, RouteRecord] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(route_prompt, client, p, method, models, timeout, clock=clock): i for i, p in enumerate(prompts)
        }
        for n, fut in enumerate(as_completed(futs), 1):
            rows[futs[fut]] = fut.result()
            if n % PROGRESS_EVERY == 0:
                log.info("%d/%d", n, len(prompts))
    return [rows[i] for i in sorted(rows)]


def output_path(out_dir: Path, benchmark: str, method: RoutingMethod) -> Path:
    """Return out_dir/<method>/<benchmark>_<method>.csv."""
    return out_dir / method / f"{benchmark}_{method}.csv"


def write_rows(path: Path, rows: Iterable[RouteRecord]) -> Path:
    """Write records as the 15-column CSV and return path.

    Values keep their Python types: success is written as True or False, an error of None as '', and an int 0
    as '0'.
    """
    return write_dicts(path, FIELDS, (r.as_row() for r in rows))
