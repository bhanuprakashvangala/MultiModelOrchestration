"""MultiObjectiveRouter: the stateful router around the pure scorer, with the legacy public method names.

It detects query complexity with keywords, or with the LLM and a keyword fallback; selects a model per operator
profile ('baseline' is a random choice over the config order); executes the query non-streaming; redacts
privacy-sensitive queries; keeps usage counts and history; and runs the three-query demo behind `mmorch score`.

Behaviour kept from the prototype on purpose:

- execute_query detects the complexity a second time when max_tokens is None, so with use_llm set every query
  costs two classifier calls before the completion.
- The request model is the config key, and there is no request timeout (the SDK default applies).
- The model is counted as soon as the call returns, before the response is checked, so a reply without content
  still counts although its result records a failure.

The legacy initialisation and warning prints are log records with the same text. openai is imported only for
type checking and inside make_client.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, cast

from mmorch.errors import MissingDependencyError
from mmorch.scoring.config import ModelProfile, ScorerConfig, Strategy
from mmorch.scoring.scorer import (
    build_rationale,
    complexity_prompt,
    detect_complexity_keywords,
    is_privacy_sensitive,
    parse_complexity_reply,
    score_models,
)
from mmorch.settings import EndpointSettings

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Represents a routing decision with full transparency."""

    selected_model: str
    strategy: str
    query_complexity: str
    is_privacy_sensitive: bool
    model_tier: str
    quality_score: float
    speed_score: float
    cost_score: float
    combined_score: float
    alternatives_considered: list[dict[str, Any]]
    decision_rationale: str
    timestamp: str


@dataclass
class QueryResult:
    """Complete result from a query execution."""

    query_id: str
    query: str
    routing_decision: RoutingDecision
    response: str
    latency_ms: float
    tokens_used: int
    cost_score: float
    success: bool
    error: str | None = None
    timestamp: str | None = None


def make_client(endpoint: EndpointSettings, default_base_url: str) -> OpenAI:
    """Create the client: LLM_API_KEY if set (even empty) else 'none'; LLM_API_BASE if set else default_base_url.

    default_base_url is the YAML's api.base_url. The SDK defaults (retries, timeout) stay untouched.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise MissingDependencyError("openai", "live") from exc
    return OpenAI(
        api_key=endpoint.api_key if endpoint.api_key is not None else "none",
        base_url=endpoint.api_base if endpoint.api_base is not None else default_base_url,
    )


class MultiObjectiveRouter:
    """Pick-and-Spin multi-objective routing over the configured models and operator profiles."""

    config: ScorerConfig
    client: OpenAI
    models: Mapping[str, ModelProfile]
    strategies: Mapping[str, Strategy]
    query_history: list[QueryResult]
    model_usage_counts: dict[str, int]

    def __init__(
        self,
        config: ScorerConfig,
        client: OpenAI,
        *,
        rng: random.Random | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """Keep the configuration and client, zero the usage counts in config order and log the three init lines.

        rng None means the module-level random.choice for the 'baseline' strategy; now stamps the timestamps.
        """
        self.config = config
        self.client = client
        self.models = config.models
        self.strategies = config.strategies

        # Initialize tracking
        self.query_history = []
        self.model_usage_counts = {model: 0 for model in self.models}

        self._rng = rng
        self._now = now

        log.info("> Pick-and-Spin Router initialized")
        log.info("> Available models: %s", list(self.models))
        log.info("> Available strategies: %s", list(self.strategies))

    def detect_query_complexity_llm(self, query: str) -> tuple[str, int]:
        """Classify complexity with the configured LLM, falling back to the keyword rules on any problem.

        One non-streaming call with temperature 0.0, max_tokens 10 and no timeout. The reply is matched by
        substring; an unparseable reply or any exception logs a warning and uses the keyword rules.
        """
        rules = self.config.complexity
        prompt = complexity_prompt(query)

        try:
            completion = self.client.chat.completions.create(
                model=rules.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=10,  # Only need one word response
            )

            # A None content fails on .strip(), and the except below falls back to the keywords, as before.
            response = cast(str, completion.choices[0].message.content).strip().upper()

            parsed = parse_complexity_reply(response, rules)
            if parsed is not None:
                return parsed
            # Unparseable response - fall back to keyword method
            log.warning("  [Warning] LLM returned unparseable complexity: '%s', falling back to keywords", response)
            return detect_complexity_keywords(query, rules)

        except Exception as exc:
            # API error - fall back to keyword method
            log.warning("  [Warning] LLM complexity detection failed: %s, falling back to keywords", exc)
            return detect_complexity_keywords(query, rules)

    def detect_query_complexity(self, query: str) -> tuple[str, int]:
        """Classify complexity with the LLM when use_llm is set, otherwise with the keyword rules.

        Returns (complexity, suggested max_tokens).
        """
        if self.config.complexity.use_llm:
            return self.detect_query_complexity_llm(query)
        return detect_complexity_keywords(query, self.config.complexity)

    def detect_privacy_sensitive(self, query: str) -> bool:
        """Return whether the query contains a privacy-sensitive keyword."""
        return is_privacy_sensitive(query, self.config.privacy_keywords)

    def calculate_model_scores(self, complexity: str, strategy: str) -> list[dict[str, Any]]:
        """Return every model's scores as dicts, best combined score first."""
        return [score.as_dict() for score in score_models(self.config, complexity, strategy)]

    def select_model(self, query: str, strategy: str = "balanced", explain: bool = True) -> RoutingDecision:
        """Select the model for a query under an operator profile and explain the choice.

        'baseline' picks a model at random with zero scores. Every other strategy takes the best-scoring model and
        lists the next three as alternatives; an unknown strategy raises KeyError.
        """
        # Analyze query
        complexity, _suggested_tokens = self.detect_query_complexity(query)
        is_sensitive = self.detect_privacy_sensitive(query)

        # Handle baseline (random) strategy
        if strategy == "baseline":
            selected_model = self._choose(list(self.models))
            model_config = self.models[selected_model]

            return RoutingDecision(
                selected_model=selected_model,
                strategy=strategy,
                query_complexity=complexity,
                is_privacy_sensitive=is_sensitive,
                model_tier=model_config.tier,
                quality_score=0.0,
                speed_score=0.0,
                cost_score=0.0,
                combined_score=0.0,
                alternatives_considered=[],
                decision_rationale="Random selection (baseline strategy)",
                timestamp=self._now().isoformat(),
            )

        # Calculate scores for all models
        scored_models = score_models(self.config, complexity, strategy)

        # Select top model
        top_choice = scored_models[0]
        selected_model = top_choice.model

        # Generate explanation
        rationale = (
            build_rationale(self.config, strategy, complexity, selected_model, top_choice, scored_models[1:3])
            if explain
            else f"Selected {selected_model} using {strategy} strategy"
        )

        return RoutingDecision(
            selected_model=selected_model,
            strategy=strategy,
            query_complexity=complexity,
            is_privacy_sensitive=is_sensitive,
            model_tier=top_choice.tier,
            quality_score=top_choice.quality_score,
            speed_score=top_choice.speed_score,
            cost_score=top_choice.cost_score,
            combined_score=top_choice.combined_score,
            alternatives_considered=[score.as_dict() for score in scored_models[1:4]],  # Top 3 alternatives
            decision_rationale=rationale,
            timestamp=self._now().isoformat(),
        )

    def execute_query(
        self,
        query_id: str,
        query: str,
        strategy: str = "balanced",
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> QueryResult:
        """Route and run one query non-streaming, record it in the history and return the result.

        Any exception from the call gives a failed result with the error text. A privacy-sensitive query is stored
        as '[REDACTED]', and a response longer than 200 characters is cut to 200 plus '...'.
        """
        # Make routing decision
        routing_decision = self.select_model(query, strategy, explain=True)

        # Determine max tokens
        if max_tokens is None:
            _, max_tokens = self.detect_query_complexity(query)

        # Execute query
        start_time = time.time()

        try:
            completion = self.client.chat.completions.create(
                model=routing_decision.selected_model,
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000
            # A None content fails on len() below and becomes a failed result, as before.
            response = cast(str, completion.choices[0].message.content)
            tokens_used = completion.usage.total_tokens if completion.usage else 0

            # Calculate cost score
            model_cost_weight = self.models[routing_decision.selected_model].cost_weight
            cost_score = tokens_used * model_cost_weight

            # Update tracking
            self.model_usage_counts[routing_decision.selected_model] += 1

            result = QueryResult(
                query_id=query_id,
                query=query if not routing_decision.is_privacy_sensitive else "[REDACTED]",
                routing_decision=routing_decision,
                response=response[:200] + "..." if len(response) > 200 else response,
                latency_ms=round(latency_ms, 2),
                tokens_used=tokens_used,
                cost_score=round(cost_score, 2),
                success=True,
                timestamp=self._now().isoformat(),
            )

        except Exception as exc:
            latency_ms = (time.time() - start_time) * 1000

            result = QueryResult(
                query_id=query_id,
                query=query if not routing_decision.is_privacy_sensitive else "[REDACTED]",
                routing_decision=routing_decision,
                response="",
                latency_ms=round(latency_ms, 2),
                tokens_used=0,
                cost_score=0,
                success=False,
                error=str(exc),
                timestamp=self._now().isoformat(),
            )

        # Store in history
        self.query_history.append(result)

        return result

    def get_usage_statistics(self) -> dict[str, Any]:
        """Return the total query count, the per-model usage counts and their distribution.

        model_usage is the router's own counts dict, not a copy. Before any query every share is the int 0.
        """
        total_queries = sum(self.model_usage_counts.values())

        return {
            "total_queries": total_queries,
            "model_usage": self.model_usage_counts,
            "model_distribution": {
                model: count / total_queries if total_queries > 0 else 0
                for model, count in self.model_usage_counts.items()
            },
        }

    def _choose(self, models: list[str]) -> str:
        """Pick a model for the 'baseline' strategy: the injected rng, else the module-level random.choice."""
        if self._rng is None:
            return random.choice(models)
        return self._rng.choice(models)


# The legacy demo: one query per strategy, run in this order.
DEMO_QUERIES: Final[tuple[tuple[str, str], ...]] = (
    ("What is the capital of France?", "speed"),
    ("Explain quantum computing in detail", "quality"),
    ("What are common treatments for diabetes?", "balanced"),
)


def run_demo(router: MultiObjectiveRouter, emit: Callable[[str], None]) -> None:
    """Run the three demo queries and emit the legacy report line by line (the command line passes print).

    The lines are those of the legacy main() minus the three initialisation lines, which the router logs.
    """
    emit("\n" + "=" * 80)
    emit("PICK-AND-SPIN ROUTING SYSTEM TEST")
    emit("=" * 80 + "\n")

    for i, (query, strategy) in enumerate(DEMO_QUERIES, 1):
        emit(f"\n[Test {i}] Strategy: {strategy}")
        emit(f"Query: {query}\n")

        result = router.execute_query(query_id=f"test_{i}", query=query, strategy=strategy)

        if result.success:
            emit(f"> Model: {result.routing_decision.selected_model}")
            emit(f"> Complexity: {result.routing_decision.query_complexity}")
            emit(f"> Privacy: {result.routing_decision.is_privacy_sensitive}")
            emit(f"> Latency: {result.latency_ms:.0f}ms")
            emit(f"> Tokens: {result.tokens_used}")
            emit(f"> Rationale: {result.routing_decision.decision_rationale}")
        else:
            emit(f"X Error: {result.error}")

    # Print statistics
    emit("\n" + "=" * 80)
    stats = router.get_usage_statistics()
    emit(f"Total queries: {stats['total_queries']}")
    emit(f"Model usage: {stats['model_usage']}")
