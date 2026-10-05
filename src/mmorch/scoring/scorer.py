"""The pure Eq. 2 logic of the multi-objective scorer.

- Per-model quality, speed and inverted-cost scores, with the complexity penalties: a simple query halves the
  cost score of xlarge and large models, and a complex query scales the quality of small models by 0.6.
- The weighted sum, Python round(x, 3), and a stable descending sort on the rounded combined score, in config
  order. Stability is what breaks ties: in the exact balanced/complex tie (qwen3 = gemma3 = 0.69) qwen3 stays first
  because it comes first in the YAML.

Also holds the scorer's own keyword complexity rules (deliberately different from mmorch.routing's), the LLM
complexity prompt and its substring parse, privacy detection and the rationale text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mmorch.scoring.config import ComplexityRules, ScorerConfig


@dataclass(frozen=True, slots=True)
class ModelScore:
    """One model's rounded scores for a query complexity and strategy."""

    model: str
    tier: str
    quality_score: float
    speed_score: float
    cost_score: float
    combined_score: float

    def as_dict(self) -> dict[str, Any]:
        """Return the scores as a dict in the legacy key order."""
        return {
            "model": self.model,
            "tier": self.tier,
            "quality_score": self.quality_score,
            "speed_score": self.speed_score,
            "cost_score": self.cost_score,
            "combined_score": self.combined_score,
        }


def score_models(config: ScorerConfig, complexity: str, strategy: str) -> list[ModelScore]:
    """Score every model in config order and sort by combined score, highest first (stable, so ties keep order).

    An unknown strategy raises KeyError.
    """
    weights = config.strategies[strategy].weights

    scored_models: list[ModelScore] = []
    for model_name, model in config.models.items():
        # Base scores from the configuration; the cost score is inverted, so a cheaper model scores higher.
        quality_score = model.quality_weight
        speed_score = model.speed_weight
        cost_score = 1.0 - model.cost_weight

        if complexity == "simple" and model.tier in ["xlarge", "large"]:
            # Penalize oversized models for simple queries.
            cost_score *= 0.5
        elif complexity == "complex" and model.tier in ["small"]:
            # Penalize undersized models for complex queries.
            quality_score *= 0.6

        combined_score = weights.quality * quality_score + weights.speed * speed_score + weights.cost * cost_score

        scored_models.append(
            ModelScore(
                model=model_name,
                tier=model.tier,
                quality_score=round(quality_score, 3),
                speed_score=round(speed_score, 3),
                cost_score=round(cost_score, 3),
                combined_score=round(combined_score, 3),
            )
        )

    scored_models.sort(key=lambda score: score.combined_score, reverse=True)
    return scored_models


def detect_complexity_keywords(query: str, rules: ComplexityRules) -> tuple[str, int]:
    """Return (complexity, max_tokens): complex if a complex indicator matches, else medium, else simple.

    Indicators match as substrings of the lowercased query.
    """
    query_lower = query.lower()
    if any(indicator in query_lower for indicator in rules.complex_indicators):
        return "complex", rules.complex_max_tokens
    if any(indicator in query_lower for indicator in rules.medium_indicators):
        return "medium", rules.medium_max_tokens
    return "simple", rules.simple_max_tokens


def complexity_prompt(query: str) -> str:
    """Return the legacy SIMPLE / MEDIUM / COMPLEX classification prompt with the full query."""
    return (
        "Analyze the complexity of this question and classify it as SIMPLE, MEDIUM, or COMPLEX.\n"
        "\n"
        '- SIMPLE: Basic facts, definitions, arithmetic, short answers (e.g., "What is 2+2?", '
        '"Define photosynthesis")\n'
        '- MEDIUM: Explanations, comparisons, moderate reasoning (e.g., "Explain how photosynthesis works", '
        '"Compare X and Y")\n'
        "- COMPLEX: Deep analysis, multi-step reasoning, synthesis, technical depth "
        '(e.g., "Analyze the implications of...", "Evaluate the relationship between...")\n'
        "\n"
        f"Question: {query}\n"
        "\n"
        "Respond with ONLY ONE WORD: SIMPLE, MEDIUM, or COMPLEX"
    )


def parse_complexity_reply(reply: str, rules: ComplexityRules) -> tuple[str, int] | None:
    """Map an upper-cased reply to (complexity, max_tokens) by substring, SIMPLE then MEDIUM then COMPLEX; else None.

    The caller strips and upper-cases the reply first.
    """
    if "SIMPLE" in reply:
        return "simple", rules.simple_max_tokens
    if "MEDIUM" in reply:
        return "medium", rules.medium_max_tokens
    if "COMPLEX" in reply:
        return "complex", rules.complex_max_tokens
    return None


def is_privacy_sensitive(query: str, keywords: Sequence[str]) -> bool:
    """Return whether the lowercased query contains any of the privacy keywords."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)


def build_rationale(
    config: ScorerConfig,
    strategy: str,
    complexity: str,
    selected_model: str,
    top: ModelScore,
    alternatives: Sequence[ModelScore],
) -> str:
    """Return the legacy human-readable explanation of a routing decision.

    The text names the selected model with its parameter count and tier, gives the score that matters for the
    strategy, comments on the complexity and lists the first two alternatives.
    """
    model_params = config.models[selected_model].params
    model_tier = top.tier

    rationale = f"Selected {selected_model} ({model_params}, {model_tier}-tier) for {strategy} strategy. "

    # Strategy-specific explanation
    if strategy == "quality":
        rationale += f"This model offers highest quality (score: {top.quality_score:.2f}). "
    elif strategy == "speed":
        rationale += f"This model offers fastest response (speed score: {top.speed_score:.2f}). "
    elif strategy == "cost":
        rationale += f"This model is most cost-effective (cost score: {top.cost_score:.2f}). "
    elif strategy == "balanced":
        rationale += f"This model balances quality, speed, and cost (combined: {top.combined_score:.2f}). "

    # Complexity consideration
    if complexity == "simple":
        rationale += "Query complexity is low, small model sufficient. "
    elif complexity == "complex":
        rationale += "Query complexity is high, larger model recommended. "

    # Alternatives
    if alternatives:
        alt_names = [a.model for a in alternatives[:2]]
        rationale += f"Alternatives considered: {', '.join(alt_names)}."

    return rationale
