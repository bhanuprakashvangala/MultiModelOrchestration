"""The scorer configuration: the YAML parsed into frozen dataclasses that keep document order.

- Keys the legacy code read with [] are required; a KeyError names the missing key.
- Keys read with .get keep their defaults (use_llm False, llm_model 'gemma3').
- Keys the code never read (api.timeout, api_name, description, simple_indicators, fallback_to_keywords) stay in
  the file and are ignored.
- No coercion: values keep their YAML types, so a weight written as 1 stays the int 1. Lists become tuples.

Mapping order is behaviour: the model order breaks the balanced/complex tie. The packaged config.yaml is read
through importlib.resources, and yaml is imported lazily (extra 'live').
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mmorch.errors import MissingDependencyError
from mmorch.settings import require_file


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One model's relative scores in [0, 1] and its size tier.

    name is the model's key in the YAML, which is also the model name sent in requests; the YAML's api_name is
    never read.
    """

    name: str
    params: str
    tier: str
    quality_weight: float
    speed_weight: float
    cost_weight: float


@dataclass(frozen=True, slots=True)
class Weights:
    """The Eq. 2 weights of an operator profile: alpha (quality), lambda (speed) and mu (cost)."""

    quality: float
    speed: float
    cost: float


@dataclass(frozen=True, slots=True)
class Strategy:
    """An operator profile (routing strategy) and its weights."""

    name: str
    weights: Weights


@dataclass(frozen=True, slots=True)
class ComplexityRules:
    """The scorer's own complexity detection: keyword indicators, token budgets and the optional LLM classifier.

    use_llm keeps its YAML value and is tested for truthiness, as before.
    """

    medium_indicators: tuple[str, ...]
    complex_indicators: tuple[str, ...]
    simple_max_tokens: int
    medium_max_tokens: int
    complex_max_tokens: int
    use_llm: bool = False
    llm_model: str = "gemma3"


@dataclass(frozen=True)
class ScorerConfig:
    """The whole scorer configuration; models and strategies are read-only mappings in document order."""

    api_base_url: str
    models: Mapping[str, ModelProfile]
    strategies: Mapping[str, Strategy]
    complexity: ComplexityRules
    privacy_keywords: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ScorerConfig:
        """Build the configuration from a parsed YAML document; a missing required key raises KeyError.

        Pure: nothing is read from files or the environment.
        """
        return cls(
            api_base_url=raw["api"]["base_url"],
            models=MappingProxyType({name: _model_profile(name, model) for name, model in raw["models"].items()}),
            strategies=MappingProxyType(
                {name: _strategy(name, strategy) for name, strategy in raw["routing_strategies"].items()}
            ),
            complexity=_complexity_rules(raw["complexity_detection"]),
            privacy_keywords=tuple(raw["privacy"]["sensitive_keywords"]),
        )


def _model_profile(name: str, raw: Mapping[str, Any]) -> ModelProfile:
    return ModelProfile(
        name=name,
        params=raw["params"],
        tier=raw["tier"],
        quality_weight=raw["quality_weight"],
        speed_weight=raw["speed_weight"],
        cost_weight=raw["cost_weight"],
    )


def _strategy(name: str, raw: Mapping[str, Any]) -> Strategy:
    weights = raw["weights"]
    return Strategy(name, Weights(quality=weights["quality"], speed=weights["speed"], cost=weights["cost"]))


def _complexity_rules(raw: Mapping[str, Any]) -> ComplexityRules:
    return ComplexityRules(
        medium_indicators=tuple(raw["medium_indicators"]),
        complex_indicators=tuple(raw["complex_indicators"]),
        simple_max_tokens=raw["simple_max_tokens"],
        medium_max_tokens=raw["medium_max_tokens"],
        complex_max_tokens=raw["complex_max_tokens"],
        use_llm=raw.get("use_llm", False),
        llm_model=raw.get("llm_model", "gemma3"),
    )


def load_config(path: Path | None = None) -> ScorerConfig:
    """Load the scorer YAML from path, or the packaged config.yaml when path is None.

    The text is read as UTF-8 and parsed with yaml.safe_load. yaml comes with the 'live' extra; without it this
    raises MissingDependencyError. A path that is not an existing file raises DataNotFoundError.
    """
    try:
        import yaml
    except ImportError as exc:
        raise MissingDependencyError("pyyaml", "live") from exc
    if path is None:
        text = files("mmorch.scoring").joinpath("config.yaml").read_text(encoding="utf-8")
    else:
        text = require_file(path, "scorer config").read_text(encoding="utf-8")
    return ScorerConfig.from_mapping(yaml.safe_load(text))
