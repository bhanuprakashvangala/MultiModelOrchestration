"""Tests for mmorch.scoring: the scorer YAML, the Eq. 2 ranking, the scorer's own rules and MultiObjectiveRouter.

The ranking table, the classification prompt, the rationale texts, the demo report and the request shapes are
literals captured from the legacy src/routing/multi_objective.py (tag v1.0.0). Nothing here opens a socket: the
router runs against a fake client, and the exact request bodies come from the real OpenAI SDK through an in-process
httpx.MockTransport.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import random
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from mmorch.errors import ConfigError, DataNotFoundError, MissingDependencyError
from mmorch.scoring import (
    ModelScore,
    MultiObjectiveRouter,
    QueryResult,
    RoutingDecision,
    ScorerConfig,
    load_config,
    score_models,
)
from mmorch.scoring.config import ComplexityRules, ModelProfile, Strategy, Weights
from mmorch.scoring.router import DEMO_QUERIES, make_client, run_demo
from mmorch.scoring.scorer import (
    build_rationale,
    complexity_prompt,
    detect_complexity_keywords,
    is_privacy_sensitive,
    parse_complexity_reply,
)
from mmorch.settings import EndpointSettings

# A literal copy of the legacy src/routing/config.yaml, including the keys the code never read.
LEGACY_YAML: dict[str, Any] = {
    "api": {"base_url": "https://your-llm-endpoint.example.org/v1", "timeout": 60},
    "models": {
        "qwen3": {
            "api_name": "qwen3",
            "params": "235B",
            "cost_weight": 0.8,
            "speed_weight": 0.6,
            "quality_weight": 0.9,
            "tier": "large",
        },
        "gemma3": {
            "api_name": "gemma3",
            "params": "27B",
            "cost_weight": 0.3,
            "speed_weight": 0.9,
            "quality_weight": 0.7,
            "tier": "small",
        },
        "llama3": {
            "api_name": "llama3",
            "params": "90B",
            "cost_weight": 0.5,
            "speed_weight": 0.7,
            "quality_weight": 0.8,
            "tier": "medium",
        },
    },
    "routing_strategies": {
        "quality": {"description": "Prioritize output quality", "weights": {"quality": 1.0, "speed": 0.1, "cost": 0.1}},
        "cost": {
            "description": "Prioritize resource efficiency",
            "weights": {"quality": 0.3, "speed": 0.2, "cost": 0.8},
        },
        "speed": {"description": "Prioritize response time", "weights": {"quality": 0.3, "speed": 0.8, "cost": 0.2}},
        "balanced": {
            "description": "Balance quality, speed, and cost",
            "weights": {"quality": 0.5, "speed": 0.3, "cost": 0.3},
        },
        "baseline": {"description": "Random selection", "weights": {"quality": 0.33, "speed": 0.33, "cost": 0.33}},
    },
    "complexity_detection": {
        "use_llm": False,
        "llm_model": "gemma3",
        "fallback_to_keywords": True,
        "simple_indicators": ["what is", "how many", "define", "name", "list"],
        "medium_indicators": ["explain", "compare", "describe", "how does"],
        "complex_indicators": ["analyze", "discuss", "evaluate", "implications", "mathematical foundations"],
        "simple_max_tokens": 100,
        "medium_max_tokens": 300,
        "complex_max_tokens": 500,
    },
    "privacy": {
        "sensitive_keywords": ["medical", "health", "diagnosis", "treatment", "patient", "confidential", "personal"]
    },
}

CONFIG = ScorerConfig.from_mapping(LEGACY_YAML)
MODEL_ORDER = ["qwen3", "gemma3", "llama3"]
STRATEGY_ORDER = ["quality", "cost", "speed", "balanced", "baseline"]
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, 678901)

# The legacy classification prompt for the question 'Explain Y', line by line.
LEGACY_PROMPT_EXPLAIN_Y = "\n".join(
    [
        "Analyze the complexity of this question and classify it as SIMPLE, MEDIUM, or COMPLEX.",
        "",
        '- SIMPLE: Basic facts, definitions, arithmetic, short answers (e.g., "What is 2+2?", "Define photosynthesis")',
        "- MEDIUM: Explanations, comparisons, moderate reasoning "
        '(e.g., "Explain how photosynthesis works", "Compare X and Y")',
        "- COMPLEX: Deep analysis, multi-step reasoning, synthesis, technical depth "
        '(e.g., "Analyze the implications of...", "Evaluate the relationship between...")',
        "",
        "Question: Explain Y",
        "",
        "Respond with ONLY ONE WORD: SIMPLE, MEDIUM, or COMPLEX",
    ]
)


def raw_config(**complexity: Any) -> dict[str, Any]:
    """A deep copy of LEGACY_YAML, with complexity_detection keys replaced by the given values."""
    raw = copy.deepcopy(LEGACY_YAML)
    raw["complexity_detection"].update(complexity)
    return raw


def typed(value: Any) -> Any:
    """value with every mapping turned into (key, value) pairs and every scalar paired with its type.

    Comparing typed() results checks key order and value types as well as values: 1 == 1.0, but not here.
    """
    if isinstance(value, Mapping):
        return [(key, typed(item)) for key, item in value.items()]
    if isinstance(value, list | tuple):
        return [typed(item) for item in value]
    return (type(value).__name__, value)


@pytest.fixture
def mmorch_logs(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog capturing the 'mmorch' loggers from INFO, even after the command line's logging setup.

    mmorch.cli.configure_logging stops propagation on the 'mmorch' logger, which would hide its records from
    caplog's root handler; this restores propagation for the test.
    """
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.INFO, logger="mmorch")
    return caplog


# ---------------------------------------------------------------- configuration


def test_packaged_yaml_holds_the_legacy_values() -> None:
    yaml = pytest.importorskip("yaml")
    resource = files("mmorch.scoring").joinpath("config.yaml")
    assert resource.is_file()
    text = resource.read_text(encoding="utf-8")
    assert typed(yaml.safe_load(text)) == typed(LEGACY_YAML)
    # Only the header comment changed in the move; it now points at `mmorch score --config`.
    assert "mmorch score --config" in text.splitlines()[0]


def test_load_config_reads_the_packaged_yaml() -> None:
    pytest.importorskip("yaml")
    config = load_config()
    assert config == CONFIG
    assert list(config.models) == MODEL_ORDER


def test_load_config_reads_the_packaged_file_through_importlib_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("yaml")
    seen: list[str] = []

    class Package:
        def joinpath(self, name: str) -> Package:
            seen.append(name)
            return self

        def read_text(self, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps(raw_config(llm_model="from-resources"))

    def fake_files(anchor: str) -> Package:
        seen.append(anchor)
        return Package()

    monkeypatch.setattr("mmorch.scoring.config.files", fake_files)
    assert load_config().complexity.llm_model == "from-resources"
    assert seen == ["mmorch.scoring", "config.yaml"]


def test_config_keeps_the_document_order() -> None:
    assert list(CONFIG.models) == MODEL_ORDER
    assert list(CONFIG.strategies) == STRATEGY_ORDER
    assert isinstance(CONFIG.models, MappingProxyType)
    assert isinstance(CONFIG.strategies, MappingProxyType)


def test_config_values_equal_the_legacy_yaml() -> None:
    assert CONFIG.api_base_url == "https://your-llm-endpoint.example.org/v1"
    assert dict(CONFIG.models) == {
        "qwen3": ModelProfile("qwen3", "235B", "large", 0.9, 0.6, 0.8),
        "gemma3": ModelProfile("gemma3", "27B", "small", 0.7, 0.9, 0.3),
        "llama3": ModelProfile("llama3", "90B", "medium", 0.8, 0.7, 0.5),
    }
    assert dict(CONFIG.strategies) == {
        "quality": Strategy("quality", Weights(1.0, 0.1, 0.1)),
        "cost": Strategy("cost", Weights(0.3, 0.2, 0.8)),
        "speed": Strategy("speed", Weights(0.3, 0.8, 0.2)),
        "balanced": Strategy("balanced", Weights(0.5, 0.3, 0.3)),
        "baseline": Strategy("baseline", Weights(0.33, 0.33, 0.33)),
    }
    assert CONFIG.complexity == ComplexityRules(
        medium_indicators=("explain", "compare", "describe", "how does"),
        complex_indicators=("analyze", "discuss", "evaluate", "implications", "mathematical foundations"),
        simple_max_tokens=100,
        medium_max_tokens=300,
        complex_max_tokens=500,
        use_llm=False,
        llm_model="gemma3",
    )
    assert CONFIG.privacy_keywords == (
        "medical",
        "health",
        "diagnosis",
        "treatment",
        "patient",
        "confidential",
        "personal",
    )


def test_config_is_read_only() -> None:
    with pytest.raises(TypeError):
        CONFIG.models["extra"] = CONFIG.models["qwen3"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.api_base_url = "x"


def test_config_keeps_yaml_types() -> None:
    raw = raw_config(use_llm="yes", simple_max_tokens=100.0)
    raw["models"]["qwen3"]["quality_weight"] = 1
    raw["routing_strategies"]["quality"]["weights"]["quality"] = 1
    config = ScorerConfig.from_mapping(raw)
    assert typed(config.models["qwen3"].quality_weight) == ("int", 1)
    assert typed(config.strategies["quality"].weights.quality) == ("int", 1)
    assert typed(config.complexity.simple_max_tokens) == ("float", 100.0)
    assert config.complexity.use_llm == "yes"  # tested for truthiness, as before
    assert typed(CONFIG.complexity.medium_max_tokens) == ("int", 300)
    assert typed(CONFIG.models["gemma3"].cost_weight) == ("float", 0.3)


def test_optional_keys_take_the_legacy_defaults() -> None:
    raw = raw_config()
    del raw["complexity_detection"]["use_llm"]
    del raw["complexity_detection"]["llm_model"]
    rules = ScorerConfig.from_mapping(raw).complexity
    assert rules.use_llm is False
    assert rules.llm_model == "gemma3"


def test_keys_the_code_never_read_are_ignored() -> None:
    raw = raw_config()
    del raw["api"]["timeout"]
    for model in raw["models"].values():
        del model["api_name"]
    for strategy in raw["routing_strategies"].values():
        del strategy["description"]
    del raw["complexity_detection"]["simple_indicators"]
    del raw["complexity_detection"]["fallback_to_keywords"]
    raw["unknown"] = {"anything": 1}
    assert ScorerConfig.from_mapping(raw) == CONFIG


@pytest.mark.parametrize(
    "path",
    [
        ("api",),
        ("api", "base_url"),
        ("models",),
        ("models", "qwen3", "params"),
        ("models", "gemma3", "tier"),
        ("models", "llama3", "quality_weight"),
        ("routing_strategies",),
        ("routing_strategies", "balanced", "weights"),
        ("routing_strategies", "cost", "weights", "speed"),
        ("complexity_detection",),
        ("complexity_detection", "medium_indicators"),
        ("complexity_detection", "complex_max_tokens"),
        ("privacy", "sensitive_keywords"),
    ],
)
def test_a_missing_required_key_raises_key_error(path: tuple[str, ...]) -> None:
    raw = raw_config()
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    with pytest.raises(KeyError) as excinfo:
        ScorerConfig.from_mapping(raw)
    assert excinfo.value.args == (path[-1],)


def test_load_config_reads_a_given_file(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    raw = raw_config(use_llm=True, llm_model="tiny")
    raw["models"] = {name: raw["models"][name] for name in ("llama3", "gemma3", "qwen3")}
    path = tmp_path / "scorer.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(path)
    assert list(config.models) == ["llama3", "gemma3", "qwen3"]
    assert config.complexity.use_llm is True
    assert config.complexity.llm_model == "tiny"


def test_load_config_reads_utf8(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    raw = raw_config()
    raw["models"]["qwen3"]["params"] = "235B Größe"  # not decodable as cp1252 round trip
    path = tmp_path / "scorer.yaml"
    path.write_bytes(json.dumps(raw, ensure_ascii=False).encode("utf-8"))
    assert load_config(path).models["qwen3"].params == "235B Größe"


def test_load_config_names_a_missing_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "missing.yaml"
    with pytest.raises(DataNotFoundError, match="scorer config not found") as excinfo:
        load_config(path)
    assert str(path) in str(excinfo.value)


def test_load_config_names_a_missing_key(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "scorer.yaml"
    path.write_text("api: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key 'base_url'") as excinfo:
        load_config(path)
    assert str(path) in str(excinfo.value)


def test_load_config_rejects_a_document_that_is_not_a_mapping(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "scorer.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unexpected structure"):
        load_config(path)


def test_load_config_without_yaml_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(MissingDependencyError) as excinfo:
        load_config()
    assert str(excinfo.value) == 'pyyaml is required for this command: pip install -e ".[live]"'
    assert (excinfo.value.package, excinfo.value.extra) == ("pyyaml", "live")


# ---------------------------------------------------------------- the Eq. 2 ranking

# Captured from the legacy calculate_model_scores on the packaged config: per (strategy, complexity), the models
# best first as (model, quality_score, speed_score, cost_score, combined_score).
GOLDEN_RANKING: dict[tuple[str, str], list[tuple[str, float, float, float, float]]] = {
    ("quality", "simple"): [
        ("qwen3", 0.9, 0.6, 0.1, 0.97),
        ("llama3", 0.8, 0.7, 0.5, 0.92),
        ("gemma3", 0.7, 0.9, 0.7, 0.86),
    ],
    ("quality", "medium"): [
        ("qwen3", 0.9, 0.6, 0.2, 0.98),
        ("llama3", 0.8, 0.7, 0.5, 0.92),
        ("gemma3", 0.7, 0.9, 0.7, 0.86),
    ],
    ("quality", "complex"): [
        ("qwen3", 0.9, 0.6, 0.2, 0.98),
        ("llama3", 0.8, 0.7, 0.5, 0.92),
        ("gemma3", 0.42, 0.9, 0.7, 0.58),
    ],
    ("cost", "simple"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.95),
        ("llama3", 0.8, 0.7, 0.5, 0.78),
        ("qwen3", 0.9, 0.6, 0.1, 0.47),
    ],
    ("cost", "medium"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.95),
        ("llama3", 0.8, 0.7, 0.5, 0.78),
        ("qwen3", 0.9, 0.6, 0.2, 0.55),
    ],
    ("cost", "complex"): [
        ("gemma3", 0.42, 0.9, 0.7, 0.866),
        ("llama3", 0.8, 0.7, 0.5, 0.78),
        ("qwen3", 0.9, 0.6, 0.2, 0.55),
    ],
    ("speed", "simple"): [
        ("gemma3", 0.7, 0.9, 0.7, 1.07),
        ("llama3", 0.8, 0.7, 0.5, 0.9),
        ("qwen3", 0.9, 0.6, 0.1, 0.77),
    ],
    ("speed", "medium"): [
        ("gemma3", 0.7, 0.9, 0.7, 1.07),
        ("llama3", 0.8, 0.7, 0.5, 0.9),
        ("qwen3", 0.9, 0.6, 0.2, 0.79),
    ],
    ("speed", "complex"): [
        ("gemma3", 0.42, 0.9, 0.7, 0.986),
        ("llama3", 0.8, 0.7, 0.5, 0.9),
        ("qwen3", 0.9, 0.6, 0.2, 0.79),
    ],
    ("balanced", "simple"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.83),
        ("llama3", 0.8, 0.7, 0.5, 0.76),
        ("qwen3", 0.9, 0.6, 0.1, 0.66),
    ],
    ("balanced", "medium"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.83),
        ("llama3", 0.8, 0.7, 0.5, 0.76),
        ("qwen3", 0.9, 0.6, 0.2, 0.69),
    ],
    # The exact tie: qwen3 and gemma3 both score 0.69, and the YAML order keeps qwen3 first.
    ("balanced", "complex"): [
        ("llama3", 0.8, 0.7, 0.5, 0.76),
        ("qwen3", 0.9, 0.6, 0.2, 0.69),
        ("gemma3", 0.42, 0.9, 0.7, 0.69),
    ],
    ("baseline", "simple"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.759),
        ("llama3", 0.8, 0.7, 0.5, 0.66),
        ("qwen3", 0.9, 0.6, 0.1, 0.528),
    ],
    ("baseline", "medium"): [
        ("gemma3", 0.7, 0.9, 0.7, 0.759),
        ("llama3", 0.8, 0.7, 0.5, 0.66),
        ("qwen3", 0.9, 0.6, 0.2, 0.561),
    ],
    ("baseline", "complex"): [
        ("gemma3", 0.42, 0.9, 0.7, 0.667),
        ("llama3", 0.8, 0.7, 0.5, 0.66),
        ("qwen3", 0.9, 0.6, 0.2, 0.561),
    ],
}
TIERS = {"qwen3": "large", "gemma3": "small", "llama3": "medium"}


@pytest.mark.parametrize(("strategy", "complexity"), list(GOLDEN_RANKING))
def test_score_models_matches_the_legacy_ranking(strategy: str, complexity: str) -> None:
    ranking = score_models(CONFIG, complexity, strategy)
    assert ranking == [
        ModelScore(model, TIERS[model], *scores) for model, *scores in GOLDEN_RANKING[strategy, complexity]
    ]


def test_the_golden_table_covers_every_profile_and_complexity() -> None:
    assert sorted(GOLDEN_RANKING) == sorted((s, c) for s in STRATEGY_ORDER for c in ("simple", "medium", "complex"))


def test_the_balanced_complex_tie_keeps_the_config_order() -> None:
    ranking = score_models(CONFIG, "complex", "balanced")
    assert [score.model for score in ranking] == ["llama3", "qwen3", "gemma3"]
    assert ranking[1].combined_score == ranking[2].combined_score == 0.69
    rationale = build_rationale(CONFIG, "balanced", "complex", "llama3", ranking[0], ranking[1:3])
    assert rationale.endswith("Alternatives considered: qwen3, gemma3.")

    # The YAML order is what breaks the tie: listing gemma3 before qwen3 swaps them.
    raw = raw_config()
    raw["models"] = {name: raw["models"][name] for name in ("gemma3", "qwen3", "llama3")}
    swapped = score_models(ScorerConfig.from_mapping(raw), "complex", "balanced")
    assert [score.model for score in swapped] == ["llama3", "gemma3", "qwen3"]


def test_scores_use_python_round_on_the_raw_sums() -> None:
    # 1.0 - 0.8 is 0.19999999999999996, shown as 0.2.
    assert score_models(CONFIG, "medium", "quality")[0].cost_score == 0.2
    # Python's round(0.6655, 3) is 0.665, while numpy's np.round gives 0.666.
    raw = raw_config()
    raw["models"] = {
        "m": {"params": "1B", "tier": "medium", "quality_weight": 0.6655, "speed_weight": 0, "cost_weight": 1}
    }
    raw["routing_strategies"] = {"q": {"weights": {"quality": 1.0, "speed": 0.0, "cost": 0.0}}}
    [score] = score_models(ScorerConfig.from_mapping(raw), "medium", "q")
    assert score == ModelScore("m", "medium", 0.665, 0, 0.0, 0.665)


def test_complexity_penalties_follow_the_tier() -> None:
    raw = raw_config()
    raw["models"] = {
        tier: {"params": "1B", "tier": tier, "quality_weight": 0.5, "speed_weight": 0.5, "cost_weight": 0.5}
        for tier in ("xlarge", "large", "medium", "small", "tiny")
    }
    config = ScorerConfig.from_mapping(raw)
    by_tier = {s.model: (s.quality_score, s.cost_score) for s in score_models(config, "simple", "quality")}
    assert by_tier == {
        "xlarge": (0.5, 0.25),
        "large": (0.5, 0.25),
        "medium": (0.5, 0.5),
        "small": (0.5, 0.5),
        "tiny": (0.5, 0.5),
    }
    by_tier = {s.model: (s.quality_score, s.cost_score) for s in score_models(config, "complex", "quality")}
    assert by_tier == {
        "xlarge": (0.5, 0.5),
        "large": (0.5, 0.5),
        "medium": (0.5, 0.5),
        "small": (0.3, 0.5),
        "tiny": (0.5, 0.5),
    }
    # Any other complexity label leaves the base scores alone.
    assert {s.quality_score for s in score_models(config, "other", "quality")} == {0.5}


def test_score_models_rejects_an_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        score_models(CONFIG, "simple", "fastest")


def test_model_score_as_dict_uses_the_legacy_key_order() -> None:
    score = ModelScore("gemma3", "small", 0.42, 0.9, 0.7, 0.69)
    assert list(score.as_dict().items()) == [
        ("model", "gemma3"),
        ("tier", "small"),
        ("quality_score", 0.42),
        ("speed_score", 0.9),
        ("cost_score", 0.7),
        ("combined_score", 0.69),
    ]


# ---------------------------------------------------------------- the scorer's own rules


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Analyze X", ("complex", 500)),
        ("Explain Y", ("medium", 300)),
        ("What is Z", ("simple", 100)),
        ("Explain and then analyze", ("complex", 500)),  # complex wins over medium
        ("DISCUSS the plan", ("complex", 500)),  # case-insensitive
        ("the mathematical foundations", ("complex", 500)),
        ("How does it work?", ("medium", 300)),
        ("re-explained", ("medium", 300)),  # substring match
        ("Name three rivers", ("simple", 100)),  # the YAML's simple_indicators are never read
        ("", ("simple", 100)),
    ],
)
def test_detect_complexity_keywords(query: str, expected: tuple[str, int]) -> None:
    assert detect_complexity_keywords(query, CONFIG.complexity) == expected


def test_complexity_prompt_is_the_legacy_text() -> None:
    assert complexity_prompt("Explain Y") == LEGACY_PROMPT_EXPLAIN_Y
    long_query = "q" * 5000
    assert f"Question: {long_query}\n" in complexity_prompt(long_query)  # the full query, never truncated


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SIMPLE", ("simple", 100)),
        ("MEDIUM", ("medium", 300)),
        ("COMPLEX", ("complex", 500)),
        ("SIMPLE OR COMPLEX", ("simple", 100)),
        ("COMPLEX, NOT MEDIUM", ("medium", 300)),  # MEDIUM is checked before COMPLEX
        ("COMPLEXITY: MEDIUM", ("medium", 300)),
        ("SIMPLEX", ("simple", 100)),
        ("THE ANSWER IS COMPLEX.", ("complex", 500)),
        ("LOW", None),
        ("", None),
        ("simple", None),  # the caller upper-cases the reply first
    ],
)
def test_parse_complexity_reply(reply: str, expected: tuple[str, int] | None) -> None:
    assert parse_complexity_reply(reply, CONFIG.complexity) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("patient records", True),
        ("PATIENT Records", True),
        ("What are common treatments for diabetes?", True),
        ("an impersonal tone", True),  # substring of 'personal'
        ("What is the capital of France?", False),
        ("", False),
    ],
)
def test_is_privacy_sensitive(query: str, expected: bool) -> None:
    assert is_privacy_sensitive(query, CONFIG.privacy_keywords) is expected


def test_privacy_without_keywords_is_never_sensitive() -> None:
    assert is_privacy_sensitive("patient", ()) is False


# Captured from the legacy _generate_rationale for the top model and scored[1:3].
LEGACY_RATIONALES = {
    ("speed", "simple"): (
        "Selected gemma3 (27B, small-tier) for speed strategy. This model offers fastest response "
        "(speed score: 0.90). Query complexity is low, small model sufficient. Alternatives considered: llama3, qwen3."
    ),
    ("quality", "medium"): (
        "Selected qwen3 (235B, large-tier) for quality strategy. This model offers highest quality "
        "(score: 0.90). Alternatives considered: llama3, gemma3."
    ),
    ("quality", "complex"): (
        "Selected qwen3 (235B, large-tier) for quality strategy. This model offers highest quality "
        "(score: 0.90). Query complexity is high, larger model recommended. Alternatives considered: llama3, gemma3."
    ),
    ("balanced", "simple"): (
        "Selected gemma3 (27B, small-tier) for balanced strategy. This model balances quality, speed, and cost "
        "(combined: 0.83). Query complexity is low, small model sufficient. Alternatives considered: llama3, qwen3."
    ),
    ("balanced", "complex"): (
        "Selected llama3 (90B, medium-tier) for balanced strategy. This model balances quality, speed, and cost "
        "(combined: 0.76). Query complexity is high, larger model recommended. Alternatives considered: qwen3, gemma3."
    ),
    ("cost", "complex"): (
        "Selected gemma3 (27B, small-tier) for cost strategy. This model is most cost-effective (cost score: 0.70). "
        "Query complexity is high, larger model recommended. Alternatives considered: llama3, qwen3."
    ),
    ("baseline", "medium"): (
        "Selected gemma3 (27B, small-tier) for baseline strategy. Alternatives considered: llama3, qwen3."
    ),
}


@pytest.mark.parametrize(("strategy", "complexity"), list(LEGACY_RATIONALES))
def test_build_rationale_is_the_legacy_text(strategy: str, complexity: str) -> None:
    top, *alternatives = score_models(CONFIG, complexity, strategy)
    rationale = build_rationale(CONFIG, strategy, complexity, top.model, top, alternatives)
    assert rationale == LEGACY_RATIONALES[strategy, complexity]


def test_build_rationale_without_alternatives_keeps_the_trailing_space() -> None:
    top = score_models(CONFIG, "simple", "speed")[0]
    assert build_rationale(CONFIG, "speed", "simple", top.model, top, []) == (
        "Selected gemma3 (27B, small-tier) for speed strategy. This model offers fastest response "
        "(speed score: 0.90). Query complexity is low, small model sufficient. "
    )


def test_build_rationale_names_only_two_alternatives() -> None:
    top = score_models(CONFIG, "simple", "speed")[0]
    alternatives = [ModelScore(name, "small", 0, 0, 0, 0) for name in ("a", "b", "c")]
    assert build_rationale(CONFIG, "custom", "other", top.model, top, alternatives) == (
        "Selected gemma3 (27B, small-tier) for custom strategy. Alternatives considered: a, b."
    )


# ---------------------------------------------------------------- MultiObjectiveRouter with a fake client

Reply = str | None | BaseException | tuple[str | None, int | None]


class FakeCompletions:
    """Stands in for client.chat.completions: create() records its keyword arguments and plays the next reply.

    A reply is the message content (str or None), an exception to raise, or a (content, total_tokens) pair where
    total_tokens None means a response without usage; plain contents report 6 tokens. A request beyond the
    scripted replies fails the test (pytest.fail is not an Exception, so the router cannot swallow it).
    """

    def __init__(self, replies: Sequence[Reply]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._replies = list(replies)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._replies:
            pytest.fail(f"unexpected completion request: {kwargs}")
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        content, total_tokens = reply if isinstance(reply, tuple) else (reply, 6)
        usage = None if total_tokens is None else SimpleNamespace(total_tokens=total_tokens)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class FakeClient:
    """Stands in for openai.OpenAI; calls lists the keyword arguments of every create() in order."""

    def __init__(self, *replies: Reply) -> None:
        self.completions = FakeCompletions(replies)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.completions.calls


def make_router(client: FakeClient, config: ScorerConfig = CONFIG, **kwargs: Any) -> MultiObjectiveRouter:
    return MultiObjectiveRouter(config, client, now=lambda: FIXED_NOW, **kwargs)


def completion_call(model: str, query: str, max_tokens: int, temperature: float = 0.7) -> dict[str, Any]:
    """The keyword arguments of the legacy execute_query request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def classifier_call(query: str, model: str = "gemma3") -> dict[str, Any]:
    """The keyword arguments of the legacy detect_query_complexity_llm request."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": complexity_prompt(query)}],
        "temperature": 0.0,
        "max_tokens": 10,
    }


def test_the_package_reexports_the_public_api() -> None:
    import mmorch.scoring as scoring

    assert sorted(scoring.__all__) == sorted(
        [
            "ScorerConfig",
            "load_config",
            "ModelScore",
            "score_models",
            "MultiObjectiveRouter",
            "RoutingDecision",
            "QueryResult",
        ]
    )


def test_result_dataclasses_keep_the_legacy_fields() -> None:
    assert [f.name for f in dataclasses.fields(RoutingDecision)] == [
        "selected_model",
        "strategy",
        "query_complexity",
        "is_privacy_sensitive",
        "model_tier",
        "quality_score",
        "speed_score",
        "cost_score",
        "combined_score",
        "alternatives_considered",
        "decision_rationale",
        "timestamp",
    ]
    assert [(f.name, f.default) for f in dataclasses.fields(QueryResult)] == [
        ("query_id", dataclasses.MISSING),
        ("query", dataclasses.MISSING),
        ("routing_decision", dataclasses.MISSING),
        ("response", dataclasses.MISSING),
        ("latency_ms", dataclasses.MISSING),
        ("tokens_used", dataclasses.MISSING),
        ("cost_score", dataclasses.MISSING),
        ("success", dataclasses.MISSING),
        ("error", None),
        ("timestamp", None),
    ]


def test_router_init_logs_the_legacy_lines(mmorch_logs: pytest.LogCaptureFixture) -> None:
    router = make_router(FakeClient())
    records = [r for r in mmorch_logs.records if r.name == "mmorch.scoring.router"]
    assert [(r.levelno, r.getMessage()) for r in records] == [
        (logging.INFO, "> Pick-and-Spin Router initialized"),
        (logging.INFO, "> Available models: ['qwen3', 'gemma3', 'llama3']"),
        (logging.INFO, "> Available strategies: ['quality', 'cost', 'speed', 'balanced', 'baseline']"),
    ]
    assert router.config is CONFIG
    assert router.models is CONFIG.models
    assert router.strategies is CONFIG.strategies
    assert list(router.model_usage_counts.items()) == [("qwen3", 0), ("gemma3", 0), ("llama3", 0)]
    assert router.query_history == []


def test_select_model_baseline_with_a_seeded_rng() -> None:
    router = make_router(FakeClient(), rng=random.Random(0))
    decisions = [router.select_model("x", "baseline") for _ in range(5)]
    reference = random.Random(0)
    assert [d.selected_model for d in decisions] == [reference.choice(MODEL_ORDER) for _ in range(5)]
    assert decisions[0] == RoutingDecision(
        selected_model="gemma3",
        strategy="baseline",
        query_complexity="simple",
        is_privacy_sensitive=False,
        model_tier="small",
        quality_score=0.0,
        speed_score=0.0,
        cost_score=0.0,
        combined_score=0.0,
        alternatives_considered=[],
        decision_rationale="Random selection (baseline strategy)",
        timestamp=FIXED_NOW.isoformat(),
    )


def test_select_model_baseline_uses_the_module_random_choice_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    offered: list[list[str]] = []

    def choice(models: list[str]) -> str:
        offered.append(list(models))
        return "llama3"

    monkeypatch.setattr(random, "choice", choice)
    decision = make_router(FakeClient()).select_model("patient notes", "baseline")
    assert offered == [MODEL_ORDER]
    assert (decision.selected_model, decision.model_tier, decision.is_privacy_sensitive) == ("llama3", "medium", True)


def test_select_model_scores_and_explains() -> None:
    decision = make_router(FakeClient()).select_model("Analyze the trade-offs")
    assert decision == RoutingDecision(
        selected_model="llama3",
        strategy="balanced",
        query_complexity="complex",
        is_privacy_sensitive=False,
        model_tier="medium",
        quality_score=0.8,
        speed_score=0.7,
        cost_score=0.5,
        combined_score=0.76,
        alternatives_considered=[
            {
                "model": "qwen3",
                "tier": "large",
                "quality_score": 0.9,
                "speed_score": 0.6,
                "cost_score": 0.2,
                "combined_score": 0.69,
            },
            {
                "model": "gemma3",
                "tier": "small",
                "quality_score": 0.42,
                "speed_score": 0.9,
                "cost_score": 0.7,
                "combined_score": 0.69,
            },
        ],
        decision_rationale=LEGACY_RATIONALES["balanced", "complex"],
        timestamp=FIXED_NOW.isoformat(),
    )


def test_select_model_lists_three_alternatives_but_explains_two() -> None:
    raw = raw_config()
    for name in ("m4", "m5"):
        raw["models"][name] = {
            "params": "1B",
            "tier": "small",
            "quality_weight": 0.1,
            "speed_weight": 0.1,
            "cost_weight": 0.9,
        }
    decision = make_router(FakeClient(), ScorerConfig.from_mapping(raw)).select_model("What is Z", "speed")
    assert [a["model"] for a in decision.alternatives_considered] == ["llama3", "qwen3", "m4"]
    assert decision.decision_rationale.endswith("Alternatives considered: llama3, qwen3.")


def test_select_model_without_explanation() -> None:
    decision = make_router(FakeClient()).select_model("What is Z", "speed", explain=False)
    assert decision.decision_rationale == "Selected gemma3 using speed strategy"


def test_select_model_rejects_an_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        make_router(FakeClient()).select_model("What is Z", "fastest")


def test_calculate_model_scores_returns_the_legacy_dicts() -> None:
    scores = make_router(FakeClient()).calculate_model_scores("complex", "balanced")
    assert scores == [score.as_dict() for score in score_models(CONFIG, "complex", "balanced")]
    assert [s["model"] for s in scores] == ["llama3", "qwen3", "gemma3"]


@pytest.mark.parametrize(
    ("query", "strategy", "model", "max_tokens"),
    [
        ("What is the capital of France?", "speed", "gemma3", 100),
        ("Explain quantum computing in detail", "quality", "qwen3", 300),
        ("Analyze the trade-offs", "balanced", "llama3", 500),
    ],
)
def test_execute_query_sends_one_completion_with_keyword_complexity(
    query: str, strategy: str, model: str, max_tokens: int
) -> None:
    client = FakeClient("Paris.")
    router = make_router(client)
    result = router.execute_query("q1", query, strategy)
    # Exactly these keyword arguments: the config key as model, no timeout, no stream.
    assert client.calls == [completion_call(model, query, max_tokens)]
    assert result.success is True
    assert (result.query_id, result.query, result.response, result.tokens_used) == ("q1", query, "Paris.", 6)
    assert result.cost_score == round(6 * CONFIG.models[model].cost_weight, 2)
    assert result.error is None
    assert result.timestamp == FIXED_NOW.isoformat()
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms == round(result.latency_ms, 2)
    assert result.routing_decision.selected_model == model
    assert router.model_usage_counts[model] == 1
    assert router.query_history == [result]


def test_execute_query_cost_score_is_rounded() -> None:
    result = make_router(FakeClient(("ok", 6))).execute_query("q", "What is Z", "speed")
    assert result.tokens_used == 6
    assert result.cost_score == 1.8  # 6 * 0.3 is 1.7999999999999998
    result = make_router(FakeClient(("ok", 7))).execute_query("q", "Explain Y", "quality")
    assert result.cost_score == 5.6  # 7 * 0.8 is 5.6000000000000005


def test_execute_query_with_explicit_max_tokens_and_temperature() -> None:
    client = FakeClient("ok")
    make_router(client).execute_query("q", "What is Z", "speed", max_tokens=42, temperature=0.2)
    assert client.calls == [completion_call("gemma3", "What is Z", 42, temperature=0.2)]


def test_execute_query_with_the_llm_detects_complexity_twice() -> None:
    query = "Tell me about rivers"
    client = FakeClient("COMPLEX", "simple", "answer")
    router = make_router(client, ScorerConfig.from_mapping(raw_config(use_llm=True)))
    result = router.execute_query("q", query, "balanced")
    # The legacy double detection: one classifier call for the decision and one for max_tokens.
    assert client.calls == [classifier_call(query), classifier_call(query), completion_call("llama3", query, 100)]
    assert type(client.calls[0]["temperature"]) is float
    assert result.routing_decision.query_complexity == "complex"
    assert result.response == "answer"


def test_execute_query_with_the_llm_and_explicit_max_tokens_classifies_once() -> None:
    client = FakeClient("MEDIUM", "answer")
    router = make_router(client, ScorerConfig.from_mapping(raw_config(use_llm=True, llm_model="tiny")))
    router.execute_query("q", "Explain Y", "quality", max_tokens=7)
    assert client.calls == [classifier_call("Explain Y", model="tiny"), completion_call("qwen3", "Explain Y", 7)]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SIMPLE", ("simple", 100)),
        (" simple, not complex\n", ("simple", 100)),
        ("Medium.", ("medium", 300)),
        ("  COMPLEX  ", ("complex", 500)),
    ],
)
def test_detect_query_complexity_llm_parses_the_reply(
    reply: str, expected: tuple[str, int], mmorch_logs: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(reply)
    router = make_router(client, ScorerConfig.from_mapping(raw_config(use_llm=True)))
    assert router.detect_query_complexity("Explain Y") == expected
    assert client.calls == [classifier_call("Explain Y")]
    assert client.calls[0]["messages"][0]["content"] == LEGACY_PROMPT_EXPLAIN_Y
    assert not [r for r in mmorch_logs.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    ("reply", "warning"),
    [
        ("banana", "  [Warning] LLM returned unparseable complexity: 'BANANA', falling back to keywords"),
        ("  low \n", "  [Warning] LLM returned unparseable complexity: 'LOW', falling back to keywords"),
        (RuntimeError("boom"), "  [Warning] LLM complexity detection failed: boom, falling back to keywords"),
        (
            None,
            "  [Warning] LLM complexity detection failed: 'NoneType' object has no attribute 'strip', "
            "falling back to keywords",
        ),
    ],
)
def test_detect_query_complexity_llm_falls_back_to_keywords(
    reply: Reply, warning: str, mmorch_logs: pytest.LogCaptureFixture
) -> None:
    router = make_router(FakeClient(reply), ScorerConfig.from_mapping(raw_config(use_llm=True)))
    mmorch_logs.clear()
    assert router.detect_query_complexity("Explain Y") == ("medium", 300)
    assert [(r.name, r.levelno, r.getMessage()) for r in mmorch_logs.records] == [
        ("mmorch.scoring.router", logging.WARNING, warning)
    ]


def test_detect_query_complexity_without_the_llm_makes_no_request() -> None:
    client = FakeClient()
    assert make_router(client).detect_query_complexity("Analyze X") == ("complex", 500)
    assert client.calls == []


def test_detect_privacy_sensitive_uses_the_config_keywords() -> None:
    router = make_router(FakeClient())
    assert router.detect_privacy_sensitive("Confidential memo") is True
    assert router.detect_privacy_sensitive("Weather today") is False


def test_a_privacy_sensitive_query_is_stored_redacted() -> None:
    query = "What are common treatments for diabetes?"
    client = FakeClient("Rest.")
    result = make_router(client).execute_query("q", query, "balanced")
    assert result.query == "[REDACTED]"
    assert result.routing_decision.is_privacy_sensitive is True
    assert client.calls == [completion_call("gemma3", query, 100)]  # the request still carries the query


@pytest.mark.parametrize(
    ("length", "expected_length"),
    [(300, 203), (201, 203), (200, 200), (0, 0)],
)
def test_long_responses_are_cut_to_200_characters(length: int, expected_length: int) -> None:
    result = make_router(FakeClient("x" * length)).execute_query("q", "What is Z", "speed")
    assert result.response == ("x" * 200 + "..." if length > 200 else "x" * length)
    assert len(result.response) == expected_length


def test_a_failed_request_gives_a_failed_result() -> None:
    query = "patient question"
    router = make_router(FakeClient(RuntimeError("down")))
    result = router.execute_query("q9", query, "speed")
    assert result == QueryResult(
        query_id="q9",
        query="[REDACTED]",
        routing_decision=result.routing_decision,
        response="",
        latency_ms=result.latency_ms,
        tokens_used=0,
        cost_score=0,
        success=False,
        error="down",
        timestamp=FIXED_NOW.isoformat(),
    )
    assert type(result.cost_score) is int
    assert isinstance(result.latency_ms, float)
    assert router.model_usage_counts == {"qwen3": 0, "gemma3": 0, "llama3": 0}
    assert router.query_history == [result]


def test_a_reply_without_content_counts_the_model_but_fails() -> None:
    # Legacy quirk: the model is counted before len(None) fails, so the failure is still counted.
    router = make_router(FakeClient(None))
    result = router.execute_query("q", "What is Z", "speed")
    assert result.success is False
    assert result.error == "object of type 'NoneType' has no len()"
    assert router.model_usage_counts["gemma3"] == 1


def test_a_reply_without_usage_reports_zero_tokens() -> None:
    result = make_router(FakeClient(("ok", None))).execute_query("q", "What is Z", "speed")
    assert (result.success, result.tokens_used, result.cost_score) == (True, 0, 0.0)


def test_get_usage_statistics() -> None:
    router = make_router(FakeClient("a", "b", "c"))
    assert router.get_usage_statistics() == {
        "total_queries": 0,
        "model_usage": {"qwen3": 0, "gemma3": 0, "llama3": 0},
        "model_distribution": {"qwen3": 0, "gemma3": 0, "llama3": 0},
    }
    for query, strategy in DEMO_QUERIES:
        router.execute_query("q", query, strategy)
    stats = router.get_usage_statistics()
    assert stats["total_queries"] == 3
    assert stats["model_usage"] is router.model_usage_counts
    assert list(stats["model_usage"].items()) == [("qwen3", 1), ("gemma3", 2), ("llama3", 0)]
    assert stats["model_distribution"] == {"qwen3": 1 / 3, "gemma3": 2 / 3, "llama3": 0.0}


# ---------------------------------------------------------------- make_client


def test_make_client_defaults_to_the_yaml_base_and_key_none() -> None:
    pytest.importorskip("openai")
    client = make_client(EndpointSettings(), "https://your-llm-endpoint.example.org/v1")
    try:
        assert client.api_key == "none"
        assert str(client.base_url) == "https://your-llm-endpoint.example.org/v1/"
        assert client.max_retries == 2  # SDK default, untouched
    finally:
        client.close()


def test_make_client_uses_the_endpoint_settings() -> None:
    pytest.importorskip("openai")
    client = make_client(EndpointSettings("http://127.0.0.1:8000/v1", "k"), "https://unused.example.org/v1")
    try:
        assert client.api_key == "k"
        assert str(client.base_url) == "http://127.0.0.1:8000/v1/"
    finally:
        client.close()


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        (EndpointSettings(), {"api_key": "none", "base_url": "https://yaml.example.org/v1"}),
        (EndpointSettings("", ""), {"api_key": "", "base_url": ""}),  # set but empty: both kept
        (EndpointSettings("http://h/v1", None), {"api_key": "none", "base_url": "http://h/v1"}),
        (EndpointSettings(None, "k"), {"api_key": "k", "base_url": "https://yaml.example.org/v1"}),
    ],
)
def test_make_client_passes_the_legacy_values(
    endpoint: EndpointSettings, expected: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A recording stand-in for openai.OpenAI, because openai>=3 itself rejects an empty key.
    openai = pytest.importorskip("openai")
    created: list[dict[str, Any]] = []
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: created.append(kwargs) or "client")
    assert make_client(endpoint, "https://yaml.example.org/v1") == "client"
    assert created == [expected]


def test_make_client_without_openai_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(MissingDependencyError, match=re.escape('pip install -e ".[live]"')) as excinfo:
        make_client(EndpointSettings("http://h/v1", "k"), "https://yaml.example.org/v1")
    assert excinfo.value.package == "openai"


# ---------------------------------------------------------------- the demo behind `mmorch score`

# The legacy main() output on stdout, minus the three initialisation lines (now logged); None marks a latency line.
LEGACY_DEMO_LINES: list[str | None] = [
    "\n" + "=" * 80,
    "PICK-AND-SPIN ROUTING SYSTEM TEST",
    "=" * 80 + "\n",
    "\n[Test 1] Strategy: speed",
    "Query: What is the capital of France?\n",
    "> Model: gemma3",
    "> Complexity: simple",
    "> Privacy: False",
    None,
    "> Tokens: 6",
    "> Rationale: " + LEGACY_RATIONALES["speed", "simple"],
    "\n[Test 2] Strategy: quality",
    "Query: Explain quantum computing in detail\n",
    "> Model: qwen3",
    "> Complexity: medium",
    "> Privacy: False",
    None,
    "> Tokens: 6",
    "> Rationale: " + LEGACY_RATIONALES["quality", "medium"],
    "\n[Test 3] Strategy: balanced",
    "Query: What are common treatments for diabetes?\n",
    "> Model: gemma3",
    "> Complexity: simple",
    "> Privacy: True",
    None,
    "> Tokens: 6",
    "> Rationale: " + LEGACY_RATIONALES["balanced", "simple"],
    "\n" + "=" * 80,
    "Total queries: 3",
    "Model usage: {'qwen3': 1, 'gemma3': 2, 'llama3': 0}",
]

# The three request bodies of the demo, as captured from the legacy script.
LEGACY_DEMO_BODIES = [
    {
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "model": "gemma3",
        "max_tokens": 100,
        "temperature": 0.7,
    },
    {
        "messages": [{"role": "user", "content": "Explain quantum computing in detail"}],
        "model": "qwen3",
        "max_tokens": 300,
        "temperature": 0.7,
    },
    {
        "messages": [{"role": "user", "content": "What are common treatments for diabetes?"}],
        "model": "gemma3",
        "max_tokens": 100,
        "temperature": 0.7,
    },
]


def assert_demo_report(lines: list[str], expected: Sequence[str | None]) -> None:
    assert len(lines) == len(expected)
    for line, want in zip(lines, expected):
        if want is None:
            assert re.fullmatch(r"> Latency: \d+ms", line), line
        else:
            assert line == want


def test_demo_queries_are_the_legacy_ones() -> None:
    assert DEMO_QUERIES == (
        ("What is the capital of France?", "speed"),
        ("Explain quantum computing in detail", "quality"),
        ("What are common treatments for diabetes?", "balanced"),
    )


def test_run_demo_emits_the_legacy_report() -> None:
    client = FakeClient("Paris.", "Paris.", "Paris.")
    lines: list[str] = []
    run_demo(make_router(client), lines.append)
    assert_demo_report(lines, LEGACY_DEMO_LINES)
    assert client.calls == [
        completion_call("gemma3", "What is the capital of France?", 100),
        completion_call("qwen3", "Explain quantum computing in detail", 300),
        completion_call("gemma3", "What are common treatments for diabetes?", 100),
    ]


def test_run_demo_reports_failures() -> None:
    lines: list[str] = []
    run_demo(make_router(FakeClient(*(RuntimeError("down") for _ in DEMO_QUERIES))), lines.append)
    assert lines == [
        *LEGACY_DEMO_LINES[:3],
        "\n[Test 1] Strategy: speed",
        "Query: What is the capital of France?\n",
        "X Error: down",
        "\n[Test 2] Strategy: quality",
        "Query: Explain quantum computing in detail\n",
        "X Error: down",
        "\n[Test 3] Strategy: balanced",
        "Query: What are common treatments for diabetes?\n",
        "X Error: down",
        "\n" + "=" * 80,
        "Total queries: 0",
        "Model usage: {'qwen3': 0, 'gemma3': 0, 'llama3': 0}",
    ]


def test_run_demo_writes_its_report_only_through_emit(mmorch_logs: pytest.LogCaptureFixture) -> None:
    router = make_router(FakeClient("a", "b", "c"))
    mmorch_logs.clear()
    run_demo(router, lambda line: None)
    assert mmorch_logs.records == []


# ---------------------------------------------------------------- exact request bodies from the real SDK


class SDKStub:
    """A real OpenAI client whose HTTP transport is an in-process stub: no socket, and every request is recorded.

    Classifier requests (max_tokens 10) are answered with classifier_reply, all others with reply.
    """

    def __init__(self) -> None:
        openai = pytest.importorskip("openai")
        httpx = pytest.importorskip("httpx")
        self.requests: list[Any] = []
        self.reply = "Paris."
        self.classifier_reply = "MEDIUM"
        self._httpx = httpx
        self.client = openai.OpenAI(
            api_key="test-key",
            base_url="http://llm.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: Any) -> Any:
        self.requests.append(request)
        body = json.loads(request.content)
        content = self.classifier_reply if body.get("max_tokens") == 10 else self.reply
        return self._httpx.Response(
            200,
            json={
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    def bodies(self) -> list[Any]:
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture
def sdk_stub() -> Iterator[SDKStub]:
    stub = SDKStub()
    try:
        yield stub
    finally:
        stub.client.close()


def test_demo_request_bodies_through_the_sdk(sdk_stub: SDKStub) -> None:
    lines: list[str] = []
    run_demo(MultiObjectiveRouter(CONFIG, sdk_stub.client), lines.append)
    assert_demo_report(lines, LEGACY_DEMO_LINES)
    assert [(r.method, r.url.path) for r in sdk_stub.requests] == [("POST", "/v1/chat/completions")] * 3
    assert {r.headers["authorization"] for r in sdk_stub.requests} == {"Bearer test-key"}
    assert sdk_stub.bodies() == LEGACY_DEMO_BODIES
    assert all(type(body["temperature"]) is float for body in sdk_stub.bodies())


def test_classifier_request_body_through_the_sdk(sdk_stub: SDKStub) -> None:
    router = MultiObjectiveRouter(ScorerConfig.from_mapping(raw_config(use_llm=True)), sdk_stub.client)
    sdk_stub.classifier_reply = "COMPLEX"
    assert router.detect_query_complexity("Explain Y") == ("complex", 500)
    [body] = sdk_stub.bodies()
    assert body == {
        "messages": [{"role": "user", "content": LEGACY_PROMPT_EXPLAIN_Y}],
        "model": "gemma3",
        "max_tokens": 10,
        "temperature": 0.0,
    }
    assert type(body["temperature"]) is float  # sent as 0.0, never 0


def test_llm_demo_sends_classifier_classifier_completion_through_the_sdk(sdk_stub: SDKStub) -> None:
    router = MultiObjectiveRouter(ScorerConfig.from_mapping(raw_config(use_llm=True)), sdk_stub.client)
    router.execute_query("test_1", "What is the capital of France?", "speed")
    assert [(body["model"], body["max_tokens"]) for body in sdk_stub.bodies()] == [
        ("gemma3", 10),
        ("gemma3", 10),
        ("gemma3", 300),
    ]
