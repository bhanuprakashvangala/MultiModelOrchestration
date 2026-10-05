"""The Eq. 2 multi-objective scorer with operator profiles, behind `mmorch score`.

- config: the scorer YAML (models, operator profiles, complexity rules, privacy keywords) as frozen dataclasses
- scorer: the pure scoring and ranking, the scorer's own keyword rules, privacy detection and the rationale text
- router: MultiObjectiveRouter, which detects complexity, selects and calls a model, and runs the demo

This package re-exports the main names. Importing it loads neither yaml nor openai: both are imported inside the
functions that need them (mmorch.scoring.config.load_config and mmorch.scoring.router.make_client).
"""

from .config import ScorerConfig, load_config
from .router import MultiObjectiveRouter, QueryResult, RoutingDecision
from .scorer import ModelScore, score_models

__all__ = [
    "ModelScore",
    "MultiObjectiveRouter",
    "QueryResult",
    "RoutingDecision",
    "ScorerConfig",
    "load_config",
    "score_models",
]
