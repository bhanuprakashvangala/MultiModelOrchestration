"""Pick's tier routing, as used for the routing traces behind Figs. 4-11.

Re-exports the public API and imports nothing optional: openai is imported only inside
mmorch.routing.runner.make_client.
"""

from .classifier import COMPLEXITY_KEYWORDS, Tier, classify_keyword, classify_llm
from .runner import CallResult, RouteRecord, TierModels, call_model, route_to_model, run_routing

__all__ = [
    "COMPLEXITY_KEYWORDS",
    "CallResult",
    "RouteRecord",
    "Tier",
    "TierModels",
    "call_model",
    "classify_keyword",
    "classify_llm",
    "route_to_model",
    "run_routing",
]
