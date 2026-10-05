"""The two tier classifiers behind the routing traces. Pure apart from the client passed in.

- Keyword rules match lowercase substrings: any HIGH phrase gives HIGH, otherwise any LOW phrase gives LOW,
  otherwise MEDIUM.
- The LLM-prompt classifier makes one non-streaming call. Anything other than an exact LOW / MEDIUM / HIGH
  reply, or any exception, gives MEDIUM.

mmorch.paper.analysis re-runs classify_keyword over the 31,019 prompts, so the reproduction's 100% agreement
check exercises this exact code.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from openai import OpenAI


class Tier(StrEnum):
    """A complexity tier. str(), f-strings, csv and json all render the bare value, e.g. 'LOW'."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Both lists verbatim and in the legacy order. Matching is by plain substring, so 'pick' also matches 'pickles'
# and 'define' matches 'undefined'; the recorded tiers were produced that way.
COMPLEXITY_KEYWORDS: Final[Mapping[Tier, tuple[str, ...]]] = MappingProxyType(
    {
        Tier.LOW: (
            "what is",
            "define",
            "who is",
            "when did",
            "where is",
            "true or false",
            "yes or no",
            "which of the following",
            "select",
            "choose",
            "pick",
            "identify",
        ),
        Tier.HIGH: (
            "explain why",
            "analyze",
            "compare and contrast",
            "evaluate",
            "prove",
            "derive",
            "justify",
            "critique",
            "design",
            "develop",
            "synthesize",
            "create",
            "formulate",
        ),
    }
)

CLASSIFIER_MAX_TOKENS: Final = 10
CLASSIFIER_TIMEOUT_S: Final = 10
CLASSIFIER_PROMPT_CHARS: Final = 500

# The labels the LLM classifier accepts, compared after strip() and upper().
_LABELS: Final = ("LOW", "MEDIUM", "HIGH")


def classify_keyword(query: str) -> Tier:
    """Return HIGH if the lowercased query contains a HIGH phrase, else LOW if it has a LOW phrase, else MEDIUM."""
    q = query.lower()
    if any(k in q for k in COMPLEXITY_KEYWORDS[Tier.HIGH]):
        return Tier.HIGH
    if any(k in q for k in COMPLEXITY_KEYWORDS[Tier.LOW]):
        return Tier.LOW
    return Tier.MEDIUM


def classifier_prompt(query: str) -> str:
    """Return the legacy classification prompt, which quotes the first 500 characters of query."""
    return (
        "Classify the complexity of this question as LOW, MEDIUM, or HIGH.\n\n"
        "LOW: Simple factual questions, definitions, true/false, multiple choice\n"
        "MEDIUM: Questions requiring some reasoning or multi-step thinking\n"
        "HIGH: Complex analysis, proofs, design problems, advanced reasoning\n\n"
        f"Question: {query[:CLASSIFIER_PROMPT_CHARS]}\n\n"
        "Respond with ONLY one word: LOW, MEDIUM, or HIGH"
    )


def parse_tier(reply: str) -> Tier:
    """Return the tier a reply names exactly (after strip and upper), or MEDIUM for anything else."""
    label = reply.strip().upper()
    return Tier(label) if label in _LABELS else Tier.MEDIUM


def classify_llm(client: OpenAI, query: str, *, model: str) -> Tier:
    """Ask model for the tier of query in one non-streaming call; MEDIUM on any exception or unexpected reply.

    The request carries exactly model, one user message, max_tokens 10, temperature 0.0 and a 10 s timeout.
    temperature stays the float 0.0, which the SDK sends as 0.0 rather than 0.
    """
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": classifier_prompt(query)}],
            max_tokens=CLASSIFIER_MAX_TOKENS,
            temperature=0.0,
            timeout=CLASSIFIER_TIMEOUT_S,
        )
        content = r.choices[0].message.content
        if content is None:
            # The legacy code called .strip() on it here, which raised inside this try and so gave MEDIUM.
            return Tier.MEDIUM
        return parse_tier(content)
    except Exception:
        return Tier.MEDIUM
