"""Tests for mmorch.routing.classifier: the keyword rules, the LLM-prompt classifier and the Tier enum.

The keyword lists, the prompt text and the request arguments are literal copies of the v1.0.0 runner
(src/routing/smart_routing.py at tag v1.0.0), so any drift from the recorded method fails here.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
from types import SimpleNamespace
from typing import Any

import pytest

from mmorch.routing.classifier import (
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_PROMPT_CHARS,
    CLASSIFIER_TIMEOUT_S,
    COMPLEXITY_KEYWORDS,
    Tier,
    classifier_prompt,
    classify_keyword,
    classify_llm,
    parse_tier,
)

LEGACY_LOW = [
    "what is", "define", "who is", "when did", "where is",
    "true or false", "yes or no", "which of the following",
    "select", "choose", "pick", "identify",
]  # fmt: skip
LEGACY_HIGH = [
    "explain why", "analyze", "compare and contrast", "evaluate",
    "prove", "derive", "justify", "critique", "design",
    "develop", "synthesize", "create", "formulate",
]  # fmt: skip

# A sentence that contains none of the phrases, to embed one phrase in.
NEUTRAL = "Tell me about {} in a sentence."


# ---------------------------------------------------------------- keyword rules


def test_keyword_lists_are_the_legacy_lists_in_order() -> None:
    assert list(COMPLEXITY_KEYWORDS) == [Tier.LOW, Tier.HIGH]
    for tier, legacy in ((Tier.LOW, LEGACY_LOW), (Tier.HIGH, LEGACY_HIGH)):
        phrases = COMPLEXITY_KEYWORDS[tier]
        assert isinstance(phrases, tuple)
        assert len(phrases) == len(legacy)
        for phrase, expected in zip(phrases, legacy):
            assert phrase == expected
    # The legacy dict was keyed by plain strings; StrEnum keys still answer to them.
    assert COMPLEXITY_KEYWORDS["HIGH"] == tuple(LEGACY_HIGH)


def test_keyword_lists_are_read_only() -> None:
    with pytest.raises(TypeError):
        COMPLEXITY_KEYWORDS[Tier.LOW] = ()


@pytest.mark.parametrize(
    ("query", "tier"),
    [
        ("What is 2+2? Prove it.", Tier.HIGH),  # a HIGH phrase wins over a LOW one
        ("What is the capital of France?", Tier.LOW),
        ("Finish the story", Tier.MEDIUM),
        ("WHAT IS LOVE", Tier.LOW),  # matching is on the lowercased query
        ("Pickles are tasty", Tier.LOW),  # plain substrings: 'pick'
        ("The redesigned API", Tier.HIGH),  # 'design'
        ("He was undefined", Tier.LOW),  # 'define'
        ("Recreate the file", Tier.HIGH),  # 'create'
        ("", Tier.MEDIUM),
        ("what  is", Tier.MEDIUM),  # no whitespace normalisation
        ("What\nis it?", Tier.MEDIUM),
        # lower(), not casefold(): the 'fi' ligature and the long s stay as they are, so these match no phrase.
        ("De\ufb01ne the term", Tier.MEDIUM),  # casefold() would give 'define'
        ("\u017felect one", Tier.MEDIUM),  # casefold() would give 'select'
    ],
)
def test_classify_keyword_precedence_and_substring_matching(query: str, tier: Tier) -> None:
    assert classify_keyword(query) is tier


@pytest.mark.parametrize("phrase", LEGACY_LOW)
def test_each_low_phrase_alone_gives_low(phrase: str) -> None:
    assert classify_keyword(NEUTRAL.format(phrase)) is Tier.LOW
    assert classify_keyword(NEUTRAL.format(phrase.upper())) is Tier.LOW


@pytest.mark.parametrize("phrase", LEGACY_HIGH)
def test_each_high_phrase_alone_gives_high(phrase: str) -> None:
    assert classify_keyword(NEUTRAL.format(phrase)) is Tier.HIGH
    assert classify_keyword(NEUTRAL.format(phrase.title())) is Tier.HIGH


def test_every_high_phrase_wins_over_every_low_phrase_in_either_order() -> None:
    for low, high in itertools.product(LEGACY_LOW, LEGACY_HIGH):
        assert classify_keyword(f"{low} ... {high}") is Tier.HIGH
        assert classify_keyword(f"{high} ... {low}") is Tier.HIGH


def test_neutral_sentence_has_no_phrase() -> None:
    assert classify_keyword(NEUTRAL.format("cats")) is Tier.MEDIUM


# ---------------------------------------------------------------- LLM-prompt classifier: prompt and reply


def test_classifier_limits() -> None:
    assert (CLASSIFIER_MAX_TOKENS, CLASSIFIER_TIMEOUT_S, CLASSIFIER_PROMPT_CHARS) == (10, 10, 500)


def test_classifier_prompt_is_the_legacy_text_with_the_first_500_characters() -> None:
    question = "".join(chr(ord("a") + i % 26) for i in range(600))
    assert classifier_prompt(question) == (
        "Classify the complexity of this question as LOW, MEDIUM, or HIGH.\n\n"
        "LOW: Simple factual questions, definitions, true/false, multiple choice\n"
        "MEDIUM: Questions requiring some reasoning or multi-step thinking\n"
        "HIGH: Complex analysis, proofs, design problems, advanced reasoning\n\n"
        "Question: " + question[:500] + "\n\n"
        "Respond with ONLY one word: LOW, MEDIUM, or HIGH"
    )
    assert question[:500] in classifier_prompt(question)
    assert question[:501] not in classifier_prompt(question)


def test_classifier_prompt_keeps_a_short_question_whole() -> None:
    question = 'Prove that "naïve" proofs fail.\nRésumé'
    assert f"Question: {question}\n\nRespond" in classifier_prompt(question)


@pytest.mark.parametrize(
    ("reply", "tier"),
    [
        (" low\n", Tier.LOW),
        ("MEDIUM", Tier.MEDIUM),
        ("\tHigh  ", Tier.HIGH),
        ("HIGH.", Tier.MEDIUM),
        ("", Tier.MEDIUM),
        ("banana", Tier.MEDIUM),
        ("LOW MEDIUM", Tier.MEDIUM),
        ("Tier.HIGH", Tier.MEDIUM),
    ],
)
def test_parse_tier(reply: str, tier: Tier) -> None:
    assert parse_tier(reply) is tier


# ---------------------------------------------------------------- LLM-prompt classifier: the call


class FakeCompletions:
    """client.chat.completions: records the keyword arguments of every create() call."""

    def __init__(self, content: object = "LOW", error: Exception | None = None, choices: list[Any] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.content = content
        self.error = error
        self.choices = choices

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.choices is not None:
            return SimpleNamespace(choices=self.choices)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def fake_client(completions: FakeCompletions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_classify_llm_sends_exactly_the_legacy_request() -> None:
    completions = FakeCompletions(content="HIGH")
    question = "q" * 600

    assert classify_llm(fake_client(completions), question, model="small-x") is Tier.HIGH

    assert completions.calls == [
        {
            "model": "small-x",
            "messages": [{"role": "user", "content": classifier_prompt(question)}],
            "max_tokens": 10,
            "temperature": 0.0,
            "timeout": 10,
        }
    ]
    call = completions.calls[0]
    # The SDK sends 0.0 for a float and 0 for an int, and the timeout as the x-stainless-read-timeout header.
    assert type(call["temperature"]) is float
    assert type(call["max_tokens"]) is int
    assert type(call["timeout"]) is int


@pytest.mark.parametrize(("content", "tier"), [(" low\n", Tier.LOW), ("Medium", Tier.MEDIUM), ("HIGH", Tier.HIGH)])
def test_classify_llm_parses_the_reply(content: str, tier: Tier) -> None:
    assert classify_llm(fake_client(FakeCompletions(content=content)), "q", model="m") is tier


@pytest.mark.parametrize(
    "completions",
    [
        pytest.param(FakeCompletions(error=RuntimeError("connection refused")), id="raising-client"),
        pytest.param(FakeCompletions(error=TimeoutError("timed out")), id="timeout"),
        pytest.param(FakeCompletions(content=None), id="none-content"),
        pytest.param(FakeCompletions(content="I think HIGH"), id="unparseable-label"),
        pytest.param(FakeCompletions(content=42), id="non-string-content"),
        pytest.param(FakeCompletions(choices=[]), id="no-choices"),
    ],
)
def test_classify_llm_falls_back_to_medium(completions: FakeCompletions) -> None:
    assert classify_llm(fake_client(completions), "Prove it", model="m") is Tier.MEDIUM
    assert len(completions.calls) == 1


def test_classify_llm_does_not_catch_base_exceptions() -> None:
    # As before, only Exception is caught: a KeyboardInterrupt still stops the run.
    with pytest.raises(KeyboardInterrupt):
        classify_llm(fake_client(FakeCompletions(error=KeyboardInterrupt())), "q", model="m")


# ---------------------------------------------------------------- Tier


def test_tier_members_in_order() -> None:
    assert [t.value for t in Tier] == ["LOW", "MEDIUM", "HIGH"]
    assert Tier("HIGH") is Tier.HIGH


@pytest.mark.parametrize("tier", list(Tier))
def test_tier_renders_as_its_bare_value(tier: Tier) -> None:
    value = tier.value
    assert str(tier) == value
    assert f"{tier}" == value
    assert format(tier, ">8s") == f"{value:>8s}"
    assert json.dumps(tier) == json.dumps(value)
    assert json.dumps({"complexity": tier}) == f'{{"complexity": "{value}"}}'
    buffer = io.StringIO()
    csv.writer(buffer).writerow([tier])
    assert buffer.getvalue() == f"{value}\r\n"
    assert tier == value


def test_tier_compares_equal_to_the_recorded_strings() -> None:
    assert Tier.LOW == "LOW"
    assert Tier.LOW != "low"
    assert {"LOW": 1}[Tier.LOW] == 1
