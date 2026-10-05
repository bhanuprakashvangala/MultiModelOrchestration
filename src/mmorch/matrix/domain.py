"""Domain classification of prompts for the prototype.

The DistilBERT path is tried first. It is used only when torch and transformers import, which is checked once;
otherwise the legacy warning is logged once and only keywords are used. The classifier instance is cached per
process, a failed initialisation returns None and is retried on the next call, and a confidence below 0.4 gives
GENERAL. Without a fine-tuned model in models/domain_classifier_distilbert (relative to the current directory),
the classifier is DistilBERT with a fresh, untrained 4-label head, as before.

The keyword fallback counts substring hits per domain, breaks ties by dict order (BIOLOGY, CHEMISTRY,
MATERIALS) and gives GENERAL below a confidence of 0.3.

torch and transformers (extra 'matrix-ml') are imported only inside the functions that use them.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from mmorch.matrix.endpoints import DomainType

log = logging.getLogger(__name__)

# The three legacy lists, verbatim and in order.
DOMAIN_KEYWORDS: Final[Mapping[DomainType, tuple[str, ...]]] = MappingProxyType(
    {
        DomainType.BIOLOGY: (
            "protein",
            "gene",
            "dna",
            "rna",
            "cell",
            "enzyme",
            "antibody",
            "virus",
            "bacteria",
            "genome",
            "mutation",
            "evolution",
            "disease",
            "drug",
            "medicine",
            "biological",
            "organism",
            "tissue",
            "molecular",
            "pathway",
            "receptor",
        ),
        DomainType.CHEMISTRY: (
            "molecule",
            "reaction",
            "compound",
            "element",
            "chemical",
            "synthesis",
            "catalyst",
            "acid",
            "base",
            "ph",
            "bond",
            "organic",
            "inorganic",
            "polymer",
            "solution",
            "concentration",
            "molarity",
            "oxidation",
            "reduction",
            "electrochemistry",
            "thermodynamics",
        ),
        DomainType.MATERIALS: (
            "material",
            "crystal",
            "lattice",
            "semiconductor",
            "metal",
            "alloy",
            "composite",
            "nanomaterial",
            "graphene",
            "polymer",
            "ceramic",
            "glass",
            "mechanical",
            "thermal",
            "electrical",
            "optical",
            "magnetic",
            "properties",
            "structure",
            "defect",
            "phase",
        ),
    }
)

DEFAULT_MODEL_DIR: Final = "models/domain_classifier_distilbert"
BASE_MODEL: Final = "distilbert-base-uncased"
LABELS: Final = ("biology", "chemistry", "materials", "general")

# Below these confidences a prompt counts as GENERAL.
TRANSFORMER_MIN_CONFIDENCE: Final = 0.4
KEYWORD_MIN_CONFIDENCE: Final = 0.3

# The transformer's labels as domains; any other label counts as GENERAL.
_DOMAIN_FOR_LABEL: Final[Mapping[str, DomainType]] = MappingProxyType(
    {
        "biology": DomainType.BIOLOGY,
        "chemistry": DomainType.CHEMISTRY,
        "materials": DomainType.MATERIALS,
        "general": DomainType.GENERAL,
    }
)


def classify_keywords(prompt: str) -> tuple[DomainType, float]:
    """Classify by keyword hits per domain; GENERAL when the best domain's share of hits is below 0.3.

    Keywords match as lowercase substrings, so 'ph' also matches 'phase'. The confidence is the best domain's
    hits divided by all hits (at least 1); equal hits go to the first domain in BIOLOGY, CHEMISTRY, MATERIALS.
    """
    prompt_lower = prompt.lower()
    domain_scores: dict[DomainType, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in prompt_lower)
        domain_scores[domain] = score

    # max() keeps the first of equal scores.
    best_domain = max(domain_scores, key=domain_scores.__getitem__)
    confidence = domain_scores[best_domain] / max(1, sum(domain_scores.values()))

    if confidence < KEYWORD_MIN_CONFIDENCE:
        return DomainType.GENERAL, confidence

    log.info("Keyword-based classified prompt as %s with confidence %.2f", best_domain.value, confidence)
    return best_domain, confidence


@functools.cache
def transformer_available() -> bool:
    """Whether torch and transformers import; checked once, logging the legacy warning when they do not."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        log.warning("Transformer domain classifier not available, using keyword-based classification")
        return False
    return True


class TransformerDomainClassifier:
    """The DistilBERT classifier: the fine-tuned model in model_dir if present, else a fresh 4-label head."""

    device: Any
    tokenizer: Any
    model: Any

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR) -> None:
        """Load the tokenizer and model (fine-tuned from model_dir, or BASE_MODEL with 4 labels) on GPU or CPU."""
        import torch
        from transformers import AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The tokenizer always comes from the base model, also for a fine-tuned model_dir.
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        self.model = self._load_or_create_model(BASE_MODEL, model_dir)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_or_create_model(model_name: str, model_dir: str) -> Any:
        """Load the fine-tuned model from model_dir, or create model_name with a new 4-label head."""
        from transformers import AutoModelForSequenceClassification

        if os.path.exists(model_dir):
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                log.info("Loaded fine-tuned domain classifier")
                return model
            except Exception as e:
                log.warning("Could not load fine-tuned model: %s", e)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            problem_type="single_label_classification",
        )

        model.config.id2label = dict(enumerate(LABELS))
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        return model

    def classify(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Return (label, confidence, probability per label) for text, truncated to 512 tokens."""
        import torch

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)

            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_idx].item()

            domain_mapping = dict(enumerate(LABELS))
            prob_dict = {label: probabilities[0, index].item() for index, label in enumerate(LABELS)}

            return domain_mapping[predicted_idx], confidence, prob_dict


# The process-wide classifier, created on first use by get_transformer_classifier.
_classifier: TransformerDomainClassifier | None = None


def get_transformer_classifier() -> TransformerDomainClassifier | None:
    """Return the process-wide classifier, creating it on first use; None (not cached) if that fails."""
    global _classifier

    if _classifier is None:
        try:
            _classifier = TransformerDomainClassifier()
            log.info("Domain classifier initialized successfully")
        except Exception as e:
            log.error("Failed to initialize domain classifier: %s", e)
            return None

    return _classifier


def classify_domain(prompt: str) -> tuple[DomainType, float]:
    """Classify with DistilBERT when available, falling back to keywords on any failure."""
    if transformer_available():
        try:
            classifier = get_transformer_classifier()
            if classifier is not None:
                domain_str, confidence, probabilities = classifier.classify(prompt)
                domain = _DOMAIN_FOR_LABEL.get(domain_str, DomainType.GENERAL)

                log.info("Transformer classified prompt as %s with confidence %.2f", domain.value, confidence)
                log.debug("Probabilities: %s", probabilities)

                if confidence < TRANSFORMER_MIN_CONFIDENCE:
                    return DomainType.GENERAL, confidence

                return domain, confidence

        except Exception as e:
            log.warning("Transformer classification failed, falling back to keywords: %s", e)

    return classify_keywords(prompt)
