"""Tests for mmorch.matrix: the registry, payloads, endpoint choice, helm argv, domain classification,
BackendManager and Orchestrator.

helm never runs: every manager gets a recording runner (FakeHelm), and helm.run itself is tested against a
monkeypatched subprocess.run. Sleeps are recorded instead of waited, and torch and transformers are faked. Nothing
here opens a socket: the aiohttp requests of check_health and _generate_from_endpoint are tested against a stub
server in tests/integration/test_matrix_api.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import logging
import os
import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from mmorch.errors import MissingDependencyError
from mmorch.matrix import domain, helm
from mmorch.matrix.backend_manager import BackendManager
from mmorch.matrix.endpoints import (
    BACKEND_PRIORITIES,
    HEALTHY_PRIORITY,
    MODEL_CATALOG,
    MODEL_FOR_DOMAIN,
    BackendStatus,
    BackendType,
    DomainType,
    ModelEndpoint,
    build_payload,
    choose_endpoint,
    default_endpoints,
)
from mmorch.matrix.orchestrator import (
    DEPLOYMENT_ORDER,
    MATRIX_VIEW_BACKENDS,
    MATRIX_VIEW_MODELS,
    DeploymentConfig,
    Orchestrator,
)
from mmorch.settings import DEFAULT_CHART_DIR

REGISTRY_MODELS = ["biogpt", "chemberta", "matscibert"]
REGISTRY_BACKENDS = ["vllm", "tgi", "tensorrt"]
DEFAULT_METRICS = {"latency_ms": 0, "throughput": 0, "cost_per_token": 0, "success_rate": 100}


# ---------------------------------------------------------------- fakes and fixtures


class FakeHelm:
    """A helm runner that records each argv and answers with a fixed return code, or raises."""

    def __init__(self, returncode: int = 0, stderr: str = "", error: Exception | None = None) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.error = error
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(list(cmd), self.returncode, stdout="", stderr=self.stderr)


class RecordingSleep:
    """An async sleep that returns at once, records each delay and then calls on_sleep(delay) if set."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.on_sleep: Callable[[float], None] | None = None

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self.on_sleep is not None:
            self.on_sleep(delay)


@pytest.fixture
def chart_dir(tmp_path: Path) -> str:
    """An umbrella chart directory, as the str the managers take, with a subchart for each of the nine releases."""
    root = tmp_path / "umbrella"
    for model, backend in DEPLOYMENT_ORDER:
        (root / "charts" / f"{model}-{backend}").mkdir(parents=True)
    return str(root)


@pytest.fixture
def matrix_log(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog at DEBUG for mmorch.matrix, also after a CLI test has stopped the 'mmorch' logger propagating."""
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.DEBUG, logger="mmorch.matrix")
    return caplog


def messages(caplog: pytest.LogCaptureFixture, level: int | None = None) -> list[str]:
    """The captured mmorch.matrix messages, optionally only those of one level."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("mmorch.matrix") and (level is None or record.levelno == level)
    ]


def endpoint_of(manager: BackendManager, model: str, backend: str) -> ModelEndpoint:
    return next(ep for ep in manager.endpoints[model] if ep.backend == backend)


# ---------------------------------------------------------------- endpoints: enums, registry, URLs


def test_enums_keep_the_legacy_values_and_stay_plain_enums() -> None:
    assert [member.value for member in BackendType] == ["vllm", "tgi", "tensorrt"]
    assert [member.value for member in BackendStatus] == [
        "healthy",
        "unhealthy",
        "unknown",
        "deployed",
        "not_deployed",
        "deploying",
    ]
    assert [member.value for member in DomainType] == ["biology", "chemistry", "materials", "general"]
    # Rendered with .value, as before; they are not str subclasses.
    for enum in (BackendType, BackendStatus, DomainType):
        assert not issubclass(enum, str)
    assert DomainType.BIOLOGY.value == "biology"


def test_default_endpoints_follow_the_legacy_registry() -> None:
    endpoints = default_endpoints("ns")

    assert list(endpoints) == REGISTRY_MODELS
    flat = [endpoint for model_endpoints in endpoints.values() for endpoint in model_endpoints]
    assert [(ep.model, ep.backend, ep.port) for ep in flat] == [
        ("biogpt", "vllm", 8000),
        ("biogpt", "tgi", 8001),
        ("biogpt", "tensorrt", 8002),
        ("chemberta", "vllm", 8003),
        ("chemberta", "tgi", 8004),
        ("chemberta", "tensorrt", 8005),
        ("matscibert", "vllm", 8006),
        ("matscibert", "tgi", 8007),
        ("matscibert", "tensorrt", 8008),
    ]
    domains = {"biogpt": DomainType.BIOLOGY, "chemberta": DomainType.CHEMISTRY, "matscibert": DomainType.MATERIALS}
    for ep in flat:
        assert ep.host == f"{ep.model}-{ep.backend}-service"
        assert ep.helm_release == f"{ep.model}-{ep.backend}"
        assert ep.namespace == "ns"
        assert ep.domain is domains[ep.model]
        assert ep.status is BackendStatus.NOT_DEPLOYED
        assert ep.deployment_status is BackendStatus.NOT_DEPLOYED
        assert ep.performance_metrics == DEFAULT_METRICS
        assert [type(value) for value in ep.performance_metrics.values()] == [int, int, int, int]

    biogpt_vllm, biogpt_tgi, biogpt_tensorrt = endpoints["biogpt"]
    assert biogpt_vllm.health_check_url == "http://biogpt-vllm-service:8000/health"
    assert biogpt_vllm.generate_url == "http://biogpt-vllm-service:8000/v1/completions"
    assert biogpt_tgi.health_check_url == "http://biogpt-tgi-service:8001/health"
    assert biogpt_tgi.generate_url == "http://biogpt-tgi-service:8001/generate"
    assert biogpt_tensorrt.health_check_url == "http://biogpt-tensorrt-service:8002/v2/health/ready"
    assert biogpt_tensorrt.generate_url == "http://biogpt-tensorrt-service:8002/v2/models/biogpt/infer"
    assert (
        endpoints["matscibert"][2].generate_url == "http://matscibert-tensorrt-service:8008/v2/models/matscibert/infer"
    )


def test_default_endpoints_are_new_objects_each_time() -> None:
    first, second = default_endpoints("a"), default_endpoints("b")
    assert first["biogpt"][0] is not second["biogpt"][0]
    assert first["biogpt"][0].performance_metrics is not second["biogpt"][0].performance_metrics
    assert second["biogpt"][0].namespace == "b"


@pytest.mark.parametrize(
    ("backend", "health_check_url", "generate_url"),
    [
        ("vllm", "http://host:9/health", "http://host:9/v1/completions"),
        ("tgi", "http://host:9/health", "http://host:9/generate"),
        ("tensorrt", "http://host:9/v2/health/ready", "http://host:9/v2/models/m/infer"),
        ("onnx", "", ""),
    ],
)
def test_model_endpoint_sets_urls_and_resets_metrics(backend: str, health_check_url: str, generate_url: str) -> None:
    endpoint = ModelEndpoint("m", backend, "host", 9, DomainType.GENERAL, performance_metrics={"latency_ms": 5.5})

    assert endpoint.health_check_url == health_check_url
    assert endpoint.generate_url == generate_url
    assert endpoint.performance_metrics == DEFAULT_METRICS
    assert endpoint.address == "host:9"
    assert endpoint.namespace == "default"
    assert endpoint.helm_release == ""


def test_model_endpoint_repr_shows_only_the_legacy_fields() -> None:
    text = repr(ModelEndpoint("m", "vllm", "host", 9, DomainType.GENERAL))
    assert text.startswith("ModelEndpoint(model='m', backend='vllm', host='host', port=9, domain=")
    assert text.endswith(f"performance_metrics={DEFAULT_METRICS!r})")
    assert "url" not in text


def test_domain_model_and_priority_tables() -> None:
    assert dict(MODEL_FOR_DOMAIN) == {
        DomainType.BIOLOGY: "biogpt",
        DomainType.CHEMISTRY: "chemberta",
        DomainType.MATERIALS: "matscibert",
        DomainType.GENERAL: "biogpt",
    }
    assert {name: dict(priorities) for name, priorities in BACKEND_PRIORITIES.items()} == {
        "speed": {"tensorrt": 1, "vllm": 2, "tgi": 3},
        "cost": {"tgi": 1, "vllm": 2, "tensorrt": 3},
        "quality": {"vllm": 1, "tensorrt": 2, "tgi": 3},
        "balanced": {"vllm": 1, "tgi": 2, "tensorrt": 3},
    }
    assert list(BACKEND_PRIORITIES) == ["speed", "cost", "quality", "balanced"]
    assert HEALTHY_PRIORITY == {"vllm": 1, "tgi": 2, "tensorrt": 3}


def test_model_catalog_is_verbatim() -> None:
    assert MODEL_CATALOG == (
        {
            "name": "biogpt",
            "description": "Microsoft BioGPT for biomedical text generation",
            "domain": "biology",
            "backends": ["vllm", "tgi", "tensorrt"],
        },
        {
            "name": "chemberta",
            "description": "ChemBERTa for chemistry and molecular understanding",
            "domain": "chemistry",
            "backends": ["vllm", "tgi", "tensorrt"],
        },
        {
            "name": "matscibert",
            "description": "MatSciBERT for materials science applications",
            "domain": "materials",
            "backends": ["vllm", "tgi", "tensorrt"],
        },
    )
    assert [list(entry) for entry in MODEL_CATALOG] == [["name", "description", "domain", "backends"]] * 3


# ---------------------------------------------------------------- endpoints: payloads


@pytest.mark.parametrize(
    ("backend", "body"),
    [
        ("vllm", '{"model": "biogpt", "prompt": "Hi", "max_tokens": 150, "temperature": 0.7}'),
        ("tgi", '{"inputs": "Hi", "parameters": {"max_new_tokens": 150, "temperature": 0.7}}'),
        (
            "tensorrt",
            '{"inputs": [{"name": "input_text", "shape": [1], "datatype": "BYTES", "data": ["Hi"]}], '
            '"parameters": {"max_tokens": 150, "temperature": 0.7}}',
        ),
    ],
)
def test_build_payload_shapes_and_key_order(backend: str, body: str) -> None:
    endpoint = ModelEndpoint("biogpt", backend, "host", 1, DomainType.BIOLOGY)
    # json.dumps keeps the key order, which aiohttp's json= sends as is.
    assert json.dumps(build_payload(endpoint, "Hi")) == body


def test_build_payload_takes_the_limits() -> None:
    vllm = ModelEndpoint("chemberta", "vllm", "host", 1, DomainType.CHEMISTRY)
    tgi = ModelEndpoint("chemberta", "tgi", "host", 1, DomainType.CHEMISTRY)
    tensorrt = ModelEndpoint("chemberta", "tensorrt", "host", 1, DomainType.CHEMISTRY)

    assert build_payload(vllm, "x", max_tokens=5, temperature=1.2) == {
        "model": "chemberta",
        "prompt": "x",
        "max_tokens": 5,
        "temperature": 1.2,
    }
    assert build_payload(tgi, "x", max_tokens=5, temperature=1.2)["parameters"] == {
        "max_new_tokens": 5,
        "temperature": 1.2,
    }
    assert build_payload(tensorrt, "x", max_tokens=5, temperature=1.2)["parameters"] == {
        "max_tokens": 5,
        "temperature": 1.2,
    }


def test_build_payload_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="onnx"):
        build_payload(ModelEndpoint("m", "onnx", "host", 1, DomainType.GENERAL), "Hi")


# ---------------------------------------------------------------- endpoints: choice


@pytest.mark.parametrize(
    ("optimize_for", "backend"),
    [
        ("speed", "tensorrt"),
        ("cost", "tgi"),
        ("quality", "vllm"),
        ("balanced", "vllm"),
        ("fastest", "vllm"),  # unknown: balanced
        ("", "vllm"),
    ],
)
def test_choose_endpoint_with_default_metrics(optimize_for: str, backend: str) -> None:
    for model, endpoints in default_endpoints("ns").items():
        chosen = choose_endpoint(endpoints, optimize_for)
        assert chosen is not None
        assert (chosen.model, chosen.backend) == (model, backend)


def test_choose_endpoint_adds_latency_and_failure_rate() -> None:
    vllm, tgi, tensorrt = default_endpoints("ns")["biogpt"]

    vllm.performance_metrics["success_rate"] = 0  # 1 + 1.0
    tgi.performance_metrics["latency_ms"] = 500  # 2 + 0.5
    assert choose_endpoint([vllm, tgi, tensorrt], "balanced") is vllm

    vllm.performance_metrics["latency_ms"] = 600  # 1 + 0.6 + 1.0
    assert choose_endpoint([vllm, tgi, tensorrt], "balanced") is tgi


def test_choose_endpoint_defaults_missing_metrics_to_the_worst_case() -> None:
    vllm, tgi, tensorrt = default_endpoints("ns")["biogpt"]
    tensorrt.performance_metrics.clear()  # 1 + 1000 / 1000 + (100 - 0) / 100 = 3.0
    assert choose_endpoint([vllm, tgi, tensorrt], "speed") is vllm  # 2.0


def test_choose_endpoint_keeps_the_first_of_equal_scores() -> None:
    vllm, tgi, tensorrt = default_endpoints("ns")["biogpt"]
    vllm.performance_metrics["latency_ms"] = 1000  # 1 + 1.0 == 2 + 0.0
    assert choose_endpoint([vllm, tgi, tensorrt], "balanced") is vllm
    assert choose_endpoint([tgi, vllm, tensorrt], "balanced") is tgi

    first = ModelEndpoint("m", "onnx", "a", 1, DomainType.GENERAL)
    second = ModelEndpoint("m", "triton", "b", 2, DomainType.GENERAL)
    assert choose_endpoint([first, second], "speed") is first  # both 99


def test_choose_endpoint_of_nothing_is_none() -> None:
    assert choose_endpoint([], "balanced") is None


# ---------------------------------------------------------------- helm argv and the subprocess seam


def test_install_and_uninstall_argv() -> None:
    assert helm.install_cmd("biogpt-vllm", "chart/charts/biogpt-vllm", "ns") == [
        "helm",
        "upgrade",
        "--install",
        "biogpt-vllm",
        "chart/charts/biogpt-vllm",
        "--namespace",
        "ns",
        "--create-namespace",
        "--wait",
        "--timeout",
        "5m",
    ]
    assert helm.uninstall_cmd("biogpt-vllm", "ns") == ["helm", "uninstall", "biogpt-vllm", "--namespace", "ns"]


def test_umbrella_argv() -> None:
    chart = "./deploy/helm/pick-and-spin-umbrella"
    assert helm.umbrella_enable_cmd(chart, "ns", "matscibert-tgi") == [
        "helm",
        "upgrade",
        "--install",
        "multi-llm",
        "./deploy/helm/pick-and-spin-umbrella",
        "--namespace",
        "ns",
        "--create-namespace",
        "--set",
        "matscibert_tgi.enabled=true",
        "--wait",
        "--timeout",
        "5m",
    ]
    assert helm.umbrella_disable_cmd(chart, "ns", "matscibert-tgi") == [
        "helm",
        "upgrade",
        "multi-llm",
        "./deploy/helm/pick-and-spin-umbrella",
        "--namespace",
        "ns",
        "--set",
        "matscibert_tgi.enabled=false",
        "--wait",
        "--timeout",
        "2m",
    ]
    assert helm.UMBRELLA_RELEASE == "multi-llm"
    assert helm.values_key("chemberta-tensorrt") == "chemberta_tensorrt"


def test_subchart_path_joins_the_raw_chart_dir() -> None:
    path = helm.subchart_path(DEFAULT_CHART_DIR, "biogpt-vllm")
    assert path == os.path.join("./deploy/helm/pick-and-spin-umbrella", "charts", "biogpt-vllm")
    assert path.startswith("./deploy")  # pathlib would drop the './'


def test_run_is_one_synchronous_subprocess_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 3, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = helm.run(("helm", "uninstall", "biogpt-vllm"))

    assert calls == [((["helm", "uninstall", "biogpt-vllm"],), {"capture_output": True, "text": True})]
    assert (result.returncode, result.stdout, result.stderr) == (3, "out", "err")


# ---------------------------------------------------------------- domain: keywords


def test_domain_keywords_are_the_legacy_lists() -> None:
    assert list(domain.DOMAIN_KEYWORDS) == [DomainType.BIOLOGY, DomainType.CHEMISTRY, DomainType.MATERIALS]
    assert domain.DOMAIN_KEYWORDS[DomainType.BIOLOGY] == (
        "protein", "gene", "dna", "rna", "cell", "enzyme", "antibody", "virus",
        "bacteria", "genome", "mutation", "evolution", "disease", "drug", "medicine",
        "biological", "organism", "tissue", "molecular", "pathway", "receptor",
    )  # fmt: skip
    assert domain.DOMAIN_KEYWORDS[DomainType.CHEMISTRY] == (
        "molecule", "reaction", "compound", "element", "chemical", "synthesis",
        "catalyst", "acid", "base", "ph", "bond", "organic", "inorganic",
        "polymer", "solution", "concentration", "molarity", "oxidation", "reduction",
        "electrochemistry", "thermodynamics",
    )  # fmt: skip
    assert domain.DOMAIN_KEYWORDS[DomainType.MATERIALS] == (
        "material", "crystal", "lattice", "semiconductor", "metal", "alloy",
        "composite", "nanomaterial", "graphene", "polymer", "ceramic", "glass",
        "mechanical", "thermal", "electrical", "optical", "magnetic", "properties",
        "structure", "defect", "phase",
    )  # fmt: skip


def test_domain_constants() -> None:
    assert domain.DEFAULT_MODEL_DIR == "models/domain_classifier_distilbert"
    assert domain.BASE_MODEL == "distilbert-base-uncased"
    assert domain.LABELS == ("biology", "chemistry", "materials", "general")


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("protein gene", (DomainType.BIOLOGY, 1.0)),
        ("polymer", (DomainType.CHEMISTRY, 0.5)),  # in two lists: the tie goes by dict order
        ("phase", (DomainType.CHEMISTRY, 0.5)),  # 'ph' is a chemistry keyword
        ("PROTEIN in a CRYSTAL lattice", (DomainType.MATERIALS, 2 / 3)),
        ("hello", (DomainType.GENERAL, 0.0)),
        ("", (DomainType.GENERAL, 0.0)),
    ],
)
def test_classify_keywords(prompt: str, expected: tuple[DomainType, float]) -> None:
    assert domain.classify_keywords(prompt) == expected


def test_classify_keywords_logs_only_a_domain(matrix_log: pytest.LogCaptureFixture) -> None:
    domain.classify_keywords("protein gene")
    domain.classify_keywords("hello")
    assert messages(matrix_log) == ["Keyword-based classified prompt as biology with confidence 1.00"]
    assert matrix_log.records[0].levelno == logging.INFO


# ---------------------------------------------------------------- domain: transformer path (faked)


class FakeClassifier:
    """Stands in for TransformerDomainClassifier: returns a fixed result, or raises."""

    def __init__(self, result: tuple[str, float, dict[str, float]] | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.texts: list[str] = []

    def classify(self, text: str) -> tuple[str, float, dict[str, float]]:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@pytest.fixture
def transformer(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeClassifier | None], None]:
    """install(classifier): pretend torch and transformers import and serve this classifier (or None)."""

    def install(classifier: FakeClassifier | None) -> None:
        monkeypatch.setattr(domain, "transformer_available", lambda: True)
        monkeypatch.setattr(domain, "get_transformer_classifier", lambda: classifier)

    return install


PROBABILITIES = {"biology": 0.05, "chemistry": 0.03, "materials": 0.9, "general": 0.02}


def test_classify_domain_uses_a_confident_transformer(
    transformer: Callable[[FakeClassifier | None], None], matrix_log: pytest.LogCaptureFixture
) -> None:
    classifier = FakeClassifier(("materials", 0.9, PROBABILITIES))
    transformer(classifier)

    assert domain.classify_domain("protein gene") == (DomainType.MATERIALS, 0.9)
    assert classifier.texts == ["protein gene"]
    assert messages(matrix_log, logging.INFO) == ["Transformer classified prompt as materials with confidence 0.90"]
    assert messages(matrix_log, logging.DEBUG) == [f"Probabilities: {PROBABILITIES}"]


@pytest.mark.parametrize(
    ("label", "confidence", "expected"),
    [
        ("chemistry", 0.35, (DomainType.GENERAL, 0.35)),  # below 0.4
        ("chemistry", 0.4, (DomainType.CHEMISTRY, 0.4)),
        ("physics", 0.8, (DomainType.GENERAL, 0.8)),  # an unknown label
        ("biology", 1.0, (DomainType.BIOLOGY, 1.0)),
    ],
)
def test_classify_domain_maps_transformer_labels(
    transformer: Callable[[FakeClassifier | None], None],
    label: str,
    confidence: float,
    expected: tuple[DomainType, float],
) -> None:
    transformer(FakeClassifier((label, confidence, PROBABILITIES)))
    assert domain.classify_domain("hello") == expected


def test_classify_domain_falls_back_without_a_classifier(transformer: Callable[[FakeClassifier | None], None]) -> None:
    transformer(None)
    assert domain.classify_domain("protein gene") == (DomainType.BIOLOGY, 1.0)


def test_classify_domain_falls_back_when_the_transformer_fails(
    transformer: Callable[[FakeClassifier | None], None], matrix_log: pytest.LogCaptureFixture
) -> None:
    transformer(FakeClassifier(error=RuntimeError("CUDA out of memory")))

    assert domain.classify_domain("polymer") == (DomainType.CHEMISTRY, 0.5)
    assert messages(matrix_log, logging.WARNING) == [
        "Transformer classification failed, falling back to keywords: CUDA out of memory"
    ]


def test_classify_domain_without_transformers_uses_keywords_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected() -> None:
        pytest.fail("the classifier must not be created without torch and transformers")

    monkeypatch.setattr(domain, "transformer_available", lambda: False)
    monkeypatch.setattr(domain, "get_transformer_classifier", unexpected)
    assert domain.classify_domain("crystal lattice") == (DomainType.MATERIALS, 1.0)


@pytest.fixture
def fresh_probe() -> Iterator[None]:
    """transformer_available caches its answer per process; start and end each test without that cache."""
    domain.transformer_available.cache_clear()
    yield
    domain.transformer_available.cache_clear()


@pytest.mark.usefixtures("fresh_probe")
@pytest.mark.parametrize("missing", ["torch", "transformers"])
def test_transformer_available_warns_once_when_a_package_is_missing(
    monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture, missing: str
) -> None:
    for name in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None if name == missing else types.ModuleType(name))

    assert domain.transformer_available() is False
    assert domain.transformer_available() is False
    assert messages(matrix_log, logging.WARNING) == [
        "Transformer domain classifier not available, using keyword-based classification"
    ]


@pytest.mark.usefixtures("fresh_probe")
def test_transformer_available_with_both_packages(
    monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    assert domain.transformer_available() is True
    assert messages(matrix_log) == []


def test_get_transformer_classifier_creates_one_instance(
    monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    created: list[object] = []

    class Created:
        def __init__(self) -> None:
            created.append(self)

    monkeypatch.setattr(domain, "_classifier", None)
    monkeypatch.setattr(domain, "TransformerDomainClassifier", Created)

    first = domain.get_transformer_classifier()
    assert domain.get_transformer_classifier() is first
    assert created == [first]
    assert messages(matrix_log) == ["Domain classifier initialized successfully"]


def test_get_transformer_classifier_retries_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    attempts: list[int] = []

    class Failing:
        def __init__(self) -> None:
            attempts.append(1)
            raise OSError("no weights")

    monkeypatch.setattr(domain, "_classifier", None)
    monkeypatch.setattr(domain, "TransformerDomainClassifier", Failing)

    assert domain.get_transformer_classifier() is None
    assert domain.get_transformer_classifier() is None
    assert len(attempts) == 2
    assert messages(matrix_log, logging.ERROR) == ["Failed to initialize domain classifier: no weights"] * 2


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Row:
    """A 1 x n tensor, indexed as [0, i] like the softmax output."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: tuple[int, int]) -> _Scalar:
        batch, column = index
        assert batch == 0
        return _Scalar(self.values[column])


@pytest.fixture
def fake_ml(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Fake torch and transformers modules that record every call; the model's softmax output is `row`."""
    state = types.SimpleNamespace(calls=[], row=[0.1, 0.1, 0.7, 0.1], fine_tuned_error=None)

    class Inputs(dict[str, Any]):
        def to(self, device: str) -> Inputs:
            state.calls.append(("inputs.to", device))
            return self

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> Inputs:
            state.calls.append(("tokenize", text, kwargs))
            return Inputs(input_ids=[[1, 2]])

    class Model:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace()

        def to(self, device: str) -> None:
            state.calls.append(("model.to", device))

        def eval(self) -> None:
            state.calls.append(("model.eval",))

        def __call__(self, **inputs: Any) -> types.SimpleNamespace:
            state.calls.append(("model", inputs))
            return types.SimpleNamespace(logits=_Row(state.row))

    def model_from_pretrained(name: str, **kwargs: Any) -> Model:
        state.calls.append(("model.from_pretrained", name, kwargs))
        if not kwargs and state.fine_tuned_error is not None:
            raise state.fine_tuned_error
        return Model()

    torch = types.ModuleType("torch")
    torch.device = lambda name: f"device:{name}"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = contextlib.nullcontext
    torch.softmax = lambda logits, dim: (state.calls.append(("softmax", dim)), logits)[1]
    torch.argmax = lambda probabilities, dim: _Scalar(
        max(range(len(probabilities.values)), key=probabilities.values.__getitem__)
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda name: (state.calls.append(("tokenizer.from_pretrained", name)), Tokenizer())[1]
    )
    transformers.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=model_from_pretrained)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


def test_transformer_classifier_creates_a_four_label_head(fake_ml: types.SimpleNamespace, tmp_path: Path) -> None:
    classifier = domain.TransformerDomainClassifier(str(tmp_path / "no-fine-tuned-model"))

    assert fake_ml.calls == [
        ("tokenizer.from_pretrained", "distilbert-base-uncased"),
        (
            "model.from_pretrained",
            "distilbert-base-uncased",
            {"num_labels": 4, "problem_type": "single_label_classification"},
        ),
        ("model.to", "device:cpu"),
        ("model.eval",),
    ]
    assert classifier.model.config.id2label == {0: "biology", 1: "chemistry", 2: "materials", 3: "general"}
    assert classifier.model.config.label2id == {"biology": 0, "chemistry": 1, "materials": 2, "general": 3}

    fake_ml.calls.clear()
    assert classifier.classify("crystal growth") == (
        "materials",
        0.7,
        {"biology": 0.1, "chemistry": 0.1, "materials": 0.7, "general": 0.1},
    )
    assert fake_ml.calls == [
        (
            "tokenize",
            "crystal growth",
            {"padding": True, "truncation": True, "max_length": 512, "return_tensors": "pt"},
        ),
        ("inputs.to", "device:cpu"),
        ("model", {"input_ids": [[1, 2]]}),
        ("softmax", -1),
    ]


def test_transformer_classifier_prefers_the_fine_tuned_model(
    fake_ml: types.SimpleNamespace, tmp_path: Path, matrix_log: pytest.LogCaptureFixture
) -> None:
    model_dir = tmp_path / "fine-tuned"
    model_dir.mkdir()

    domain.TransformerDomainClassifier(str(model_dir))

    assert ("model.from_pretrained", str(model_dir), {}) in fake_ml.calls
    assert ("tokenizer.from_pretrained", "distilbert-base-uncased") in fake_ml.calls
    assert messages(matrix_log) == ["Loaded fine-tuned domain classifier"]


def test_transformer_classifier_falls_back_to_the_base_model(
    fake_ml: types.SimpleNamespace, tmp_path: Path, matrix_log: pytest.LogCaptureFixture
) -> None:
    model_dir = tmp_path / "fine-tuned"
    model_dir.mkdir()
    fake_ml.fine_tuned_error = OSError("broken checkpoint")

    domain.TransformerDomainClassifier(str(model_dir))

    loads = [call[1:] for call in fake_ml.calls if call[0] == "model.from_pretrained"]
    assert loads == [
        (str(model_dir), {}),
        ("distilbert-base-uncased", {"num_labels": 4, "problem_type": "single_label_classification"}),
    ]
    assert messages(matrix_log, logging.WARNING) == ["Could not load fine-tuned model: broken checkpoint"]


# ---------------------------------------------------------------- BackendManager: setup and helm


def test_backend_manager_setup() -> None:
    manager = BackendManager("ns")

    assert manager.namespace == "ns"
    assert list(manager.endpoints) == REGISTRY_MODELS
    assert all(ep.namespace == "ns" for eps in manager.endpoints.values() for ep in eps)
    assert manager.health_check_interval == 30
    assert manager.last_health_check == {}
    assert BackendManager().namespace == "default"


def test_backend_manager_classifies_with_the_injected_function() -> None:
    prompts: list[str] = []

    def classify(prompt: str) -> tuple[DomainType, float]:
        prompts.append(prompt)
        return DomainType.MATERIALS, 0.75

    assert BackendManager(classify=classify).classify_domain("x") == (DomainType.MATERIALS, 0.75)
    assert prompts == ["x"]


def test_backend_manager_classifies_with_classify_domain_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(domain, "transformer_available", lambda: False)
    assert BackendManager().classify_domain("protein gene") == (DomainType.BIOLOGY, 1.0)


def test_deploy_endpoint_installs_the_subchart(chart_dir: str, matrix_log: pytest.LogCaptureFixture) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    endpoint = endpoint_of(manager, "chemberta", "tgi")

    assert asyncio.run(manager.deploy_endpoint(endpoint)) is True

    assert run_helm.calls == [
        [
            "helm",
            "upgrade",
            "--install",
            "chemberta-tgi",
            os.path.join(chart_dir, "charts", "chemberta-tgi"),
            "--namespace",
            "ns",
            "--create-namespace",
            "--wait",
            "--timeout",
            "5m",
        ]
    ]
    assert endpoint.deployment_status is BackendStatus.DEPLOYED
    assert endpoint.status is BackendStatus.NOT_DEPLOYED
    assert sleep.delays == [10]
    assert messages(matrix_log) == ["Deploying chemberta-tgi in namespace ns", "Successfully deployed chemberta-tgi"]


def test_deploy_endpoint_failure(chart_dir: str, matrix_log: pytest.LogCaptureFixture) -> None:
    run_helm, sleep = FakeHelm(returncode=1, stderr="Error: quota exceeded"), RecordingSleep()
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    endpoint = endpoint_of(manager, "biogpt", "vllm")

    assert asyncio.run(manager.deploy_endpoint(endpoint)) is False

    assert len(run_helm.calls) == 1
    assert endpoint.deployment_status is BackendStatus.NOT_DEPLOYED
    assert sleep.delays == []
    assert messages(matrix_log, logging.ERROR) == ["Failed to deploy biogpt-vllm: Error: quota exceeded"]


def test_deploy_endpoint_error(chart_dir: str, matrix_log: pytest.LogCaptureFixture) -> None:
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=FakeHelm(error=FileNotFoundError("helm")))
    endpoint = endpoint_of(manager, "biogpt", "vllm")

    assert asyncio.run(manager.deploy_endpoint(endpoint)) is False

    assert endpoint.deployment_status is BackendStatus.NOT_DEPLOYED
    assert messages(matrix_log, logging.ERROR) == ["Error deploying biogpt-vllm: helm"]


def test_deploy_endpoint_without_the_subchart_stays_deploying(
    tmp_path: Path, matrix_log: pytest.LogCaptureFixture
) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    chart_dir = str(tmp_path / "empty-umbrella")
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    endpoint = endpoint_of(manager, "biogpt", "vllm")

    assert asyncio.run(manager.deploy_endpoint(endpoint)) is False
    assert messages(matrix_log, logging.ERROR) == [
        f"Helm chart not found at {os.path.join(chart_dir, 'charts', 'biogpt-vllm')}"
    ]
    # The legacy quirk, kept: the endpoint is left DEPLOYING, so the next call succeeds without running helm.
    assert endpoint.deployment_status is BackendStatus.DEPLOYING
    assert asyncio.run(manager.deploy_endpoint(endpoint)) is True
    assert run_helm.calls == []
    assert sleep.delays == []


def test_deploy_endpoint_looks_for_the_default_chart_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)  # the relative default chart directory does not exist here
    run_helm = FakeHelm()
    manager = BackendManager(run_helm=run_helm, sleep=RecordingSleep())

    assert asyncio.run(manager.deploy_endpoint(endpoint_of(manager, "matscibert", "tensorrt"))) is False
    assert run_helm.calls == []
    expected = os.path.join("./deploy/helm/pick-and-spin-umbrella", "charts", "matscibert-tensorrt")
    assert messages(matrix_log, logging.ERROR) == [f"Helm chart not found at {expected}"]


@pytest.mark.parametrize("state", [BackendStatus.DEPLOYED, BackendStatus.DEPLOYING])
def test_deploy_endpoint_skips_a_deployed_endpoint(chart_dir: str, state: BackendStatus) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    endpoint = endpoint_of(manager, "biogpt", "tgi")
    endpoint.deployment_status = state

    assert asyncio.run(manager.deploy_endpoint(endpoint)) is True
    assert endpoint.deployment_status is state
    assert run_helm.calls == []
    assert sleep.delays == []


@pytest.mark.parametrize("state", [BackendStatus.DEPLOYED, BackendStatus.DEPLOYING])
def test_undeploy_endpoint_uninstalls_the_release(state: BackendStatus, matrix_log: pytest.LogCaptureFixture) -> None:
    run_helm = FakeHelm()
    manager = BackendManager("ns", run_helm=run_helm)
    endpoint = endpoint_of(manager, "matscibert", "tensorrt")
    endpoint.deployment_status = state
    endpoint.status = BackendStatus.HEALTHY

    assert asyncio.run(manager.undeploy_endpoint(endpoint)) is True

    assert run_helm.calls == [["helm", "uninstall", "matscibert-tensorrt", "--namespace", "ns"]]
    assert endpoint.deployment_status is BackendStatus.NOT_DEPLOYED
    assert endpoint.status is BackendStatus.NOT_DEPLOYED
    assert messages(matrix_log) == ["Successfully undeployed matscibert-tensorrt"]


def test_undeploy_endpoint_failure_keeps_the_state(matrix_log: pytest.LogCaptureFixture) -> None:
    manager = BackendManager("ns", run_helm=FakeHelm(returncode=1, stderr="release not found"))
    endpoint = endpoint_of(manager, "biogpt", "vllm")
    endpoint.deployment_status = BackendStatus.DEPLOYED
    endpoint.status = BackendStatus.HEALTHY

    assert asyncio.run(manager.undeploy_endpoint(endpoint)) is False
    assert (endpoint.deployment_status, endpoint.status) == (BackendStatus.DEPLOYED, BackendStatus.HEALTHY)

    manager = BackendManager("ns", run_helm=FakeHelm(error=OSError("no helm")))
    endpoint = endpoint_of(manager, "biogpt", "vllm")
    endpoint.deployment_status = BackendStatus.DEPLOYED
    assert asyncio.run(manager.undeploy_endpoint(endpoint)) is False

    assert messages(matrix_log, logging.ERROR) == [
        "Failed to undeploy biogpt-vllm: release not found",
        "Error undeploying biogpt-vllm: no helm",
    ]


def test_undeploy_endpoint_of_an_undeployed_endpoint_does_nothing() -> None:
    run_helm = FakeHelm()
    manager = BackendManager("ns", run_helm=run_helm)
    assert asyncio.run(manager.undeploy_endpoint(endpoint_of(manager, "biogpt", "vllm"))) is True
    assert run_helm.calls == []


# ---------------------------------------------------------------- BackendManager: health and selection


@pytest.mark.parametrize("state", [BackendStatus.NOT_DEPLOYED, BackendStatus.DEPLOYING, BackendStatus.HEALTHY])
def test_check_health_of_an_undeployed_endpoint_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, state: BackendStatus
) -> None:
    monkeypatch.setitem(sys.modules, "aiohttp", None)  # any request would need aiohttp
    manager = BackendManager()
    endpoint = endpoint_of(manager, "biogpt", "vllm")
    endpoint.deployment_status = state
    endpoint.status = BackendStatus.HEALTHY

    assert asyncio.run(manager.check_health(endpoint)) is BackendStatus.NOT_DEPLOYED
    assert endpoint.status is BackendStatus.NOT_DEPLOYED


def test_requests_need_aiohttp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "aiohttp", None)
    manager = BackendManager()
    endpoint = endpoint_of(manager, "biogpt", "vllm")
    endpoint.deployment_status = BackendStatus.DEPLOYED

    with pytest.raises(MissingDependencyError) as health_error:
        asyncio.run(manager.check_health(endpoint))
    with pytest.raises(MissingDependencyError) as generate_error:
        asyncio.run(manager._generate_from_endpoint(endpoint, "Hi"))

    for error in (health_error.value, generate_error.value):
        assert (error.package, error.extra) == ("aiohttp", "matrix")
        assert str(error) == "aiohttp is required for this command: pip install 'mmorch[matrix]'"
    # A missing package is not a failed generation.
    assert endpoint.performance_metrics["success_rate"] == 100


def test_check_all_health_checks_every_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = BackendManager()
    seen: list[str] = []

    async def check_health(endpoint: ModelEndpoint) -> BackendStatus:
        seen.append(endpoint.helm_release)
        if endpoint.backend == "tgi":
            raise RuntimeError("boom")  # gathered with return_exceptions=True
        return BackendStatus.NOT_DEPLOYED

    monkeypatch.setattr(manager, "check_health", check_health)

    asyncio.run(manager.check_all_health())

    assert seen == [f"{model}-{backend}" for model in REGISTRY_MODELS for backend in REGISTRY_BACKENDS]


def test_healthy_and_best_endpoints() -> None:
    manager = BackendManager()
    vllm, tgi, tensorrt = manager.endpoints["biogpt"]

    assert manager.get_best_endpoint("biogpt") is None
    assert manager.get_healthy_endpoints("gpt") == []
    assert manager.get_best_endpoint("gpt") is None

    tensorrt.status = BackendStatus.HEALTHY
    tgi.status = BackendStatus.HEALTHY
    assert manager.get_healthy_endpoints("biogpt") == [tgi, tensorrt]
    assert manager.get_best_endpoint("biogpt") is tgi  # vLLM > TGI > TensorRT among the healthy ones
    assert manager.get_best_endpoint("biogpt", "tensorrt") is tensorrt
    assert manager.get_best_endpoint("biogpt", "vllm") is tgi  # the preferred backend is not healthy

    vllm.status = BackendStatus.HEALTHY
    assert manager.get_best_endpoint("biogpt") is vllm
    assert manager.get_best_endpoint("biogpt", "") is vllm


def healthy_check(log: list[str]) -> Callable[[ModelEndpoint], Any]:
    """A check_health replacement that records the endpoint and marks it HEALTHY."""

    async def check_health(endpoint: ModelEndpoint) -> BackendStatus:
        log.append(endpoint.helm_release)
        endpoint.status = BackendStatus.HEALTHY
        return BackendStatus.HEALTHY

    return check_health


def test_select_optimal_endpoint_deploys_and_checks_its_choice(
    chart_dir: str, monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    prompts: list[str] = []

    def classify(prompt: str) -> tuple[DomainType, float]:
        prompts.append(prompt)
        return DomainType.CHEMISTRY, 0.8

    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep, classify=classify)
    checked: list[str] = []
    monkeypatch.setattr(manager, "check_health", healthy_check(checked))

    endpoint = asyncio.run(manager.select_optimal_endpoint("acid base titration", "speed"))

    assert endpoint is endpoint_of(manager, "chemberta", "tensorrt")
    assert prompts == ["acid base titration"]
    assert run_helm.calls == [
        helm.install_cmd("chemberta-tensorrt", os.path.join(chart_dir, "charts", "chemberta-tensorrt"), "ns")
    ]
    assert sleep.delays == [10, 5]
    assert checked == ["chemberta-tensorrt"]
    assert messages(matrix_log, logging.INFO)[:2] == [
        "Classified prompt as chemistry with confidence 0.80",
        "Deploying optimal endpoint: chemberta-tensorrt",
    ]


def test_select_optimal_endpoint_reuses_a_deployed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager(run_helm=run_helm, sleep=sleep, classify=lambda prompt: (DomainType.GENERAL, 0.0))
    deployed = endpoint_of(manager, "biogpt", "vllm")  # GENERAL goes to biogpt
    deployed.deployment_status = BackendStatus.DEPLOYED
    checked: list[str] = []
    monkeypatch.setattr(manager, "check_health", healthy_check(checked))

    assert asyncio.run(manager.select_optimal_endpoint("hello")) is deployed
    assert (run_helm.calls, sleep.delays, checked) == ([], [], [])


def test_select_optimal_endpoint_after_a_failed_deploy(chart_dir: str, matrix_log: pytest.LogCaptureFixture) -> None:
    sleep = RecordingSleep()
    manager = BackendManager(
        "ns",
        chart_dir=chart_dir,
        run_helm=FakeHelm(returncode=1),
        sleep=sleep,
        classify=lambda prompt: (DomainType.MATERIALS, 1.0),
    )

    assert asyncio.run(manager.select_optimal_endpoint("crystal", "cost")) is None
    assert sleep.delays == []
    assert messages(matrix_log, logging.ERROR)[-1] == "Failed to deploy matscibert-tgi"


def test_select_optimal_endpoint_with_a_deploying_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # The DEPLOYING quirk: the deploy reports success without helm, then the health check runs anyway.
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager(run_helm=run_helm, sleep=sleep, classify=lambda prompt: (DomainType.BIOLOGY, 1.0))
    stuck = endpoint_of(manager, "biogpt", "vllm")
    stuck.deployment_status = BackendStatus.DEPLOYING

    assert asyncio.run(manager.select_optimal_endpoint("protein")) is stuck
    assert run_helm.calls == []
    assert sleep.delays == [5]
    assert stuck.status is BackendStatus.NOT_DEPLOYED  # check_health: not DEPLOYED


# ---------------------------------------------------------------- BackendManager: generation and status


def recording_generate(calls: list[tuple[str, str, dict[str, Any]]]) -> Callable[..., Any]:
    """A _generate_from_endpoint replacement that records (release, prompt, kwargs) and returns a result dict."""

    async def generate(endpoint: ModelEndpoint, prompt: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((endpoint.helm_release, prompt, kwargs))
        return {
            "model": endpoint.model,
            "backend": endpoint.backend,
            "response": {"text": "ok"},
            "endpoint": endpoint.address,
        }

    return generate


def test_generate_completion_with_routing_adds_times_and_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = BackendManager()
    endpoint = endpoint_of(manager, "matscibert", "vllm")
    selections: list[tuple[str, str]] = []

    async def select(prompt: str, optimize_for: str = "balanced") -> ModelEndpoint:
        selections.append((prompt, optimize_for))
        return endpoint

    generated: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(manager, "select_optimal_endpoint", select)
    monkeypatch.setattr(manager, "_generate_from_endpoint", recording_generate(generated))

    result = asyncio.run(manager.generate_completion_with_routing("alloy", "cost", max_tokens=9))

    assert selections == [("alloy", "cost")]
    assert generated == [("matscibert-vllm", "alloy", {"max_tokens": 9})]
    assert list(result) == [
        "model",
        "backend",
        "response",
        "endpoint",
        "selection_time_ms",
        "total_time_ms",
        "domain",
    ]
    assert result["domain"] == "materials"
    assert 0 <= result["selection_time_ms"] <= result["total_time_ms"]
    assert endpoint.performance_metrics["latency_ms"] == result["total_time_ms"]


def test_generate_completion_with_routing_without_an_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = BackendManager()

    async def select(prompt: str, optimize_for: str = "balanced") -> None:
        return None

    monkeypatch.setattr(manager, "select_optimal_endpoint", select)

    with pytest.raises(RuntimeError, match=r"^No suitable endpoint could be deployed$"):
        asyncio.run(manager.generate_completion_with_routing("x"))


def test_generate_completion_uses_the_best_healthy_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    run_helm = FakeHelm()
    manager = BackendManager(run_helm=run_helm)
    endpoint_of(manager, "biogpt", "tensorrt").status = BackendStatus.HEALTHY
    endpoint_of(manager, "biogpt", "tgi").status = BackendStatus.HEALTHY
    generated: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(manager, "_generate_from_endpoint", recording_generate(generated))

    result = asyncio.run(manager.generate_completion("biogpt", "Hi", backend=None, max_tokens=5, temperature=0.2))

    assert generated == [("biogpt-tgi", "Hi", {"max_tokens": 5, "temperature": 0.2})]
    assert result["endpoint"] == "biogpt-tgi-service:8001"
    assert run_helm.calls == []


def test_generate_completion_deploys_the_requested_backend(chart_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    checked: list[str] = []
    generated: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(manager, "check_health", healthy_check(checked))
    monkeypatch.setattr(manager, "_generate_from_endpoint", recording_generate(generated))

    asyncio.run(manager.generate_completion("biogpt", "Hi", backend="tensorrt"))

    assert [call[3] for call in run_helm.calls] == ["biogpt-tensorrt"]
    assert sleep.delays == [10, 5]
    assert checked == ["biogpt-tensorrt"]
    assert generated == [("biogpt-tensorrt", "Hi", {})]


def test_generate_completion_tries_each_backend_in_turn(chart_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    checked: list[str] = []

    async def check_health(endpoint: ModelEndpoint) -> BackendStatus:
        checked.append(endpoint.helm_release)
        endpoint.status = BackendStatus.HEALTHY if endpoint.backend == "tensorrt" else BackendStatus.UNHEALTHY
        return endpoint.status

    generated: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(manager, "check_health", check_health)
    monkeypatch.setattr(manager, "_generate_from_endpoint", recording_generate(generated))

    asyncio.run(manager.generate_completion("chemberta", "Hi"))

    assert checked == ["chemberta-vllm", "chemberta-tgi", "chemberta-tensorrt"]
    assert sleep.delays == [10, 5, 10, 5, 10, 5]
    assert generated == [("chemberta-tensorrt", "Hi", {})]


def test_generate_completion_without_a_healthy_endpoint(chart_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    run_helm = FakeHelm(returncode=1)
    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=RecordingSleep())

    with pytest.raises(RuntimeError, match=r"^No healthy endpoints available for model: biogpt$"):
        asyncio.run(manager.generate_completion("biogpt", "Hi"))
    assert len(run_helm.calls) == 3

    with pytest.raises(RuntimeError, match=r"^No healthy endpoints available for model: gpt$"):
        asyncio.run(manager.generate_completion("gpt", "Hi"))
    assert len(run_helm.calls) == 3


def test_get_system_status_counts_and_order() -> None:
    manager = BackendManager("ns")
    vllm, tgi, _ = manager.endpoints["biogpt"]
    vllm.deployment_status, vllm.status = BackendStatus.DEPLOYED, BackendStatus.HEALTHY
    tgi.deployment_status, tgi.status = BackendStatus.DEPLOYED, BackendStatus.UNKNOWN  # deployed, not healthy
    stuck = endpoint_of(manager, "matscibert", "tensorrt")
    stuck.deployment_status, stuck.status = BackendStatus.DEPLOYING, BackendStatus.HEALTHY  # not deployed
    vllm.performance_metrics["latency_ms"] = 12.5

    status = manager.get_system_status()

    assert list(status) == ["timestamp", "matrix", "summary"]
    datetime.fromisoformat(status["timestamp"])
    assert list(status["matrix"]) == REGISTRY_MODELS
    assert [model["domain"] for model in status["matrix"].values()] == ["biology", "chemistry", "materials"]
    assert all(list(model["backends"]) == REGISTRY_BACKENDS for model in status["matrix"].values())
    assert status["matrix"]["biogpt"]["backends"]["vllm"] == {
        "status": "healthy",
        "deployment_status": "deployed",
        "endpoint": "biogpt-vllm-service:8000",
        "performance": {"latency_ms": 12.5, "throughput": 0, "cost_per_token": 0, "success_rate": 100},
    }
    assert status["matrix"]["matscibert"]["backends"]["tensorrt"]["deployment_status"] == "deploying"
    assert status["summary"] == {"total_endpoints": 9, "deployed": 2, "healthy": 1, "unhealthy": 1, "not_deployed": 7}
    assert list(status["summary"]) == ["total_endpoints", "deployed", "healthy", "unhealthy", "not_deployed"]


# ---------------------------------------------------------------- Orchestrator: setup and umbrella helm


EXPECTED_DEPLOYMENT_KEYS = [
    "matscibert-tgi",
    "matscibert-vllm",
    "matscibert-tensorrt",
    "biogpt-tgi",
    "biogpt-vllm",
    "biogpt-tensorrt",
    "chemberta-tgi",
    "chemberta-vllm",
    "chemberta-tensorrt",
]


def test_orchestrator_setup() -> None:
    orchestrator = Orchestrator("ns", chart_dir="charts-dir")

    assert list(orchestrator.deployment_configs) == EXPECTED_DEPLOYMENT_KEYS
    for key, config in orchestrator.deployment_configs.items():
        model, backend = key.split("-")
        assert config == DeploymentConfig(model_name=model, backend=backend, helm_release=key, namespace="ns")
        assert (config.enabled, config.last_used, config.usage_count, config.auto_shutdown_minutes) == (
            False,
            None,
            0,
            30,
        )
    assert orchestrator.namespace == "ns"
    assert orchestrator.umbrella_chart_path == "charts-dir"
    assert orchestrator.auto_shutdown_enabled is True
    assert orchestrator.monitoring_interval == 60
    # It owns a BackendManager in the same namespace, with the manager's own registry order.
    assert orchestrator.backend_manager.namespace == "ns"
    assert list(orchestrator.backend_manager.endpoints) == REGISTRY_MODELS
    assert (MATRIX_VIEW_MODELS, MATRIX_VIEW_BACKENDS) == (
        ("matscibert", "biogpt", "chemberta"),
        ("tgi", "vllm", "tensorrt"),
    )


def test_orchestrator_defaults() -> None:
    orchestrator = Orchestrator()
    assert orchestrator.namespace == "default"
    assert orchestrator.umbrella_chart_path == DEFAULT_CHART_DIR
    assert orchestrator.backend_manager.namespace == "default"


def test_orchestrator_shares_helm_and_sleep_with_its_backend_manager(chart_dir: str) -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    orchestrator = Orchestrator("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
    manager = orchestrator.backend_manager

    assert asyncio.run(manager.deploy_endpoint(endpoint_of(manager, "biogpt", "vllm"))) is True
    assert run_helm.calls == [helm.install_cmd("biogpt-vllm", os.path.join(chart_dir, "charts", "biogpt-vllm"), "ns")]
    assert sleep.delays == [10]


def test_orchestrator_uses_an_injected_backend_manager() -> None:
    manager = BackendManager("elsewhere")
    assert Orchestrator("ns", backend_manager=manager).backend_manager is manager


def test_orchestrator_deploy_enables_the_release(matrix_log: pytest.LogCaptureFixture) -> None:
    when = datetime(2026, 5, 1, 9, 30)
    run_helm = FakeHelm()
    orchestrator = Orchestrator("ns", run_helm=run_helm, now=lambda: when)

    assert asyncio.run(orchestrator.deploy_endpoint("biogpt", "vllm")) is True

    assert run_helm.calls == [
        [
            "helm",
            "upgrade",
            "--install",
            "multi-llm",
            "./deploy/helm/pick-and-spin-umbrella",
            "--namespace",
            "ns",
            "--create-namespace",
            "--set",
            "biogpt_vllm.enabled=true",
            "--wait",
            "--timeout",
            "5m",
        ]
    ]
    config = orchestrator.deployment_configs["biogpt-vllm"]
    assert (config.enabled, config.last_used, config.usage_count) == (True, when, 1)

    # An enabled pair is not deployed again.
    assert asyncio.run(orchestrator.deploy_endpoint("biogpt", "vllm")) is True
    assert len(run_helm.calls) == 1
    assert config.usage_count == 1
    assert messages(matrix_log) == [
        "Deploying biogpt-vllm with helm...",
        "Successfully deployed biogpt-vllm",
        "biogpt-vllm is already deployed",
    ]


def test_orchestrator_deploy_failures(matrix_log: pytest.LogCaptureFixture) -> None:
    run_helm = FakeHelm(returncode=1, stderr="denied")
    orchestrator = Orchestrator("ns", run_helm=run_helm)

    assert asyncio.run(orchestrator.deploy_endpoint("gpt", "vllm")) is False
    assert run_helm.calls == []
    assert asyncio.run(orchestrator.deploy_endpoint("biogpt", "vllm")) is False
    assert orchestrator.deployment_configs["biogpt-vllm"] == DeploymentConfig("biogpt", "vllm", "biogpt-vllm", "ns")

    orchestrator = Orchestrator("ns", run_helm=FakeHelm(error=OSError("no helm")))
    assert asyncio.run(orchestrator.deploy_endpoint("biogpt", "vllm")) is False

    assert messages(matrix_log, logging.ERROR) == [
        "Unknown deployment configuration: gpt-vllm",
        "Failed to deploy biogpt-vllm: denied",
        "Error deploying biogpt-vllm: no helm",
    ]


def test_orchestrator_undeploy_disables_the_release(matrix_log: pytest.LogCaptureFixture) -> None:
    run_helm = FakeHelm()
    orchestrator = Orchestrator("ns", chart_dir="charts-dir", run_helm=run_helm)
    config = orchestrator.deployment_configs["matscibert-tgi"]
    config.enabled = True

    assert asyncio.run(orchestrator.undeploy_endpoint("matscibert", "tgi")) is True
    assert run_helm.calls == [
        [
            "helm",
            "upgrade",
            "multi-llm",
            "charts-dir",
            "--namespace",
            "ns",
            "--set",
            "matscibert_tgi.enabled=false",
            "--wait",
            "--timeout",
            "2m",
        ]
    ]
    assert config.enabled is False

    assert asyncio.run(orchestrator.undeploy_endpoint("matscibert", "tgi")) is True
    assert len(run_helm.calls) == 1
    assert messages(matrix_log) == [
        "Undeploying matscibert-tgi...",
        "Successfully undeployed matscibert-tgi",
        "matscibert-tgi is already undeployed",
    ]


def test_orchestrator_undeploy_failures(matrix_log: pytest.LogCaptureFixture) -> None:
    orchestrator = Orchestrator("ns", run_helm=FakeHelm(returncode=1, stderr="timed out"))
    orchestrator.deployment_configs["biogpt-tgi"].enabled = True

    assert asyncio.run(orchestrator.undeploy_endpoint("gpt", "tgi")) is False
    assert asyncio.run(orchestrator.undeploy_endpoint("biogpt", "tgi")) is False
    assert orchestrator.deployment_configs["biogpt-tgi"].enabled is True

    orchestrator = Orchestrator("ns", run_helm=FakeHelm(error=OSError("no helm")))
    orchestrator.deployment_configs["biogpt-tgi"].enabled = True
    assert asyncio.run(orchestrator.undeploy_endpoint("biogpt", "tgi")) is False

    assert messages(matrix_log, logging.ERROR) == [
        "Unknown deployment configuration: gpt-tgi",
        "Failed to undeploy biogpt-tgi: timed out",
        "Error undeploying biogpt-tgi: no helm",
    ]


# ---------------------------------------------------------------- Orchestrator: requests


def test_process_request_deploys_on_demand_and_classifies_twice(
    chart_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2026, 5, 1, 9, 30)
    run_helm, sleep = FakeHelm(), RecordingSleep()
    prompts: list[str] = []

    def classify(prompt: str) -> tuple[DomainType, float]:
        prompts.append(prompt)
        return DomainType.BIOLOGY, 0.9

    manager = BackendManager("ns", chart_dir=chart_dir, run_helm=run_helm, sleep=sleep, classify=classify)
    checked: list[str] = []
    generated: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(manager, "check_health", healthy_check(checked))
    monkeypatch.setattr(manager, "_generate_from_endpoint", recording_generate(generated))
    orchestrator = Orchestrator(
        "ns", chart_dir=chart_dir, backend_manager=manager, run_helm=run_helm, sleep=sleep, now=lambda: when
    )

    result = asyncio.run(orchestrator.process_request("protein folding", "quality"))

    # The prototype selects the endpoint twice: once here, once inside generate_completion_with_routing.
    assert prompts == ["protein folding", "protein folding"]
    assert run_helm.calls == [
        helm.install_cmd("biogpt-vllm", os.path.join(chart_dir, "charts", "biogpt-vllm"), "ns"),
        helm.umbrella_enable_cmd(chart_dir, "ns", "biogpt-vllm"),
    ]
    assert sleep.delays == [10, 5, 10]
    assert checked == ["biogpt-vllm"]
    assert generated == [("biogpt-vllm", "protein folding", {})]
    assert list(result) == [
        "model",
        "backend",
        "response",
        "endpoint",
        "selection_time_ms",
        "total_time_ms",
        "domain",
        "orchestration_time_ms",
        "deployment_config",
    ]
    assert result["deployment_config"] == "biogpt-vllm"
    assert result["domain"] == "biology"
    assert result["orchestration_time_ms"] >= 0
    config = orchestrator.deployment_configs["biogpt-vllm"]
    # Counted once by the deploy and once by the request, as before.
    assert (config.enabled, config.last_used, config.usage_count) == (True, when, 2)


class FakeBackendManager:
    """Stands in for BackendManager in process_request: a fixed endpoint and a fixed result, or an error."""

    def __init__(self, endpoint: ModelEndpoint | None, error: Exception | None = None) -> None:
        self.endpoint = endpoint
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    async def select_optimal_endpoint(self, prompt: str, optimize_for: str = "balanced") -> ModelEndpoint | None:
        self.calls.append(("select", prompt, optimize_for))
        return self.endpoint

    async def generate_completion_with_routing(self, prompt: str, optimize_for: str = "balanced") -> dict[str, Any]:
        self.calls.append(("generate", prompt, optimize_for))
        if self.error is not None:
            raise self.error
        return {"model": "m", "response": "text"}


def make_orchestrator(
    manager: FakeBackendManager, run_helm: FakeHelm | None = None, sleep: RecordingSleep | None = None
) -> Orchestrator:
    return Orchestrator(
        "ns",
        backend_manager=manager,
        run_helm=run_helm or FakeHelm(),
        sleep=sleep or RecordingSleep(),
        now=lambda: datetime(2026, 5, 1, 9, 30),
    )


def test_process_request_without_an_endpoint() -> None:
    manager = FakeBackendManager(None)
    result = asyncio.run(make_orchestrator(manager).process_request("x", "speed"))

    assert result == {"error": "No suitable endpoint could be selected", "status": "failed"}
    assert manager.calls == [("select", "x", "speed")]


def test_process_request_when_the_deploy_fails() -> None:
    sleep = RecordingSleep()
    manager = FakeBackendManager(default_endpoints("ns")["chemberta"][1])
    orchestrator = make_orchestrator(manager, FakeHelm(returncode=1), sleep)

    result = asyncio.run(orchestrator.process_request("acid"))

    assert result == {"error": "Failed to deploy chemberta-tgi", "status": "deployment_failed"}
    assert manager.calls == [("select", "acid", "balanced")]
    assert sleep.delays == []
    assert orchestrator.deployment_configs["chemberta-tgi"].usage_count == 0


def test_process_request_with_an_enabled_deployment() -> None:
    run_helm, sleep = FakeHelm(), RecordingSleep()
    manager = FakeBackendManager(default_endpoints("ns")["chemberta"][1])
    orchestrator = make_orchestrator(manager, run_helm, sleep)
    config = orchestrator.deployment_configs["chemberta-tgi"]
    config.enabled = True

    result = asyncio.run(orchestrator.process_request("acid", "cost"))

    assert manager.calls == [("select", "acid", "cost"), ("generate", "acid", "cost")]
    assert (run_helm.calls, sleep.delays) == ([], [])
    assert list(result) == ["model", "response", "orchestration_time_ms", "deployment_config"]
    assert result["deployment_config"] == "chemberta-tgi"
    assert (config.last_used, config.usage_count) == (datetime(2026, 5, 1, 9, 30), 1)


def test_process_request_with_an_endpoint_outside_the_configs() -> None:
    manager = FakeBackendManager(ModelEndpoint("gpt", "vllm", "host", 1, DomainType.GENERAL))
    run_helm = FakeHelm()

    result = asyncio.run(make_orchestrator(manager, run_helm).process_request("x"))

    assert result["deployment_config"] == "unknown"
    assert run_helm.calls == []


def test_process_request_when_generation_fails(matrix_log: pytest.LogCaptureFixture) -> None:
    manager = FakeBackendManager(default_endpoints("ns")["biogpt"][0], error=RuntimeError("Generation failed"))
    orchestrator = make_orchestrator(manager)
    orchestrator.deployment_configs["biogpt-vllm"].enabled = True

    result = asyncio.run(orchestrator.process_request("protein"))

    assert result == {"error": "Generation failed", "status": "generation_failed"}
    assert messages(matrix_log, logging.ERROR) == ["Error processing request: Generation failed"]


# ---------------------------------------------------------------- Orchestrator: idle shutdown and status


def test_auto_shutdown_undeploys_idle_deployments(matrix_log: pytest.LogCaptureFixture) -> None:
    now = datetime(2026, 5, 1, 12, 0)
    run_helm, sleep = FakeHelm(), RecordingSleep()
    orchestrator = Orchestrator("ns", chart_dir="charts-dir", run_helm=run_helm, sleep=sleep, now=lambda: now)
    sleep.on_sleep = lambda delay: setattr(orchestrator, "auto_shutdown_enabled", False)  # one iteration
    configs = orchestrator.deployment_configs
    for key, enabled, idle in [
        ("biogpt-tgi", True, timedelta(minutes=31)),
        ("matscibert-vllm", True, timedelta(minutes=30)),  # not more than 30 minutes
        ("chemberta-tensorrt", True, None),  # never used: never shut down
        ("biogpt-vllm", False, timedelta(hours=2)),  # not enabled
    ]:
        configs[key].enabled = enabled
        configs[key].last_used = None if idle is None else now - idle

    asyncio.run(orchestrator.auto_shutdown_unused())

    assert run_helm.calls == [helm.umbrella_disable_cmd("charts-dir", "ns", "biogpt-tgi")]
    assert sleep.delays == [60]
    assert [key for key, config in configs.items() if config.enabled] == ["matscibert-vllm", "chemberta-tensorrt"]
    assert messages(matrix_log, logging.INFO)[0] == "Auto-shutting down biogpt-tgi (unused for 0:31:00)"


def test_auto_shutdown_keeps_running_after_an_error(matrix_log: pytest.LogCaptureFixture) -> None:
    reads: list[Exception | datetime] = [RuntimeError("clock failed"), datetime(2026, 5, 1, 12, 0)]

    def now() -> datetime:
        value = reads.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    sleep = RecordingSleep()
    orchestrator = Orchestrator("ns", run_helm=FakeHelm(), sleep=sleep, now=now)

    def stop_after_two(delay: float) -> None:
        if len(sleep.delays) == 2:
            orchestrator.auto_shutdown_enabled = False

    sleep.on_sleep = stop_after_two

    asyncio.run(orchestrator.auto_shutdown_unused())

    assert sleep.delays == [60, 60]
    assert reads == []
    assert messages(matrix_log, logging.ERROR) == ["Error in auto-shutdown monitoring: clock failed"]


def test_auto_shutdown_stops_when_disabled() -> None:
    sleep = RecordingSleep()
    orchestrator = Orchestrator("ns", sleep=sleep)
    orchestrator.auto_shutdown_enabled = False
    asyncio.run(orchestrator.auto_shutdown_unused())
    assert sleep.delays == []


def test_get_deployment_status(fake_clock: Any) -> None:
    t0 = datetime(2026, 5, 1, 12, 0)
    # Two clock reads, as before: the timestamp, then the reference time for 'time since use'.
    clock = fake_clock(t0, t0 + timedelta(seconds=1))
    orchestrator = Orchestrator("ns", now=clock)
    configs = orchestrator.deployment_configs
    for key, enabled, idle, uses in [
        ("biogpt-vllm", True, timedelta(minutes=10), 3),  # idle: more than 5 minutes
        ("chemberta-tgi", True, timedelta(minutes=4), 1),
        ("matscibert-tensorrt", False, timedelta(hours=1), 2),  # not enabled: never idle
    ]:
        configs[key].enabled = enabled
        configs[key].last_used = t0 - idle
        configs[key].usage_count = uses

    status = orchestrator.get_deployment_status()

    assert clock.remaining == 0
    assert list(status) == ["timestamp", "namespace", "deployments", "summary"]
    assert status["timestamp"] == "2026-05-01T12:00:00"
    assert status["namespace"] == "ns"
    assert list(status["deployments"]) == EXPECTED_DEPLOYMENT_KEYS
    assert status["deployments"]["biogpt-vllm"] == {
        "model": "biogpt",
        "backend": "vllm",
        "enabled": True,
        "usage_count": 3,
        "last_used": "2026-05-01T11:50:00",
        "time_since_use_seconds": 601.0,
        "auto_shutdown_minutes": 30,
    }
    assert status["deployments"]["chemberta-tgi"]["time_since_use_seconds"] == 241.0
    assert status["deployments"]["matscibert-tensorrt"]["time_since_use_seconds"] == 3601.0
    assert status["deployments"]["biogpt-tgi"] == {
        "model": "biogpt",
        "backend": "tgi",
        "enabled": False,
        "usage_count": 0,
        "last_used": None,
        "time_since_use_seconds": None,
        "auto_shutdown_minutes": 30,
    }
    assert status["summary"] == {"total": 9, "deployed": 2, "idle": 1, "total_usage": 6}


def test_matrix_view_of_an_idle_matrix() -> None:
    assert Orchestrator().get_matrix_view().split("\n") == [
        "3x3 LLM Matrix Status",
        "=" * 60,
        "Model           | TGI          | vLLM         | TensorRT    ",
        "-" * 60,
        "matscibert      | OFF        | OFF        | OFF        |",
        "biogpt          | OFF        | OFF        | OFF        |",
        "chemberta       | OFF        | OFF        | OFF        |",
        "=" * 60,
        "Deployed: 0/9 | Resource Usage: 0 GPU(s)",
        "",
    ]
    assert f"{'Model':<15} | {'TGI':<12} | {'vLLM':<12} | {'TensorRT':<12}" == (
        "Model           | TGI          | vLLM         | TensorRT    "
    )


def test_matrix_view_marks_deployed_and_unknown_cells() -> None:
    orchestrator = Orchestrator()
    orchestrator.deployment_configs["biogpt-vllm"].enabled = True
    orchestrator.deployment_configs["chemberta-tensorrt"].enabled = True
    del orchestrator.deployment_configs["matscibert-tgi"]

    lines = orchestrator.get_matrix_view().splitlines()

    assert lines[4:7] == [
        "matscibert      | ? UNKNOWN  | OFF        | OFF        |",
        "biogpt          | OFF        | DEPLOYED   | OFF        |",
        "chemberta       | OFF        | OFF        | DEPLOYED   |",
    ]
    assert lines[8] == "Deployed: 2/8 | Resource Usage: 2 GPU(s)"


# ---------------------------------------------------------------- the API module without its extra


def test_api_module_needs_pydantic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pydantic", None)
    monkeypatch.delitem(sys.modules, "mmorch.matrix.api", raising=False)

    with pytest.raises(MissingDependencyError) as excinfo:
        importlib.import_module("mmorch.matrix.api")

    assert (excinfo.value.package, excinfo.value.extra) == ("pydantic", "matrix")


def test_generate_from_endpoint_with_an_unknown_backend(
    monkeypatch: pytest.MonkeyPatch, matrix_log: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "aiohttp", types.ModuleType("aiohttp"))  # importable; never reached
    endpoint = ModelEndpoint("chemberta", "onnx", "host", 9, DomainType.CHEMISTRY)

    with pytest.raises(ValueError, match="onnx"):
        asyncio.run(BackendManager()._generate_from_endpoint(endpoint, "Hi"))

    # As before, it counts as a failed generation.
    assert endpoint.performance_metrics["success_rate"] == 90
    assert messages(matrix_log, logging.ERROR)[0].startswith("Generation failed for chemberta on onnx: ")
