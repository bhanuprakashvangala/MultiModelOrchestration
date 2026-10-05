"""A pure description of the 3x3 prototype, with no I/O.

- The enums, with one DomainType instead of the legacy duplicate. They are plain Enums, rendered with .value.
- ModelEndpoint with its per-backend URLs and metrics reset.
- The endpoint registry in legacy order, and the domain -> model map.
- Backend priorities and the endpoint scoring rule.
- Per-backend request payloads.
- The /api/v1/models catalog.

Two orders are behaviour and stay as they were: the registry here lists biogpt, chemberta, matscibert x vllm,
tgi, tensorrt (ports 8000-8008), while mmorch.matrix.orchestrator lists matscibert, biogpt, chemberta x tgi, vllm,
tensorrt. They decide ties, the key order of the JSON status documents and the rows of the matrix view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


class BackendType(Enum):
    """An inference server."""

    VLLM = "vllm"
    TGI = "tgi"
    TENSORRT = "tensorrt"


class BackendStatus(Enum):
    """The health or deployment state of an endpoint."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEPLOYED = "deployed"
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"


class DomainType(Enum):
    """The science domain of a prompt."""

    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    GENERAL = "general"


@dataclass
class ModelEndpoint:
    """One model served by one backend: its address, Helm release, state and running metrics.

    The health and generate URLs follow from the backend and stay empty for an unknown one. Whatever metrics are
    passed in, they start as the legacy ints: latency, throughput and cost 0, success rate 100.
    """

    model: str
    backend: str
    host: str
    port: int
    domain: DomainType
    status: BackendStatus = BackendStatus.NOT_DEPLOYED
    deployment_status: BackendStatus = BackendStatus.NOT_DEPLOYED
    helm_release: str = ""
    namespace: str = "default"
    performance_metrics: dict[str, float] = field(default_factory=dict)
    health_check_url: str = field(init=False, default="", repr=False)
    generate_url: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        """Set the backend's health and generate URLs and reset the performance metrics."""
        if self.backend == "vllm":
            self.health_check_url = f"http://{self.host}:{self.port}/health"
            self.generate_url = f"http://{self.host}:{self.port}/v1/completions"
        elif self.backend == "tgi":
            self.health_check_url = f"http://{self.host}:{self.port}/health"
            self.generate_url = f"http://{self.host}:{self.port}/generate"
        elif self.backend == "tensorrt":
            self.health_check_url = f"http://{self.host}:{self.port}/v2/health/ready"
            self.generate_url = f"http://{self.host}:{self.port}/v2/models/{self.model}/infer"

        self.performance_metrics = {
            "latency_ms": 0,
            "throughput": 0,
            "cost_per_token": 0,
            "success_rate": 100,
        }

    @property
    def address(self) -> str:
        """The endpoint as 'host:port'."""
        return f"{self.host}:{self.port}"


MODEL_FOR_DOMAIN: Final[Mapping[DomainType, str]] = MappingProxyType(
    {
        DomainType.BIOLOGY: "biogpt",
        DomainType.CHEMISTRY: "chemberta",
        DomainType.MATERIALS: "matscibert",
        DomainType.GENERAL: "biogpt",  # general prompts go to biogpt
    }
)

# Lower is better; the keys of the outer mapping are the optimize_for values.
BACKEND_PRIORITIES: Final[Mapping[str, Mapping[str, int]]] = MappingProxyType(
    {
        "speed": {"tensorrt": 1, "vllm": 2, "tgi": 3},
        "cost": {"tgi": 1, "vllm": 2, "tensorrt": 3},
        "quality": {"vllm": 1, "tensorrt": 2, "tgi": 3},
        "balanced": {"vllm": 1, "tgi": 2, "tensorrt": 3},
    }
)

# The order among healthy endpoints when no backend is preferred.
HEALTHY_PRIORITY: Final = {"vllm": 1, "tgi": 2, "tensorrt": 3}


def default_endpoints(namespace: str) -> dict[str, list[ModelEndpoint]]:
    """Return the nine endpoints (biogpt, chemberta, matscibert x vllm, tgi, tensorrt) on ports 8000-8008.

    Each endpoint's Helm release is '<model>-<backend>' and its namespace is the given one.
    """
    endpoints = {
        "biogpt": [
            ModelEndpoint("biogpt", "vllm", "biogpt-vllm-service", 8000, DomainType.BIOLOGY),
            ModelEndpoint("biogpt", "tgi", "biogpt-tgi-service", 8001, DomainType.BIOLOGY),
            ModelEndpoint("biogpt", "tensorrt", "biogpt-tensorrt-service", 8002, DomainType.BIOLOGY),
        ],
        "chemberta": [
            ModelEndpoint("chemberta", "vllm", "chemberta-vllm-service", 8003, DomainType.CHEMISTRY),
            ModelEndpoint("chemberta", "tgi", "chemberta-tgi-service", 8004, DomainType.CHEMISTRY),
            ModelEndpoint("chemberta", "tensorrt", "chemberta-tensorrt-service", 8005, DomainType.CHEMISTRY),
        ],
        "matscibert": [
            ModelEndpoint("matscibert", "vllm", "matscibert-vllm-service", 8006, DomainType.MATERIALS),
            ModelEndpoint("matscibert", "tgi", "matscibert-tgi-service", 8007, DomainType.MATERIALS),
            ModelEndpoint("matscibert", "tensorrt", "matscibert-tensorrt-service", 8008, DomainType.MATERIALS),
        ],
    }

    for model_name, model_endpoints in endpoints.items():
        for endpoint in model_endpoints:
            endpoint.helm_release = f"{model_name}-{endpoint.backend}"
            endpoint.namespace = namespace

    return endpoints


def choose_endpoint(endpoints: Sequence[ModelEndpoint], optimize_for: str) -> ModelEndpoint | None:
    """Return the endpoint with the lowest priority-plus-performance score; the first one wins a tie.

    The score is the backend's priority for optimize_for (an unknown optimize_for uses 'balanced', an unknown
    backend 99) plus latency_ms / 1000 plus (100 - success_rate) / 100. Only a strictly lower score replaces the
    current best, so equal scores keep the earlier endpoint. None when no score is below infinity, as for an
    empty sequence.
    """
    priorities = BACKEND_PRIORITIES.get(optimize_for, BACKEND_PRIORITIES["balanced"])

    best_endpoint: ModelEndpoint | None = None
    best_score = float("inf")

    for endpoint in endpoints:
        priority_score = priorities.get(endpoint.backend, 99)
        perf_score = (
            endpoint.performance_metrics.get("latency_ms", 1000) / 1000
            + (100 - endpoint.performance_metrics.get("success_rate", 0)) / 100
        )
        total_score = priority_score + perf_score

        if total_score < best_score:
            best_score = total_score
            best_endpoint = endpoint

    return best_endpoint


def build_payload(
    endpoint: ModelEndpoint,
    prompt: str,
    *,
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Return the request body for the endpoint's backend; ValueError for an unknown backend.

    - vllm: an OpenAI completions body {model, prompt, max_tokens, temperature}
    - tgi: {inputs, parameters: {max_new_tokens, temperature}}
    - tensorrt: a KServe v2 inference body with one BYTES input 'input_text', plus {max_tokens, temperature}
    """
    if endpoint.backend == "vllm":
        return {
            "model": endpoint.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    if endpoint.backend == "tgi":
        return {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    if endpoint.backend == "tensorrt":
        return {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt],
                }
            ],
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    raise ValueError(f"Unknown backend: {endpoint.backend!r}")


# The /api/v1/models entries, verbatim and in order.
MODEL_CATALOG: Final[tuple[dict[str, object], ...]] = (
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
