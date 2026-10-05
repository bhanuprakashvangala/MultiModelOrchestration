"""Tests for the prototype's HTTP side: the FastAPI app of mmorch.matrix.api through FastAPI's in-process
TestClient, and BackendManager's aiohttp requests against a stub model server on 127.0.0.1.

They need the 'matrix' extra (fastapi, pydantic, aiohttp) and httpx, and skip without them. helm never runs: the
API tests use fakes, and the one test that wires the real managers answers subprocess.run itself.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socketserver
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mmorch.errors import MissingDependencyError
from mmorch.matrix.backend_manager import BackendManager
from mmorch.matrix.endpoints import MODEL_CATALOG, BackendStatus, DomainType, ModelEndpoint, default_endpoints
from mmorch.settings import MatrixSettings

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # behind fastapi.testclient

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mmorch.matrix.api import DESCRIPTION, TITLE, VERSION, create_app, serve

# ---------------------------------------------------------------- fakes and fixtures


class FakeManager:
    """Stands in for the BackendManager behind /api/v1/generate and /api/v1/status; records its calls."""

    def __init__(self) -> None:
        self.endpoints = default_endpoints("default")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result: dict[str, Any] = {
            "model": "biogpt",
            "backend": "tgi",
            "response": {"generated_text": "Hello"},
            "endpoint": "biogpt-tgi-service:8001",
        }
        self.error: Exception | None = None

    async def generate_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("generate_completion", kwargs))
        if self.error is not None:
            raise self.error
        return dict(self.result)

    async def check_all_health(self) -> None:
        self.calls.append(("check_all_health", {}))

    def get_system_status(self) -> dict[str, Any]:
        self.calls.append(("get_system_status", {}))
        return {"timestamp": "2026-05-01T12:00:00", "matrix": {}, "summary": {"total_endpoints": 9}}


class FakeOrchestrator:
    """Stands in for the Orchestrator; records its calls. Its monitor task runs until it is cancelled."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.result: dict[str, Any] = {
            "model": "chemberta",
            "backend": "tensorrt",
            "response": {"outputs": []},
            "endpoint": "chemberta-tensorrt-service:8005",
            "selection_time_ms": 1.5,
            "total_time_ms": 3.0,
            "domain": "chemistry",
            "orchestration_time_ms": 4.0,
            "deployment_config": "chemberta-tensorrt",
        }
        self.error: Exception | None = None
        self.succeed = True
        self.monitor_started = threading.Event()
        self.monitor_cancelled = threading.Event()

    async def process_request(self, prompt: str, optimize_for: str = "balanced") -> dict[str, Any]:
        self.calls.append(("process_request", prompt, optimize_for))
        if self.error is not None:
            raise self.error
        return dict(self.result)

    async def deploy_endpoint(self, model: str, backend: str) -> bool:
        self.calls.append(("deploy_endpoint", model, backend))
        return self.succeed

    async def undeploy_endpoint(self, model: str, backend: str) -> bool:
        self.calls.append(("undeploy_endpoint", model, backend))
        return self.succeed

    def get_deployment_status(self) -> dict[str, Any]:
        return {"namespace": "ns", "deployments": {}, "summary": {"total": 9}}

    def get_matrix_view(self) -> str:
        return "3x3 LLM Matrix Status\nDeployed: 0/9 | Resource Usage: 0 GPU(s)\n"

    async def auto_shutdown_unused(self) -> None:
        self.monitor_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.monitor_cancelled.set()
            raise


@dataclass
class Api:
    client: TestClient
    app: FastAPI
    manager: FakeManager
    orchestrator: FakeOrchestrator
    created: list[FakeOrchestrator]


@pytest.fixture
def api() -> Iterator[Api]:
    """The app with fakes behind it, served by a TestClient that runs its lifespan; created lists factory calls."""
    manager, orchestrator = FakeManager(), FakeOrchestrator()
    created: list[FakeOrchestrator] = []

    def factory() -> FakeOrchestrator:
        created.append(orchestrator)
        return orchestrator

    app = create_app(MatrixSettings(), backend_manager=manager, orchestrator_factory=factory)
    with TestClient(app) as client:
        yield Api(client, app, manager, orchestrator, created)


@pytest.fixture
def matrix_log(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog at DEBUG for mmorch.matrix, also after a CLI test has stopped the 'mmorch' logger propagating."""
    monkeypatch.setattr(logging.getLogger("mmorch"), "propagate", True)
    caplog.set_level(logging.DEBUG, logger="mmorch.matrix")
    return caplog


def messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("mmorch.matrix") and r.levelno == level]


# ---------------------------------------------------------------- the app


def test_app_metadata_and_docs(api: Api) -> None:
    assert (api.app.title, api.app.description, api.app.version) == (
        "3x3 Domain-Specific LLM Matrix API",
        "Intelligent routing system for domain-specific LLMs (BioGPT, ChemBERTa, MatSciBERT) x (vLLM, TGI, TensorRT)",
        "2.0.0",
    )
    assert (api.app.title, api.app.description, api.app.version) == (TITLE, DESCRIPTION, VERSION)
    assert api.client.get("/api/docs").status_code == 200
    assert api.client.get("/api/redoc").status_code == 200
    assert api.client.get("/docs").status_code == 404


def test_openapi_operations_are_the_legacy_ones(api: Api) -> None:
    document = api.client.get("/openapi.json").json()

    operations = [
        (path, method, operation["operationId"], operation["summary"], operation["description"])
        for path, methods in document["paths"].items()
        for method, operation in methods.items()
    ]
    # The ids, summaries and descriptions come from the route functions' names and docstrings.
    assert operations == [
        (
            "/api/v1/generate",
            "post",
            "generate_text_api_v1_generate_post",
            "Generate Text",
            "Generate text using the specified model and backend",
        ),
        (
            "/api/v1/status",
            "get",
            "get_system_status_api_v1_status_get",
            "Get System Status",
            "Get system status for all models and backends",
        ),
        ("/api/v1/models", "get", "list_models_api_v1_models_get", "List Models", "List available models"),
        (
            "/api/v1/intelligent-route",
            "post",
            "intelligent_route_api_v1_intelligent_route_post",
            "Intelligent Route",
            "Use intelligent routing to automatically select the best model and backend",
        ),
        (
            "/api/v1/orchestrator/status",
            "get",
            "get_orchestrator_status_api_v1_orchestrator_status_get",
            "Get Orchestrator Status",
            "Get orchestrator deployment status",
        ),
        (
            "/api/v1/orchestrator/matrix",
            "get",
            "get_matrix_view_api_v1_orchestrator_matrix_get",
            "Get Matrix View",
            "Get visual matrix representation",
        ),
        (
            "/api/v1/orchestrator/deploy",
            "post",
            "deploy_endpoint_api_v1_orchestrator_deploy_post",
            "Deploy Endpoint",
            "Manually deploy a specific endpoint",
        ),
        (
            "/api/v1/orchestrator/undeploy",
            "post",
            "undeploy_endpoint_api_v1_orchestrator_undeploy_post",
            "Undeploy Endpoint",
            "Manually undeploy a specific endpoint",
        ),
        ("/health", "get", "health_check_health_get", "Health Check", "Health check endpoint"),
    ]

    for action in ("deploy", "undeploy"):
        parameters = document["paths"][f"/api/v1/orchestrator/{action}"]["post"]["parameters"]
        assert [(p["name"], p["in"], p["required"]) for p in parameters] == [
            ("model", "query", True),
            ("backend", "query", True),
        ]

    schemas = document["components"]["schemas"]
    assert list(schemas["GenerateRequest"]["properties"]) == [
        "prompt",
        "model",
        "backend",
        "max_tokens",
        "temperature",
        "optimize_for",
    ]
    assert schemas["GenerateRequest"]["required"] == ["prompt"]
    assert schemas["GenerateRequest"]["properties"]["max_tokens"] == {
        "type": "integer",
        "maximum": 2000.0,
        "minimum": 1.0,
        "title": "Max Tokens",
        "default": 150,
    }
    assert list(schemas["IntelligentRoutingRequest"]["properties"]) == [
        "prompt",
        "optimize_for",
        "max_tokens",
        "temperature",
    ]
    assert schemas["GenerateResponse"]["required"] == [
        "text",
        "model",
        "backend",
        "endpoint",
        "domain",
        "response_time_ms",
        "selection_time_ms",
    ]
    # Pydantic would publish model docstrings here; the models have none, as before.
    assert all("description" not in schemas[name] for name in ("GenerateRequest", "GenerateResponse"))


def test_cors_allows_any_origin(api: Api) -> None:
    origin = {"Origin": "https://lab.example.org"}

    response = api.client.get("/health", headers=origin)
    # Starlette releases differ between '*' and the echoed origin here; both allow any origin.
    assert response.headers["access-control-allow-origin"] in ("*", "https://lab.example.org")
    assert response.headers["access-control-allow-credentials"] == "true"

    preflight = api.client.options(
        "/api/v1/generate",
        headers={**origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-Trace"},
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "https://lab.example.org"
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert preflight.headers["access-control-allow-headers"].lower() == "x-trace"


def test_health(api: Api) -> None:
    response = api.client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["status", "timestamp"]
    assert body["status"] == "healthy"
    datetime.fromisoformat(body["timestamp"])


def test_models_lists_the_catalog(api: Api) -> None:
    response = api.client.get("/api/v1/models")

    assert response.status_code == 200
    assert response.json() == {"models": list(MODEL_CATALOG)}


def test_status_checks_health_first(api: Api) -> None:
    response = api.client.get("/api/v1/status")

    assert response.status_code == 200
    assert response.json() == {"timestamp": "2026-05-01T12:00:00", "matrix": {}, "summary": {"total_endpoints": 9}}
    assert [name for name, _ in api.manager.calls] == ["check_all_health", "get_system_status"]
    assert api.created == []


# ---------------------------------------------------------------- /api/v1/generate


def test_generate_rejects_an_unknown_model(api: Api) -> None:
    response = api.client.post("/api/v1/generate", json={"prompt": "Hi", "model": "gpt"})

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Model 'gpt' not available. Available models: ['biogpt', 'chemberta', 'matscibert']"
    }
    assert api.manager.calls == []
    assert api.created == []


@pytest.mark.parametrize("field", ["model", "backend"])
def test_generate_rejects_an_explicit_null(api: Api, field: str) -> None:
    # The fields are 'str = Field(default=None)': they may be left out, but null is not a string.
    response = api.client.post("/api/v1/generate", json={"prompt": "Hi", field: None})

    assert response.status_code == 422
    assert [error["loc"] for error in response.json()["detail"]] == [["body", field]]
    assert api.manager.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": "Hi", "max_tokens": 0},
        {"prompt": "Hi", "max_tokens": 2001},
        {"prompt": "Hi", "temperature": 0.05},
        {"prompt": "Hi", "temperature": 2.5},
    ],
)
def test_generate_validates_the_request(api: Api, body: dict[str, Any]) -> None:
    assert api.client.post("/api/v1/generate", json=body).status_code == 422


def test_generate_with_a_model_calls_the_backend_manager(api: Api) -> None:
    request = {"prompt": "Hi", "model": "biogpt", "backend": "tgi", "max_tokens": 64, "temperature": 0.2}

    response = api.client.post("/api/v1/generate", json=request)

    assert response.status_code == 200
    assert api.manager.calls == [
        (
            "generate_completion",
            {"model": "biogpt", "prompt": "Hi", "backend": "tgi", "max_tokens": 64, "temperature": 0.2},
        )
    ]
    body = response.json()
    assert list(body) == ["text", "model", "backend", "endpoint", "domain", "response_time_ms", "selection_time_ms"]
    assert body["text"] == "{'generated_text': 'Hello'}"  # str() of the backend's answer, as before
    assert (body["model"], body["backend"], body["endpoint"]) == ("biogpt", "tgi", "biogpt-tgi-service:8001")
    assert body["domain"] == "unknown"
    assert body["selection_time_ms"] == 0
    assert body["response_time_ms"] >= 0
    assert api.created == []


def test_generate_with_a_model_and_the_defaults(api: Api) -> None:
    api.client.post("/api/v1/generate", json={"prompt": "Hi", "model": "chemberta"})

    assert api.manager.calls == [
        (
            "generate_completion",
            {"model": "chemberta", "prompt": "Hi", "backend": None, "max_tokens": 150, "temperature": 0.7},
        )
    ]


@pytest.mark.parametrize("model", [{}, {"model": ""}])
def test_generate_without_a_model_routes_through_the_orchestrator(api: Api, model: dict[str, str]) -> None:
    response = api.client.post("/api/v1/generate", json={"prompt": "acid", "optimize_for": "speed", **model})

    assert response.status_code == 200
    assert api.orchestrator.calls == [("process_request", "acid", "speed")]
    assert api.manager.calls == []
    body = response.json()
    assert body["text"] == "{'outputs': []}"
    assert (body["model"], body["backend"], body["domain"]) == ("chemberta", "tensorrt", "chemistry")
    assert body["endpoint"] == "chemberta-tensorrt-service:8005"
    assert body["selection_time_ms"] == 1.5


def test_generate_answers_an_orchestrator_error_with_placeholders(api: Api) -> None:
    api.orchestrator.result = {"error": "No suitable endpoint could be selected", "status": "failed"}

    response = api.client.post("/api/v1/generate", json={"prompt": "Hi"})

    # As before, the error result is not reported: the response is built from the defaults.
    assert response.status_code == 200
    body = response.json()
    assert body | {"response_time_ms": None} == {
        "text": "Generated response",
        "model": "auto-selected",
        "backend": "auto-selected",
        "endpoint": "auto-selected",
        "domain": "unknown",
        "response_time_ms": None,
        "selection_time_ms": 0.0,
    }


def test_generate_failure(api: Api) -> None:
    api.manager.error = RuntimeError("No healthy endpoints available for model: biogpt")

    response = api.client.post("/api/v1/generate", json={"prompt": "Hi", "model": "biogpt"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Generation failed: No healthy endpoints available for model: biogpt"}


# ---------------------------------------------------------------- the orchestrator routes


def test_intelligent_route_returns_the_result(api: Api) -> None:
    response = api.client.post("/api/v1/intelligent-route", json={"prompt": "acid", "optimize_for": "cost"})

    assert response.status_code == 200
    assert response.json() == api.orchestrator.result
    assert api.orchestrator.calls == [("process_request", "acid", "cost")]


def test_intelligent_route_wraps_an_error_result(api: Api) -> None:
    api.orchestrator.result = {"error": "Failed to deploy biogpt-vllm", "status": "deployment_failed"}

    response = api.client.post("/api/v1/intelligent-route", json={"prompt": "protein"})

    # The handler catches its own HTTPException, whose str() is '500: <error>', as before.
    assert response.status_code == 500
    assert response.json() == {"detail": "Routing failed: 500: Failed to deploy biogpt-vllm"}


def test_intelligent_route_failure(api: Api) -> None:
    api.orchestrator.error = RuntimeError("boom")

    response = api.client.post("/api/v1/intelligent-route", json={"prompt": "protein"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Routing failed: boom"}


def test_intelligent_route_validates_the_request(api: Api) -> None:
    assert api.client.post("/api/v1/intelligent-route", json={"optimize_for": "cost"}).status_code == 422
    assert api.orchestrator.calls == []


def test_orchestrator_status_and_matrix(api: Api) -> None:
    status = api.client.get("/api/v1/orchestrator/status")
    matrix = api.client.get("/api/v1/orchestrator/matrix")

    assert status.json() == api.orchestrator.get_deployment_status()
    assert matrix.json() == {"matrix": api.orchestrator.get_matrix_view()}


@pytest.mark.parametrize(("action", "done"), [("deploy", "Deployed"), ("undeploy", "Undeployed")])
def test_deploy_and_undeploy(api: Api, action: str, done: str) -> None:
    url = f"/api/v1/orchestrator/{action}"

    response = api.client.post(url, params={"model": "biogpt", "backend": "vllm"})
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": f"{done} biogpt-vllm"}
    assert api.orchestrator.calls == [(f"{action}_endpoint", "biogpt", "vllm")]

    api.orchestrator.succeed = False
    response = api.client.post(url, params={"model": "biogpt", "backend": "vllm"})
    assert response.status_code == 500
    assert response.json() == {"detail": f"Failed to {action} biogpt-vllm"}

    # Both are required query parameters, never a JSON body.
    assert api.client.post(url, params={"model": "biogpt"}).status_code == 422
    assert api.client.post(url, json={"model": "biogpt", "backend": "vllm"}).status_code == 422
    assert len(api.orchestrator.calls) == 2


# ---------------------------------------------------------------- lifecycle and wiring


def test_the_orchestrator_starts_lazily_inside_the_running_loop() -> None:
    orchestrator = FakeOrchestrator()
    created: list[FakeOrchestrator] = []

    def factory() -> FakeOrchestrator:
        created.append(orchestrator)
        return orchestrator

    # Creating the app happens outside any event loop and starts nothing. The legacy script created the
    # orchestrator and its task here and crashed with 'RuntimeError: no running event loop'.
    app = create_app(backend_manager=FakeManager(), orchestrator_factory=factory)
    assert created == []
    assert (app.state.orchestrator, app.state.auto_shutdown_task) == (None, None)

    with TestClient(app) as client:
        client.get("/health")
        client.get("/api/v1/status")
        client.post("/api/v1/generate", json={"prompt": "Hi", "model": "biogpt"})
        assert created == []

        assert client.get("/api/v1/orchestrator/matrix").status_code == 200
        client.get("/api/v1/orchestrator/status")
        client.post("/api/v1/intelligent-route", json={"prompt": "Hi"})
        client.post("/api/v1/generate", json={"prompt": "Hi"})

        assert created == [orchestrator]
        assert app.state.orchestrator is orchestrator
        assert orchestrator.monitor_started.wait(5)
        task = app.state.auto_shutdown_task
        assert not task.done()

    # Shutting the app down cancels the auto-shutdown task.
    assert orchestrator.monitor_cancelled.is_set()
    assert task.cancelled()


def test_default_wiring_follows_the_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chart_dir = str(tmp_path / "umbrella")
    os.makedirs(os.path.join(chart_dir, "charts", "biogpt-vllm"))
    helm_calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "text": True}
        helm_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="denied")  # a failure: no waits follow

    monkeypatch.setattr(subprocess, "run", fake_run)

    app = create_app(MatrixSettings(namespace="team-ns", chart_dir=chart_dir))
    with TestClient(app) as client:
        response = client.post("/api/v1/orchestrator/deploy", params={"model": "biogpt", "backend": "vllm"})
        status = client.get("/api/v1/orchestrator/status").json()
        system = client.get("/api/v1/status").json()

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to deploy biogpt-vllm"}
    assert status["namespace"] == "team-ns"
    assert system["summary"]["not_deployed"] == 9
    assert helm_calls == [
        [
            "helm",
            "upgrade",
            "--install",
            "multi-llm",
            chart_dir,
            "--namespace",
            "team-ns",
            "--create-namespace",
            "--set",
            "biogpt_vllm.enabled=true",
            "--wait",
            "--timeout",
            "5m",
        ]
    ]

    # /api/v1/generate and /api/v1/status have their own BackendManager in namespace 'default', as before.
    manager = app.state.backend_manager
    assert isinstance(manager, BackendManager)
    assert manager.namespace == "default"
    assert app.state.orchestrator.backend_manager is not manager
    assert app.state.orchestrator.backend_manager.namespace == "team-ns"

    assert asyncio.run(manager.deploy_endpoint(manager.endpoints["biogpt"][0])) is False
    assert helm_calls[-1] == [
        "helm",
        "upgrade",
        "--install",
        "biogpt-vllm",
        os.path.join(chart_dir, "charts", "biogpt-vllm"),
        "--namespace",
        "default",
        "--create-namespace",
        "--wait",
        "--timeout",
        "5m",
    ]


def test_create_app_needs_fastapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastapi", None)

    with pytest.raises(MissingDependencyError) as excinfo:
        create_app()

    assert (excinfo.value.package, excinfo.value.extra) == ("fastapi", "matrix")
    assert str(excinfo.value) == 'fastapi is required for this command: pip install -e ".[matrix]"'


def test_serve_runs_uvicorn_with_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    runs: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: runs.append((app, kwargs)))

    serve(MatrixSettings(namespace="ns", host="api.internal", port=9001))

    [(app, kwargs)] = runs
    assert isinstance(app, FastAPI)
    assert app.title == "3x3 Domain-Specific LLM Matrix API"
    assert kwargs == {"host": "api.internal", "port": 9001, "log_level": "info"}


def test_serve_needs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)

    with pytest.raises(MissingDependencyError) as excinfo:
        serve(MatrixSettings())

    assert str(excinfo.value) == 'uvicorn is required for this command: pip install -e ".[matrix]"'


# ---------------------------------------------------------------- aiohttp requests against a stub model server


class BackendStub(ThreadingHTTPServer):
    """A model server stand-in on 127.0.0.1 that records (method, path, headers, body) and answers `status`.

    With mode 'drop' it closes the connection without an answer.
    """

    daemon_threads = True

    def __init__(self) -> None:
        self.reset()
        super().__init__(("127.0.0.1", 0), _BackendStubHandler)

    def reset(self) -> None:
        self.requests: list[tuple[str, str, dict[str, str], bytes]] = []
        self.status = 200
        self.body = b'{"generated_text": "Hello"}'
        self.content_type = "application/json"
        self.mode = "answer"

    def server_bind(self) -> None:
        # HTTPServer.server_bind also does a reverse DNS lookup of the host name, which can take seconds.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    @property
    def port(self) -> int:
        return int(self.server_address[1])


class _BackendStubHandler(BaseHTTPRequestHandler):
    server: BackendStub

    def do_GET(self) -> None:
        self._answer(b"")

    def do_POST(self) -> None:
        self._answer(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def _answer(self, body: bytes) -> None:
        stub = self.server
        stub.requests.append((self.command, self.path, dict(self.headers.items()), body))
        if stub.mode == "drop":
            self.close_connection = True
            return
        self.send_response(stub.status)
        self.send_header("Content-Type", stub.content_type)
        self.send_header("Content-Length", str(len(stub.body)))
        self.end_headers()
        self.wfile.write(stub.body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the per-request log lines off stderr."""


@pytest.fixture(scope="module")
def backend_stub_server() -> Iterator[BackendStub]:
    pytest.importorskip("aiohttp")
    stub = BackendStub()
    thread = threading.Thread(target=stub.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        stub.shutdown()
        stub.server_close()
        thread.join()


@pytest.fixture
def backend_stub(backend_stub_server: BackendStub) -> BackendStub:
    """The running stub, with its answer reset and no recorded requests."""
    backend_stub_server.reset()
    return backend_stub_server


def stub_endpoint(stub: BackendStub, backend: str, *, deployed: bool = True) -> ModelEndpoint:
    endpoint = ModelEndpoint("chemberta", backend, "127.0.0.1", stub.port, DomainType.CHEMISTRY)
    if deployed:
        endpoint.deployment_status = BackendStatus.DEPLOYED
    return endpoint


@pytest.mark.parametrize(
    ("backend", "path"), [("vllm", "/health"), ("tgi", "/health"), ("tensorrt", "/v2/health/ready")]
)
def test_check_health_against_a_stub(backend_stub: BackendStub, backend: str, path: str) -> None:
    manager = BackendManager()
    endpoint = stub_endpoint(backend_stub, backend)

    assert asyncio.run(manager.check_health(endpoint)) is BackendStatus.HEALTHY
    assert endpoint.status is BackendStatus.HEALTHY
    assert [(method, request_path) for method, request_path, _, _ in backend_stub.requests] == [("GET", path)]

    backend_stub.status = 503
    assert asyncio.run(manager.check_health(endpoint)) is BackendStatus.UNHEALTHY
    assert endpoint.status is BackendStatus.UNHEALTHY


def test_check_health_when_the_server_hangs_up(backend_stub: BackendStub, matrix_log: pytest.LogCaptureFixture) -> None:
    backend_stub.mode = "drop"
    endpoint = stub_endpoint(backend_stub, "vllm")

    assert asyncio.run(BackendManager().check_health(endpoint)) is BackendStatus.UNHEALTHY
    [warning] = messages(matrix_log, logging.WARNING)
    assert warning.startswith("Health check failed for chemberta-vllm: ")


def test_check_all_health_against_a_stub(backend_stub: BackendStub) -> None:
    manager = BackendManager()
    deployed = ModelEndpoint(
        "biogpt", "tgi", "127.0.0.1", backend_stub.port, DomainType.BIOLOGY, deployment_status=BackendStatus.DEPLOYED
    )
    manager.endpoints["biogpt"][1] = deployed

    asyncio.run(manager.check_all_health())

    assert [(method, path) for method, path, _, _ in backend_stub.requests] == [("GET", "/health")]
    assert deployed.status is BackendStatus.HEALTHY
    others = [ep for eps in manager.endpoints.values() for ep in eps if ep is not deployed]
    assert all(ep.status is BackendStatus.NOT_DEPLOYED for ep in others)


@pytest.mark.parametrize(
    ("backend", "path", "body"),
    [
        (
            "vllm",
            "/v1/completions",
            b'{"model": "chemberta", "prompt": "Hi", "max_tokens": 64, "temperature": 0.2}',
        ),
        ("tgi", "/generate", b'{"inputs": "Hi", "parameters": {"max_new_tokens": 64, "temperature": 0.2}}'),
        (
            "tensorrt",
            "/v2/models/chemberta/infer",
            b'{"inputs": [{"name": "input_text", "shape": [1], "datatype": "BYTES", "data": ["Hi"]}], '
            b'"parameters": {"max_tokens": 64, "temperature": 0.2}}',
        ),
    ],
)
def test_generate_from_endpoint_posts_the_backend_payload(
    backend_stub: BackendStub, backend: str, path: str, body: bytes
) -> None:
    endpoint = stub_endpoint(backend_stub, backend, deployed=False)

    result = asyncio.run(BackendManager()._generate_from_endpoint(endpoint, "Hi", max_tokens=64, temperature=0.2))

    assert result == {
        "model": "chemberta",
        "backend": backend,
        "response": {"generated_text": "Hello"},
        "endpoint": f"127.0.0.1:{backend_stub.port}",
    }
    assert list(result) == ["model", "backend", "response", "endpoint"]
    [(method, request_path, headers, request_body)] = backend_stub.requests
    assert (method, request_path, request_body) == ("POST", path, body)
    assert headers["Content-Type"] == "application/json"
    assert endpoint.performance_metrics["success_rate"] == 100


def test_generate_from_endpoint_defaults(backend_stub: BackendStub) -> None:
    endpoint = stub_endpoint(backend_stub, "vllm")

    # Keyword arguments other than max_tokens and temperature are ignored.
    asyncio.run(BackendManager()._generate_from_endpoint(endpoint, "Hi", optimize_for="speed"))

    [(_, _, _, body)] = backend_stub.requests
    assert body == b'{"model": "chemberta", "prompt": "Hi", "max_tokens": 150, "temperature": 0.7}'


def test_generate_from_endpoint_failure_costs_success_rate(
    backend_stub: BackendStub, matrix_log: pytest.LogCaptureFixture
) -> None:
    backend_stub.status = 500
    backend_stub.body = b"model overloaded"
    backend_stub.content_type = "text/plain"
    manager = BackendManager()
    endpoint = stub_endpoint(backend_stub, "tgi")

    for expected_rate in (90, 80):
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(manager._generate_from_endpoint(endpoint, "Hi"))
        assert str(excinfo.value) == "Generation failed with status 500: model overloaded"
        assert endpoint.performance_metrics["success_rate"] == expected_rate

    endpoint.performance_metrics["success_rate"] = 5
    with pytest.raises(RuntimeError):
        asyncio.run(manager._generate_from_endpoint(endpoint, "Hi"))
    assert endpoint.performance_metrics["success_rate"] == 0
    assert messages(matrix_log, logging.ERROR)[0] == (
        "Generation failed for chemberta on tgi: Generation failed with status 500: model overloaded"
    )


def test_generate_through_the_api_reaches_the_model_server(backend_stub: BackendStub) -> None:
    # The whole path of POST /api/v1/generate with a model: healthy endpoint -> payload -> response text.
    manager = BackendManager()
    manager.endpoints["matscibert"][0] = ModelEndpoint(
        "matscibert",
        "vllm",
        "127.0.0.1",
        backend_stub.port,
        DomainType.MATERIALS,
        status=BackendStatus.HEALTHY,
        deployment_status=BackendStatus.DEPLOYED,
    )

    with TestClient(create_app(backend_manager=manager)) as client:
        response = client.post(
            "/api/v1/generate", json={"prompt": "Alloy?", "model": "matscibert", "max_tokens": 32, "temperature": 0.5}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "{'generated_text': 'Hello'}"
    assert (body["model"], body["backend"], body["endpoint"]) == (
        "matscibert",
        "vllm",
        f"127.0.0.1:{backend_stub.port}",
    )
    [(method, path, _, request_body)] = backend_stub.requests
    assert (method, path) == ("POST", "/v1/completions")
    assert request_body == b'{"model": "matscibert", "prompt": "Alloy?", "max_tokens": 32, "temperature": 0.5}'
