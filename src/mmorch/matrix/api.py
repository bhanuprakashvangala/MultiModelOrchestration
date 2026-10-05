"""The FastAPI interface of the prototype, behind `mmorch serve`.

The title, description, version '2.0.0', the /api/docs and /api/redoc pages, the permissive CORS, the request
and response models, the routes, the status codes and the error details are as before. This module is imported
only by `mmorch serve` and by `uvicorn --factory mmorch.matrix.api:create_app`. Only `mmorch serve` reads the
environment; the factory, called with no arguments, uses the MatrixSettings() defaults.

State lives on the app. One BackendManager in namespace 'default' serves /api/v1/generate and /api/v1/status,
as the legacy global did. The Orchestrator is created lazily on first use, inside the running event loop, and
then starts its auto-shutdown task; the task is kept on app.state and cancelled at shutdown.

Kept from the prototype: the route functions' names and docstrings, which FastAPI publishes as the operation ids,
summaries and descriptions of the OpenAPI document; the routes without a response model; and the
intelligent-route handler wrapping its own HTTPException, so an error result answers 500 with
'Routing failed: 500: <error>'.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from mmorch.errors import MissingDependencyError
from mmorch.matrix.backend_manager import BackendManager
from mmorch.matrix.endpoints import MODEL_CATALOG
from mmorch.matrix.orchestrator import Orchestrator
from mmorch.settings import MatrixSettings

if TYPE_CHECKING:
    from fastapi import FastAPI

# The request and response models below are defined at import time, so pydantic is needed here already.
try:
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise MissingDependencyError("pydantic", "matrix") from exc

log = logging.getLogger(__name__)

TITLE: Final = "3x3 Domain-Specific LLM Matrix API"
DESCRIPTION: Final = (
    "Intelligent routing system for domain-specific LLMs (BioGPT, ChemBERTa, MatSciBERT) x (vLLM, TGI, TensorRT)"
)
VERSION: Final = "2.0.0"

# The request and response models have no docstrings on purpose: pydantic would publish them as schema
# descriptions and change the OpenAPI document behind /api/docs.


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    # The None defaults on str fields are verbatim: an explicit JSON null is still rejected with 422.
    model: str = Field(  # type: ignore[assignment]
        default=None,
        description="Model to use: biogpt, chemberta, matscibert (optional - will auto-select based on domain)",
    )
    backend: str = Field(  # type: ignore[assignment]
        default=None,
        description="Backend preference: vllm, tgi, tensorrt (optional)",
    )
    max_tokens: int = Field(default=150, ge=1, le=2000)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    optimize_for: str = Field(default="balanced", description="Optimization preference: speed, cost, quality, balanced")


class GenerateResponse(BaseModel):
    text: str
    model: str
    backend: str
    endpoint: str
    domain: str
    response_time_ms: float
    selection_time_ms: float


class IntelligentRoutingRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for intelligent routing")
    optimize_for: str = Field(default="balanced", description="Optimization: speed, cost, quality, balanced")
    max_tokens: int = Field(default=150, ge=1, le=2000)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


def create_app(
    settings: MatrixSettings | None = None,
    *,
    backend_manager: BackendManager | None = None,
    orchestrator_factory: Callable[[], Orchestrator] | None = None,
) -> FastAPI:
    """Build the API app with its routes, CORS and lazily created orchestrator.

    settings defaults to MatrixSettings(). backend_manager serves /api/v1/generate and /api/v1/status; it
    defaults to BackendManager('default', chart_dir=settings.chart_dir), in namespace 'default' whatever
    settings.namespace says, as before. orchestrator_factory builds the orchestrator on first use; it defaults to
    Orchestrator(settings.namespace, chart_dir=settings.chart_dir). Creating the app starts nothing.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise MissingDependencyError("fastapi", "matrix") from exc

    config = settings if settings is not None else MatrixSettings()
    manager = backend_manager if backend_manager is not None else BackendManager("default", chart_dir=config.chart_dir)
    make_orchestrator: Callable[[], Orchestrator] = (
        orchestrator_factory
        if orchestrator_factory is not None
        else functools.partial(Orchestrator, config.namespace, chart_dir=config.chart_dir)
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Serve the app; at shutdown, cancel the orchestrator's auto-shutdown task if it was started."""
        try:
            yield
        finally:
            task: asyncio.Task[None] | None = app.state.auto_shutdown_task
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title=TITLE,
        description=DESCRIPTION,
        version=VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.backend_manager = manager
    app.state.orchestrator = None
    app.state.auto_shutdown_task = None

    def get_orchestrator() -> Orchestrator:
        """Return the app's orchestrator, creating it and starting its auto-shutdown task on first use."""
        orchestrator: Orchestrator | None = app.state.orchestrator
        if orchestrator is None:
            orchestrator = make_orchestrator()
            app.state.orchestrator = orchestrator
            # Called from a request handler, so the event loop is running.
            app.state.auto_shutdown_task = asyncio.create_task(orchestrator.auto_shutdown_unused())
        return orchestrator

    # The route functions keep their legacy names and docstrings: FastAPI publishes them in the OpenAPI document.

    @app.post("/api/v1/generate", response_model=GenerateResponse)
    async def generate_text(request: GenerateRequest) -> GenerateResponse:
        """Generate text using the specified model and backend"""
        if request.model and request.model not in manager.endpoints:
            available_models = list(manager.endpoints.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not available. Available models: {available_models}",
            )

        start_time = datetime.now()

        try:
            # Without a model, the orchestrator routes the prompt.
            if not request.model:
                orchestrator = get_orchestrator()
                result = await orchestrator.process_request(
                    prompt=request.prompt,
                    optimize_for=request.optimize_for,
                )
            else:
                result = await manager.generate_completion(
                    model=request.model,
                    prompt=request.prompt,
                    backend=request.backend,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            return GenerateResponse(
                text=str(result.get("response", "Generated response")),
                model=result.get("model", "auto-selected"),
                backend=result.get("backend", "auto-selected"),
                endpoint=result.get("endpoint", "auto-selected"),
                domain=result.get("domain", "unknown"),
                response_time_ms=response_time,
                selection_time_ms=result.get("selection_time_ms", 0),
            )

        except Exception as e:
            log.error("Generation failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}") from e

    @app.get("/api/v1/status", response_model=None)
    async def get_system_status() -> dict[str, Any]:
        """Get system status for all models and backends"""
        await manager.check_all_health()
        return manager.get_system_status()

    @app.get("/api/v1/models", response_model=None)
    async def list_models() -> dict[str, Any]:
        """List available models"""
        return {"models": list(MODEL_CATALOG)}

    @app.post("/api/v1/intelligent-route", response_model=None)
    async def intelligent_route(request: IntelligentRoutingRequest) -> dict[str, Any]:
        """Use intelligent routing to automatically select the best model and backend"""
        orchestrator = get_orchestrator()

        try:
            result = await orchestrator.process_request(
                prompt=request.prompt,
                optimize_for=request.optimize_for,
            )

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            return result

        except Exception as e:
            # This also catches the HTTPException above, whose str() is '500: <error>'.
            log.error("Intelligent routing failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Routing failed: {e}") from e

    @app.get("/api/v1/orchestrator/status", response_model=None)
    async def get_orchestrator_status() -> dict[str, Any]:
        """Get orchestrator deployment status"""
        orchestrator = get_orchestrator()
        return orchestrator.get_deployment_status()

    @app.get("/api/v1/orchestrator/matrix", response_model=None)
    async def get_matrix_view() -> dict[str, str]:
        """Get visual matrix representation"""
        orchestrator = get_orchestrator()
        return {"matrix": orchestrator.get_matrix_view()}

    @app.post("/api/v1/orchestrator/deploy", response_model=None)
    async def deploy_endpoint(model: str, backend: str) -> dict[str, str]:
        """Manually deploy a specific endpoint"""
        orchestrator = get_orchestrator()

        success = await orchestrator.deploy_endpoint(model, backend)

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to deploy {model}-{backend}")
        return {"status": "success", "message": f"Deployed {model}-{backend}"}

    @app.post("/api/v1/orchestrator/undeploy", response_model=None)
    async def undeploy_endpoint(model: str, backend: str) -> dict[str, str]:
        """Manually undeploy a specific endpoint"""
        orchestrator = get_orchestrator()

        success = await orchestrator.undeploy_endpoint(model, backend)

        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to undeploy {model}-{backend}")
        return {"status": "success", "message": f"Undeployed {model}-{backend}"}

    @app.get("/health", response_model=None)
    async def health_check() -> dict[str, str]:
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    return app


def serve(settings: MatrixSettings) -> None:
    """Run the API with uvicorn at the configured host and port, logging at 'info'."""
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingDependencyError("uvicorn", "matrix") from exc

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
