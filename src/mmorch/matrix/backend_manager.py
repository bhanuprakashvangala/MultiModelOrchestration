"""BackendManager: on-demand deployment, health checks and routing across the 3x3 model x backend matrix.

Ported method for method. Method names, log texts, the 10 s and 5 s sleeps, the 5 s health timeout, the result
dicts and the DEPLOYING-after-missing-chart quirk are unchanged. aiohttp is imported only inside check_health
and _generate_from_endpoint. The helm runner, the sleep and the domain classifier are constructor parameters
with production defaults.

Known prototype behaviour, kept on purpose (each is a separate follow-up):

- a missing subchart leaves the endpoint DEPLOYING, so later deploys report success without running helm;
- the registry names biogpt, chemberta and matscibert subcharts, which deploy/helm does not contain;
- helm runs through a blocking subprocess.run inside the async methods.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, TypeAlias

from mmorch.errors import MissingDependencyError
from mmorch.matrix import helm
from mmorch.matrix.domain import classify_domain
from mmorch.matrix.endpoints import (
    HEALTHY_PRIORITY,
    MODEL_FOR_DOMAIN,
    BackendStatus,
    DomainType,
    ModelEndpoint,
    build_payload,
    choose_endpoint,
    default_endpoints,
)
from mmorch.settings import DEFAULT_CHART_DIR

log = logging.getLogger(__name__)

Sleep: TypeAlias = Callable[[float], Awaitable[None]]


class BackendManager:
    """Deploys, health-checks and routes to the nine endpoints of one Kubernetes namespace."""

    namespace: str
    endpoints: dict[str, list[ModelEndpoint]]
    health_check_interval: int
    last_health_check: dict[str, Any]

    def __init__(
        self,
        kubernetes_namespace: str = "default",
        *,
        chart_dir: str = DEFAULT_CHART_DIR,
        run_helm: helm.HelmRunner = helm.run,
        sleep: Sleep = asyncio.sleep,
        classify: Callable[[str], tuple[DomainType, float]] = classify_domain,
    ) -> None:
        """Set up the nine endpoints of the namespace; helm, sleep and domain classification are injectable.

        chart_dir is the umbrella chart directory, whose charts/ subdirectory holds one subchart per release.
        """
        self.namespace = kubernetes_namespace
        self.endpoints = default_endpoints(self.namespace)
        self.health_check_interval = 30  # seconds; kept from the prototype, which never reads it
        self.last_health_check = {}
        self._chart_dir = chart_dir
        self._run_helm = run_helm
        self._sleep = sleep
        self._classify = classify

    def classify_domain(self, prompt: str) -> tuple[DomainType, float]:
        """Classify the domain of a prompt with the injected classifier."""
        return self._classify(prompt)

    async def deploy_endpoint(self, endpoint: ModelEndpoint) -> bool:
        """Install the endpoint's Helm release from its subchart and wait for it; True if already deployed.

        An endpoint that is DEPLOYED or DEPLOYING counts as deployed. When the subchart is missing this returns
        False but leaves the endpoint DEPLOYING, so the next call returns True without running helm (a known
        prototype bug, kept). A successful install is followed by a 10 s wait for the pod.
        """
        if endpoint.deployment_status in [BackendStatus.DEPLOYED, BackendStatus.DEPLOYING]:
            return True

        try:
            endpoint.deployment_status = BackendStatus.DEPLOYING
            log.info("Deploying %s in namespace %s", endpoint.helm_release, endpoint.namespace)

            chart_path = helm.subchart_path(self._chart_dir, endpoint.helm_release)
            if not os.path.exists(chart_path):
                log.error("Helm chart not found at %s", chart_path)
                return False

            cmd = helm.install_cmd(endpoint.helm_release, chart_path, endpoint.namespace)
            result = self._run_helm(cmd)

            if result.returncode == 0:
                endpoint.deployment_status = BackendStatus.DEPLOYED
                log.info("Successfully deployed %s", endpoint.helm_release)

                # Wait for the pod to be ready.
                await self._sleep(10)
                return True

            log.error("Failed to deploy %s: %s", endpoint.helm_release, result.stderr)
            endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
            return False

        except Exception as e:
            log.error("Error deploying %s: %s", endpoint.helm_release, e)
            endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
            return False

    async def undeploy_endpoint(self, endpoint: ModelEndpoint) -> bool:
        """Uninstall the endpoint's Helm release; True if it was not deployed."""
        if endpoint.deployment_status == BackendStatus.NOT_DEPLOYED:
            return True

        try:
            cmd = helm.uninstall_cmd(endpoint.helm_release, endpoint.namespace)
            result = self._run_helm(cmd)

            if result.returncode == 0:
                endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
                endpoint.status = BackendStatus.NOT_DEPLOYED
                log.info("Successfully undeployed %s", endpoint.helm_release)
                return True

            log.error("Failed to undeploy %s: %s", endpoint.helm_release, result.stderr)
            return False

        except Exception as e:
            log.error("Error undeploying %s: %s", endpoint.helm_release, e)
            return False

    async def check_health(self, endpoint: ModelEndpoint) -> BackendStatus:
        """Probe a deployed endpoint's health URL with a 5 s timeout and record the result.

        An endpoint that is not DEPLOYED is NOT_DEPLOYED without a request. Otherwise HTTP 200 is HEALTHY and
        any other status or error UNHEALTHY.
        """
        if endpoint.deployment_status != BackendStatus.DEPLOYED:
            endpoint.status = BackendStatus.NOT_DEPLOYED
            return BackendStatus.NOT_DEPLOYED

        try:
            import aiohttp
        except ImportError as exc:
            raise MissingDependencyError("aiohttp", "matrix") from exc

        try:
            async with (
                aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session,
                session.get(endpoint.health_check_url) as response,
            ):
                if response.status == 200:
                    endpoint.status = BackendStatus.HEALTHY
                    return BackendStatus.HEALTHY
        except Exception as e:
            log.warning("Health check failed for %s-%s: %s", endpoint.model, endpoint.backend, e)

        endpoint.status = BackendStatus.UNHEALTHY
        return BackendStatus.UNHEALTHY

    async def check_all_health(self) -> None:
        """Check every endpoint concurrently; a failing check does not stop the others."""
        tasks = [self.check_health(endpoint) for endpoints in self.endpoints.values() for endpoint in endpoints]
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_healthy_endpoints(self, model: str) -> list[ModelEndpoint]:
        """Return the model's endpoints that are currently healthy."""
        if model not in self.endpoints:
            return []

        return [ep for ep in self.endpoints[model] if ep.status == BackendStatus.HEALTHY]

    def get_best_endpoint(self, model: str, preferred_backend: str | None = None) -> ModelEndpoint | None:
        """Return the preferred backend if healthy, else the healthy endpoint with the best backend priority."""
        healthy_endpoints = self.get_healthy_endpoints(model)

        if not healthy_endpoints:
            return None

        if preferred_backend:
            for ep in healthy_endpoints:
                if ep.backend == preferred_backend:
                    return ep

        # Otherwise the first healthy endpoint in the order vLLM, TGI, TensorRT.
        return min(healthy_endpoints, key=lambda ep: HEALTHY_PRIORITY.get(ep.backend, 99))

    async def select_optimal_endpoint(self, prompt: str, optimize_for: str = "balanced") -> ModelEndpoint | None:
        """Classify the prompt, pick the model and backend, and deploy and health-check the endpoint if needed.

        optimize_for is 'speed', 'cost', 'quality' or 'balanced' (anything else counts as 'balanced'). A newly
        deployed endpoint gets 5 s before its health check; the endpoint is returned whatever that check says.
        None when the deployment fails.
        """
        domain, confidence = self.classify_domain(prompt)
        log.info("Classified prompt as %s with confidence %.2f", domain.value, confidence)

        model = MODEL_FOR_DOMAIN.get(domain, "biogpt")
        best_endpoint = choose_endpoint(self.endpoints[model], optimize_for)

        if best_endpoint is not None and best_endpoint.deployment_status != BackendStatus.DEPLOYED:
            log.info("Deploying optimal endpoint: %s", best_endpoint.helm_release)
            deployed = await self.deploy_endpoint(best_endpoint)
            if not deployed:
                log.error("Failed to deploy %s", best_endpoint.helm_release)
                return None

            # Wait before the health check.
            await self._sleep(5)
            await self.check_health(best_endpoint)

        return best_endpoint

    async def generate_completion_with_routing(
        self,
        prompt: str,
        optimize_for: str = "balanced",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a completion on the optimal endpoint and add the selection and total times and the domain.

        The total time also becomes the endpoint's latency_ms metric. RuntimeError when no endpoint could be
        deployed.
        """
        start_time = time.time()

        endpoint = await self.select_optimal_endpoint(prompt, optimize_for)

        if endpoint is None:
            raise RuntimeError("No suitable endpoint could be deployed")

        selection_time = (time.time() - start_time) * 1000

        result = await self._generate_from_endpoint(endpoint, prompt, **kwargs)

        total_time = (time.time() - start_time) * 1000
        endpoint.performance_metrics["latency_ms"] = total_time

        result["selection_time_ms"] = selection_time
        result["total_time_ms"] = total_time
        result["domain"] = endpoint.domain.value

        return result

    async def generate_completion(
        self,
        model: str,
        prompt: str,
        backend: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a completion with a given model (and backend), deploying an endpoint if none is healthy.

        Without a healthy endpoint, the model's endpoints (only the given backend, if any) are deployed in turn,
        each followed by 5 s and a health check, until one is healthy. RuntimeError when none is.
        """
        endpoint = self.get_best_endpoint(model, backend)

        if endpoint is None:
            for ep in self.endpoints.get(model, []):
                if backend and ep.backend != backend:
                    continue
                if await self.deploy_endpoint(ep):
                    await self._sleep(5)
                    if await self.check_health(ep) == BackendStatus.HEALTHY:
                        endpoint = ep
                        break

            if endpoint is None:
                raise RuntimeError(f"No healthy endpoints available for model: {model}")

        return await self._generate_from_endpoint(endpoint, prompt, **kwargs)

    async def _generate_from_endpoint(self, endpoint: ModelEndpoint, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """POST the backend's payload to the endpoint and return the response with the model and backend.

        max_tokens (default 150) and temperature (default 0.7) come from kwargs. Any failure, including a status
        other than 200 (RuntimeError), costs the endpoint 10 points of success_rate (not below 0) and is re-raised.
        """
        try:
            import aiohttp
        except ImportError as exc:
            raise MissingDependencyError("aiohttp", "matrix") from exc

        try:
            # Inside the try, as before: an unknown backend fails here and counts as a failed generation.
            payload = build_payload(
                endpoint,
                prompt,
                max_tokens=kwargs.get("max_tokens", 150),
                temperature=kwargs.get("temperature", 0.7),
            )

            async with (
                aiohttp.ClientSession() as session,
                session.post(endpoint.generate_url, json=payload) as response,
            ):
                if response.status == 200:
                    result = await response.json()
                    return {
                        "model": endpoint.model,
                        "backend": endpoint.backend,
                        "response": result,
                        "endpoint": endpoint.address,
                    }

                error_text = await response.text()
                raise RuntimeError(f"Generation failed with status {response.status}: {error_text}")

        except Exception as e:
            log.error("Generation failed for %s on %s: %s", endpoint.model, endpoint.backend, e)
            endpoint.performance_metrics["success_rate"] = max(0, endpoint.performance_metrics["success_rate"] - 10)
            raise

    def get_system_status(self) -> dict[str, Any]:
        """Return the status of every endpoint and the deployed / healthy / unhealthy / not-deployed counts.

        The document is {timestamp, matrix: {model: {domain, backends: {backend: {status, deployment_status,
        endpoint, performance}}}}, summary}, in registry order. A deployed endpoint that is not HEALTHY counts as
        unhealthy.
        """
        matrix: dict[str, Any] = {}
        summary = {
            "total_endpoints": 9,
            "deployed": 0,
            "healthy": 0,
            "unhealthy": 0,
            "not_deployed": 0,
        }
        status: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "matrix": matrix,
            "summary": summary,
        }

        for model, endpoints in self.endpoints.items():
            backends: dict[str, Any] = {}
            model_status = {
                "domain": endpoints[0].domain.value if endpoints else "unknown",
                "backends": backends,
            }

            for endpoint in endpoints:
                backends[endpoint.backend] = {
                    "status": endpoint.status.value,
                    "deployment_status": endpoint.deployment_status.value,
                    "endpoint": endpoint.address,
                    "performance": endpoint.performance_metrics,
                }

                if endpoint.deployment_status == BackendStatus.DEPLOYED:
                    summary["deployed"] += 1
                    if endpoint.status == BackendStatus.HEALTHY:
                        summary["healthy"] += 1
                    else:
                        summary["unhealthy"] += 1
                else:
                    summary["not_deployed"] += 1

            matrix[model] = model_status

        return status
