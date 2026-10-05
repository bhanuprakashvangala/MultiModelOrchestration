"""Orchestrator: umbrella-chart deployment, on-demand routing and idle shutdown for the 3x3 matrix.

- Umbrella-chart deploy and undeploy.
- process_request with on-demand deployment (it still classifies twice, as before).
- The idle auto-shutdown loop: 30 min idle, checked every 60 s.
- Deployment status and the text matrix view, in the legacy orders.

It owns its own BackendManager unless one is injected. mmorch.matrix.api creates one Orchestrator per app, on
first use.

Known prototype behaviour, kept on purpose (each is a separate follow-up): process_request selects the endpoint,
and so classifies the prompt, once itself and once more inside generate_completion_with_routing; a deployment's
usage_count goes up both when it is deployed and when the request is routed; the '--set <model>_<backend>.enabled'
keys do not match the umbrella chart's conditions; and helm runs through a blocking subprocess.run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

from mmorch.matrix import helm
from mmorch.matrix.backend_manager import BackendManager, Sleep
from mmorch.settings import DEFAULT_CHART_DIR

log = logging.getLogger(__name__)

DEPLOYMENT_ORDER: Final = (
    ("matscibert", "tgi"),
    ("matscibert", "vllm"),
    ("matscibert", "tensorrt"),
    ("biogpt", "tgi"),
    ("biogpt", "vllm"),
    ("biogpt", "tensorrt"),
    ("chemberta", "tgi"),
    ("chemberta", "vllm"),
    ("chemberta", "tensorrt"),
)
MATRIX_VIEW_MODELS: Final = ("matscibert", "biogpt", "chemberta")
MATRIX_VIEW_BACKENDS: Final = ("tgi", "vllm", "tensorrt")


@dataclass
class DeploymentConfig:
    """Configuration and usage of one model-backend deployment in the umbrella chart."""

    model_name: str
    backend: str
    helm_release: str
    namespace: str
    enabled: bool = False
    last_used: datetime | None = None
    usage_count: int = 0
    auto_shutdown_minutes: int = 30


class Orchestrator:
    """Deploys the matrix's endpoints on demand through the umbrella chart and shuts idle ones down."""

    namespace: str
    backend_manager: BackendManager
    deployment_configs: dict[str, DeploymentConfig]
    umbrella_chart_path: str
    auto_shutdown_enabled: bool
    monitoring_interval: int

    def __init__(
        self,
        namespace: str = "default",
        *,
        chart_dir: str = DEFAULT_CHART_DIR,
        backend_manager: BackendManager | None = None,
        run_helm: helm.HelmRunner = helm.run,
        sleep: Sleep = asyncio.sleep,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        """Set up the nine deployment configs in DEPLOYMENT_ORDER and a BackendManager unless one is given.

        The own BackendManager shares the namespace, chart directory, helm runner and sleep; now is the clock of
        the usage tracking, the idle shutdown and the status document.
        """
        self.namespace = namespace
        self.backend_manager = (
            backend_manager
            if backend_manager is not None
            else BackendManager(namespace, chart_dir=chart_dir, run_helm=run_helm, sleep=sleep)
        )
        self.deployment_configs = {
            f"{model}-{backend}": DeploymentConfig(
                model_name=model,
                backend=backend,
                helm_release=f"{model}-{backend}",
                namespace=self.namespace,
            )
            for model, backend in DEPLOYMENT_ORDER
        }
        self.umbrella_chart_path = chart_dir
        self.auto_shutdown_enabled = True
        self.monitoring_interval = 60  # seconds
        self._run_helm = run_helm
        self._sleep = sleep
        self._now = now

    async def deploy_endpoint(self, model: str, backend: str) -> bool:
        """Enable one model-backend pair in the umbrella chart; True if it is already enabled.

        A successful deploy marks the pair enabled, sets last_used and counts one use. False for an unknown pair
        or a failed helm run.
        """
        config_key = f"{model}-{backend}"

        if config_key not in self.deployment_configs:
            log.error("Unknown deployment configuration: %s", config_key)
            return False

        config = self.deployment_configs[config_key]

        if config.enabled:
            log.info("%s is already deployed", config_key)
            return True

        try:
            cmd = helm.umbrella_enable_cmd(self.umbrella_chart_path, self.namespace, config.helm_release)

            log.info("Deploying %s with helm...", config_key)
            result = self._run_helm(cmd)

            if result.returncode == 0:
                config.enabled = True
                config.last_used = self._now()
                config.usage_count += 1
                log.info("Successfully deployed %s", config_key)
                return True

            log.error("Failed to deploy %s: %s", config_key, result.stderr)
            return False

        except Exception as e:
            log.error("Error deploying %s: %s", config_key, e)
            return False

    async def undeploy_endpoint(self, model: str, backend: str) -> bool:
        """Disable one model-backend pair in the umbrella chart; True if it is already disabled."""
        config_key = f"{model}-{backend}"

        if config_key not in self.deployment_configs:
            log.error("Unknown deployment configuration: %s", config_key)
            return False

        config = self.deployment_configs[config_key]

        if not config.enabled:
            log.info("%s is already undeployed", config_key)
            return True

        try:
            cmd = helm.umbrella_disable_cmd(self.umbrella_chart_path, self.namespace, config.helm_release)

            log.info("Undeploying %s...", config_key)
            result = self._run_helm(cmd)

            if result.returncode == 0:
                config.enabled = False
                log.info("Successfully undeployed %s", config_key)
                return True

            log.error("Failed to undeploy %s: %s", config_key, result.stderr)
            return False

        except Exception as e:
            log.error("Error undeploying %s: %s", config_key, e)
            return False

    async def process_request(self, prompt: str, optimize_for: str = "balanced") -> dict[str, Any]:
        """Route a prompt, deploying its endpoint on demand, and return the completion or an error dict.

        An endpoint whose pair is not enabled yet is deployed through the umbrella chart and given 10 s. The
        completion gains orchestration_time_ms and deployment_config. Failures return {'error', 'status'} with
        the status 'failed', 'deployment_failed' or 'generation_failed' instead of raising.
        """
        start_time = time.time()

        endpoint = await self.backend_manager.select_optimal_endpoint(prompt, optimize_for)

        if endpoint is None:
            return {
                "error": "No suitable endpoint could be selected",
                "status": "failed",
            }

        config_key = f"{endpoint.model}-{endpoint.backend}"
        config = self.deployment_configs.get(config_key)

        if config is not None and not config.enabled:
            log.info("Deploying %s on-demand...", config_key)
            deployed = await self.deploy_endpoint(endpoint.model, endpoint.backend)

            if not deployed:
                return {
                    "error": f"Failed to deploy {config_key}",
                    "status": "deployment_failed",
                }

            # Wait for the deployment to be ready.
            await self._sleep(10)

        if config is not None:
            config.last_used = self._now()
            config.usage_count += 1

        try:
            result = await self.backend_manager.generate_completion_with_routing(
                prompt=prompt,
                optimize_for=optimize_for,
            )

            result["orchestration_time_ms"] = (time.time() - start_time) * 1000
            result["deployment_config"] = config_key if config is not None else "unknown"

            return result

        except Exception as e:
            log.error("Error processing request: %s", e)
            return {
                "error": str(e),
                "status": "generation_failed",
            }

    async def auto_shutdown_unused(self) -> None:
        """Undeploy deployments idle for longer than their auto-shutdown time, checking every monitoring interval.

        Runs until auto_shutdown_enabled is False. A deployment that was never used is never shut down. An error
        is logged and the loop goes on after the next interval.
        """
        while self.auto_shutdown_enabled:
            try:
                current_time = self._now()

                for config_key, config in self.deployment_configs.items():
                    if not config.enabled:
                        continue

                    if config.last_used is not None:
                        time_since_use = current_time - config.last_used

                        if time_since_use > timedelta(minutes=config.auto_shutdown_minutes):
                            log.info("Auto-shutting down %s (unused for %s)", config_key, time_since_use)

                            model, backend = config_key.split("-")
                            await self.undeploy_endpoint(model, backend)

                await self._sleep(self.monitoring_interval)

            except Exception as e:
                log.error("Error in auto-shutdown monitoring: %s", e)
                await self._sleep(self.monitoring_interval)

    def get_deployment_status(self) -> dict[str, Any]:
        """Return every deployment's state and the deployed / idle / usage totals.

        The document is {timestamp, namespace, deployments: {key: {...}}, summary}, in DEPLOYMENT_ORDER. An enabled
        deployment counts as idle when it was last used more than 5 minutes ago.
        """
        deployments: dict[str, Any] = {}
        summary = {
            "total": len(self.deployment_configs),
            "deployed": 0,
            "idle": 0,
            "total_usage": 0,
        }
        status: dict[str, Any] = {
            "timestamp": self._now().isoformat(),
            "namespace": self.namespace,
            "deployments": deployments,
            "summary": summary,
        }

        current_time = self._now()

        for config_key, config in self.deployment_configs.items():
            time_since_use: float | None = None
            if config.last_used is not None:
                time_since_use = (current_time - config.last_used).total_seconds()

            deployments[config_key] = {
                "model": config.model_name,
                "backend": config.backend,
                "enabled": config.enabled,
                "usage_count": config.usage_count,
                "last_used": config.last_used.isoformat() if config.last_used is not None else None,
                "time_since_use_seconds": time_since_use,
                "auto_shutdown_minutes": config.auto_shutdown_minutes,
            }

            if config.enabled:
                summary["deployed"] += 1

                if time_since_use and time_since_use > 300:  # 5 minutes
                    summary["idle"] += 1

            summary["total_usage"] += config.usage_count

        return status

    def get_matrix_view(self) -> str:
        """Return the text view of the matrix: one row per model, one column per backend.

        Each cell is DEPLOYED, OFF, or '? UNKNOWN' for a pair without a deployment config; the last line counts
        the enabled deployments and the GPUs they use (one each).
        """
        # Header
        matrix = "3x3 LLM Matrix Status\n"
        matrix += "=" * 60 + "\n"
        matrix += f"{'Model':<15} | {'TGI':<12} | {'vLLM':<12} | {'TensorRT':<12}\n"
        matrix += "-" * 60 + "\n"

        # Rows
        for model in MATRIX_VIEW_MODELS:
            row = f"{model:<15} |"
            for backend in MATRIX_VIEW_BACKENDS:
                config_key = f"{model}-{backend}"
                config = self.deployment_configs.get(config_key)

                if config is None:
                    status = "? UNKNOWN"
                elif config.enabled:
                    status = "DEPLOYED"
                else:
                    status = "OFF"

                row += f" {status:<10} |"

            matrix += row + "\n"

        matrix += "=" * 60 + "\n"

        # Summary
        deployed = sum(1 for c in self.deployment_configs.values() if c.enabled)
        total = len(self.deployment_configs)
        matrix += f"Deployed: {deployed}/{total} | "
        matrix += f"Resource Usage: {deployed * 1} GPU(s)\n"

        return matrix
