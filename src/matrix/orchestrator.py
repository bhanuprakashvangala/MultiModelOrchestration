"""
Intelligent Orchestrator for 3x3 LLM Matrix
Manages dynamic deployment and routing for domain-specific LLMs
"""

import asyncio
import os
import logging
import subprocess
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from backend_manager import BackendManager, DomainType, BackendStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for a model deployment"""
    model_name: str
    backend: str
    helm_release: str
    namespace: str
    enabled: bool = False
    last_used: Optional[datetime] = None
    usage_count: int = 0
    auto_shutdown_minutes: int = 30

class Orchestrator:
    """
    Main orchestrator for managing the 3x3 LLM matrix
    Handles on-demand deployment, auto-scaling, and intelligent routing
    """
    
    def __init__(self, namespace: str = os.environ.get("KUBERNETES_NAMESPACE", "default")):
        self.namespace = namespace
        self.backend_manager = BackendManager(namespace)
        self.deployment_configs = self._initialize_deployments()
        self.umbrella_chart_path = os.environ.get("MATRIX_CHART_DIR", "./deploy/helm/pick-and-spin-umbrella")
        self.auto_shutdown_enabled = True
        self.monitoring_interval = 60  # seconds
        
    def _initialize_deployments(self) -> Dict[str, DeploymentConfig]:
        """Initialize deployment configurations for all 9 endpoints"""
        configs = {}
        
        # MatSciBERT configurations
        configs["matscibert-tgi"] = DeploymentConfig(
            model_name="matscibert",
            backend="tgi",
            helm_release="matscibert-tgi",
            namespace=self.namespace
        )
        configs["matscibert-vllm"] = DeploymentConfig(
            model_name="matscibert",
            backend="vllm",
            helm_release="matscibert-vllm",
            namespace=self.namespace
        )
        configs["matscibert-tensorrt"] = DeploymentConfig(
            model_name="matscibert",
            backend="tensorrt",
            helm_release="matscibert-tensorrt",
            namespace=self.namespace
        )
        
        # BioGPT configurations
        configs["biogpt-tgi"] = DeploymentConfig(
            model_name="biogpt",
            backend="tgi",
            helm_release="biogpt-tgi",
            namespace=self.namespace
        )
        configs["biogpt-vllm"] = DeploymentConfig(
            model_name="biogpt",
            backend="vllm",
            helm_release="biogpt-vllm",
            namespace=self.namespace
        )
        configs["biogpt-tensorrt"] = DeploymentConfig(
            model_name="biogpt",
            backend="tensorrt",
            helm_release="biogpt-tensorrt",
            namespace=self.namespace
        )
        
        # ChemBERTa configurations
        configs["chemberta-tgi"] = DeploymentConfig(
            model_name="chemberta",
            backend="tgi",
            helm_release="chemberta-tgi",
            namespace=self.namespace
        )
        configs["chemberta-vllm"] = DeploymentConfig(
            model_name="chemberta",
            backend="vllm",
            helm_release="chemberta-vllm",
            namespace=self.namespace
        )
        configs["chemberta-tensorrt"] = DeploymentConfig(
            model_name="chemberta",
            backend="tensorrt",
            helm_release="chemberta-tensorrt",
            namespace=self.namespace
        )
        
        return configs
    
    async def deploy_endpoint(self, model: str, backend: str) -> bool:
        """Deploy a specific model-backend combination"""
        config_key = f"{model}-{backend}"
        
        if config_key not in self.deployment_configs:
            logger.error(f"Unknown deployment configuration: {config_key}")
            return False
        
        config = self.deployment_configs[config_key]
        
        if config.enabled:
            logger.info(f"{config_key} is already deployed")
            return True
        
        try:
            # Update values.yaml to enable this specific deployment
            values_update = {
                config.helm_release.replace("-", "_"): {"enabled": True}
            }
            
            # Deploy using helm with updated values
            cmd = [
                "helm", "upgrade", "--install",
                "multi-llm",
                self.umbrella_chart_path,
                "--namespace", self.namespace,
                "--create-namespace",
                "--set", f"{config.helm_release.replace('-', '_')}.enabled=true",
                "--wait", "--timeout", "5m"
            ]
            
            logger.info(f"Deploying {config_key} with helm...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                config.enabled = True
                config.last_used = datetime.now()
                config.usage_count += 1
                logger.info(f"Successfully deployed {config_key}")
                return True
            else:
                logger.error(f"Failed to deploy {config_key}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying {config_key}: {e}")
            return False
    
    async def undeploy_endpoint(self, model: str, backend: str) -> bool:
        """Undeploy a specific model-backend combination to save resources"""
        config_key = f"{model}-{backend}"
        
        if config_key not in self.deployment_configs:
            logger.error(f"Unknown deployment configuration: {config_key}")
            return False
        
        config = self.deployment_configs[config_key]
        
        if not config.enabled:
            logger.info(f"{config_key} is already undeployed")
            return True
        
        try:
            # Update values to disable this deployment
            cmd = [
                "helm", "upgrade",
                "multi-llm",
                self.umbrella_chart_path,
                "--namespace", self.namespace,
                "--set", f"{config.helm_release.replace('-', '_')}.enabled=false",
                "--wait", "--timeout", "2m"
            ]
            
            logger.info(f"Undeploying {config_key}...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                config.enabled = False
                logger.info(f"Successfully undeployed {config_key}")
                return True
            else:
                logger.error(f"Failed to undeploy {config_key}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error undeploying {config_key}: {e}")
            return False
    
    async def process_request(self, prompt: str, optimize_for: str = "balanced") -> Dict[str, Any]:
        """
        Process a request with intelligent routing and on-demand deployment
        """
        start_time = time.time()
        
        # Use backend manager to select optimal endpoint
        endpoint = await self.backend_manager.select_optimal_endpoint(prompt, optimize_for)
        
        if not endpoint:
            return {
                "error": "No suitable endpoint could be selected",
                "status": "failed"
            }
        
        # Deploy if needed
        config_key = f"{endpoint.model}-{endpoint.backend}"
        config = self.deployment_configs.get(config_key)
        
        if config and not config.enabled:
            logger.info(f"Deploying {config_key} on-demand...")
            deployed = await self.deploy_endpoint(endpoint.model, endpoint.backend)
            
            if not deployed:
                return {
                    "error": f"Failed to deploy {config_key}",
                    "status": "deployment_failed"
                }
            
            # Wait for deployment to be ready
            await asyncio.sleep(10)
        
        # Update usage tracking
        if config:
            config.last_used = datetime.now()
            config.usage_count += 1
        
        # Generate completion
        try:
            result = await self.backend_manager.generate_completion_with_routing(
                prompt=prompt,
                optimize_for=optimize_for
            )
            
            result["orchestration_time_ms"] = (time.time() - start_time) * 1000
            result["deployment_config"] = config_key if config else "unknown"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "error": str(e),
                "status": "generation_failed"
            }
    
    async def auto_shutdown_unused(self):
        """Automatically shutdown unused deployments to save resources"""
        while self.auto_shutdown_enabled:
            try:
                current_time = datetime.now()
                
                for config_key, config in self.deployment_configs.items():
                    if not config.enabled:
                        continue
                    
                    # Check if deployment has been unused for shutdown period
                    if config.last_used:
                        time_since_use = current_time - config.last_used
                        
                        if time_since_use > timedelta(minutes=config.auto_shutdown_minutes):
                            logger.info(f"Auto-shutting down {config_key} (unused for {time_since_use})")
                            
                            model, backend = config_key.split("-")
                            await self.undeploy_endpoint(model, backend)
                
                # Wait before next check
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-shutdown monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current status of all deployments"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "namespace": self.namespace,
            "deployments": {},
            "summary": {
                "total": len(self.deployment_configs),
                "deployed": 0,
                "idle": 0,
                "total_usage": 0
            }
        }
        
        current_time = datetime.now()
        
        for config_key, config in self.deployment_configs.items():
            time_since_use = None
            if config.last_used:
                time_since_use = (current_time - config.last_used).total_seconds()
            
            status["deployments"][config_key] = {
                "model": config.model_name,
                "backend": config.backend,
                "enabled": config.enabled,
                "usage_count": config.usage_count,
                "last_used": config.last_used.isoformat() if config.last_used else None,
                "time_since_use_seconds": time_since_use,
                "auto_shutdown_minutes": config.auto_shutdown_minutes
            }
            
            if config.enabled:
                status["summary"]["deployed"] += 1
                
                if time_since_use and time_since_use > 300:  # 5 minutes
                    status["summary"]["idle"] += 1
            
            status["summary"]["total_usage"] += config.usage_count
        
        return status
    
    
    def get_matrix_view(self) -> str:
        """Get a visual representation of the 3x3 matrix status"""
        models = ["matscibert", "biogpt", "chemberta"]
        backends = ["tgi", "vllm", "tensorrt"]
        
        # Create header
        matrix = "3x3 LLM Matrix Status\n"
        matrix += "=" * 60 + "\n"
        matrix += f"{'Model':<15} | {'TGI':<12} | {'vLLM':<12} | {'TensorRT':<12}\n"
        matrix += "-" * 60 + "\n"
        
        # Add rows
        for model in models:
            row = f"{model:<15} |"
            for backend in backends:
                config_key = f"{model}-{backend}"
                config = self.deployment_configs.get(config_key)
                
                if config:
                    if config.enabled:
                        status = "DEPLOYED"
                    else:
                        status = "OFF"
                else:
                    status = "? UNKNOWN"
                
                row += f" {status:<10} |"
            
            matrix += row + "\n"
        
        matrix += "=" * 60 + "\n"
        
        # Add summary
        deployed = sum(1 for c in self.deployment_configs.values() if c.enabled)
        total = len(self.deployment_configs)
        matrix += f"Deployed: {deployed}/{total} | "
        matrix += f"Resource Usage: {deployed * 1} GPU(s)\n"
        
        return matrix

# Global orchestrator instance
orchestrator = None

def get_orchestrator(namespace: str = os.environ.get("KUBERNETES_NAMESPACE", "default")):
    """Get or create the global orchestrator instance"""
    global orchestrator
    
    if orchestrator is None:
        orchestrator = Orchestrator(namespace)
        
        # Start auto-shutdown monitoring in background
        asyncio.create_task(orchestrator.auto_shutdown_unused())
    
    return orchestrator