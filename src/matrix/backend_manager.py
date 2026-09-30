"""
Dynamic Backend Management System for 3x3 Domain-Specific LLM Matrix
Handles real-time switching between vLLM, TGI, and TensorRT backends
for BioGPT, ChemBERTa, and MatSciBERT models with on-demand deployment
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
import requests
import subprocess
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from domain_classifier import get_domain_classifier
    USE_TRANSFORMER_CLASSIFIER = True
except ImportError:
    USE_TRANSFORMER_CLASSIFIER = False
    logging.warning("Transformer domain classifier not available, using keyword-based classification")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackendType(Enum):
    VLLM = "vllm"
    TGI = "tgi"
    TENSORRT = "tensorrt"

class BackendStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEPLOYED = "deployed"
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"

class DomainType(Enum):
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    MATERIALS = "materials"
    GENERAL = "general"

@dataclass
class ModelEndpoint:
    model: str
    backend: str
    host: str
    port: int
    domain: DomainType
    status: BackendStatus = BackendStatus.NOT_DEPLOYED
    deployment_status: BackendStatus = BackendStatus.NOT_DEPLOYED
    helm_release: str = ""
    namespace: str = "default"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.backend == "vllm":
            self.health_check_url = f"http://{self.host}:{self.port}/health"
            self.generate_url = f"http://{self.host}:{self.port}/v1/completions"
        elif self.backend == "tgi":
            self.health_check_url = f"http://{self.host}:{self.port}/health"
            self.generate_url = f"http://{self.host}:{self.port}/generate"
        elif self.backend == "tensorrt":
            self.health_check_url = f"http://{self.host}:{self.port}/v2/health/ready"
            self.generate_url = f"http://{self.host}:{self.port}/v2/models/{self.model}/infer"
        
        # Initialize performance metrics
        self.performance_metrics = {
            "latency_ms": 0,
            "throughput": 0,
            "cost_per_token": 0,
            "success_rate": 100
        }

class BackendManager:
    def __init__(self, kubernetes_namespace: str = "default"):
        self.namespace = kubernetes_namespace
        self.endpoints = self._initialize_endpoints()
        self.health_check_interval = 30  # seconds
        self.last_health_check = {}
        self.domain_keywords = self._initialize_domain_keywords()
        
    def _initialize_endpoints(self) -> Dict[str, List[ModelEndpoint]]:
        """Initialize all 9 model-backend combinations (3x3 matrix)"""
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
            ]
        }
        
        # Set helm release names for each endpoint
        for model_name, model_endpoints in endpoints.items():
            for endpoint in model_endpoints:
                endpoint.helm_release = f"{model_name}-{endpoint.backend}"
                endpoint.namespace = self.namespace
        
        return endpoints
    
    def _initialize_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Initialize domain-specific keywords for intelligent routing"""
        return {
            DomainType.BIOLOGY: [
                "protein", "gene", "dna", "rna", "cell", "enzyme", "antibody", "virus",
                "bacteria", "genome", "mutation", "evolution", "disease", "drug", "medicine",
                "biological", "organism", "tissue", "molecular", "pathway", "receptor"
            ],
            DomainType.CHEMISTRY: [
                "molecule", "reaction", "compound", "element", "chemical", "synthesis",
                "catalyst", "acid", "base", "ph", "bond", "organic", "inorganic",
                "polymer", "solution", "concentration", "molarity", "oxidation", "reduction",
                "electrochemistry", "thermodynamics"
            ],
            DomainType.MATERIALS: [
                "material", "crystal", "lattice", "semiconductor", "metal", "alloy",
                "composite", "nanomaterial", "graphene", "polymer", "ceramic", "glass",
                "mechanical", "thermal", "electrical", "optical", "magnetic", "properties",
                "structure", "defect", "phase"
            ]
        }
    
    def classify_domain(self, prompt: str) -> Tuple[DomainType, float]:
        """Classify the domain of a prompt using transformer or keyword-based approach"""
        
        # Try to use transformer-based classifier first
        if USE_TRANSFORMER_CLASSIFIER:
            try:
                classifier = get_domain_classifier(use_lightweight=True)
                if classifier:
                    domain_str, confidence, probabilities = classifier.classify(prompt)
                    
                    # Map string to DomainType enum
                    domain_map = {
                        "biology": DomainType.BIOLOGY,
                        "chemistry": DomainType.CHEMISTRY,
                        "materials": DomainType.MATERIALS,
                        "general": DomainType.GENERAL
                    }
                    
                    domain = domain_map.get(domain_str, DomainType.GENERAL)
                    
                    logger.info(f"Transformer classified prompt as {domain.value} with confidence {confidence:.2f}")
                    logger.debug(f"Probabilities: {probabilities}")
                    
                    # If confidence is too low, return GENERAL
                    if confidence < 0.4:
                        return DomainType.GENERAL, confidence
                    
                    return domain, confidence
                    
            except Exception as e:
                logger.warning(f"Transformer classification failed, falling back to keywords: {e}")
        
        # Fallback to keyword-based classification
        prompt_lower = prompt.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            domain_scores[domain] = score
        
        # Get the domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[best_domain] / max(1, sum(domain_scores.values()))
        
        # If confidence is too low, return GENERAL
        if confidence < 0.3:
            return DomainType.GENERAL, confidence
        
        logger.info(f"Keyword-based classified prompt as {best_domain.value} with confidence {confidence:.2f}")
        return best_domain, confidence
    
    async def deploy_endpoint(self, endpoint: ModelEndpoint) -> bool:
        """Deploy an endpoint on-demand using Helm"""
        if endpoint.deployment_status in [BackendStatus.DEPLOYED, BackendStatus.DEPLOYING]:
            return True
        
        try:
            endpoint.deployment_status = BackendStatus.DEPLOYING
            logger.info(f"Deploying {endpoint.helm_release} in namespace {endpoint.namespace}")
            
            # Check if helm chart exists
            chart_path = os.path.join(os.environ.get("MATRIX_CHART_DIR", "./deploy/helm/pick-and-spin-umbrella"), "charts", endpoint.helm_release)
            if not os.path.exists(chart_path):
                logger.error(f"Helm chart not found at {chart_path}")
                return False
            
            # Deploy using helm
            cmd = [
                "helm", "upgrade", "--install",
                endpoint.helm_release,
                chart_path,
                "--namespace", endpoint.namespace,
                "--create-namespace",
                "--wait", "--timeout", "5m"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                endpoint.deployment_status = BackendStatus.DEPLOYED
                logger.info(f"Successfully deployed {endpoint.helm_release}")
                
                # Wait for pod to be ready
                await asyncio.sleep(10)
                return True
            else:
                logger.error(f"Failed to deploy {endpoint.helm_release}: {result.stderr}")
                endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
                return False
                
        except Exception as e:
            logger.error(f"Error deploying {endpoint.helm_release}: {e}")
            endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
            return False
    
    async def undeploy_endpoint(self, endpoint: ModelEndpoint) -> bool:
        """Undeploy an endpoint to save resources"""
        if endpoint.deployment_status == BackendStatus.NOT_DEPLOYED:
            return True
        
        try:
            cmd = [
                "helm", "uninstall",
                endpoint.helm_release,
                "--namespace", endpoint.namespace
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                endpoint.deployment_status = BackendStatus.NOT_DEPLOYED
                endpoint.status = BackendStatus.NOT_DEPLOYED
                logger.info(f"Successfully undeployed {endpoint.helm_release}")
                return True
            else:
                logger.error(f"Failed to undeploy {endpoint.helm_release}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error undeploying {endpoint.helm_release}: {e}")
            return False
    
    async def check_health(self, endpoint: ModelEndpoint) -> BackendStatus:
        """Check health status of a specific endpoint"""
        # Only check health if deployed
        if endpoint.deployment_status != BackendStatus.DEPLOYED:
            endpoint.status = BackendStatus.NOT_DEPLOYED
            return BackendStatus.NOT_DEPLOYED
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(endpoint.health_check_url) as response:
                    if response.status == 200:
                        endpoint.status = BackendStatus.HEALTHY
                        return BackendStatus.HEALTHY
        except Exception as e:
            logger.warning(f"Health check failed for {endpoint.model}-{endpoint.backend}: {e}")
        
        endpoint.status = BackendStatus.UNHEALTHY
        return BackendStatus.UNHEALTHY
    
    async def check_all_health(self):
        """Check health status of all endpoints"""
        tasks = []
        for model, endpoints in self.endpoints.items():
            for endpoint in endpoints:
                tasks.append(self.check_health(endpoint))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_healthy_endpoints(self, model: str) -> List[ModelEndpoint]:
        """Get all healthy endpoints for a specific model"""
        if model not in self.endpoints:
            return []
        
        return [ep for ep in self.endpoints[model] if ep.status == BackendStatus.HEALTHY]
    
    async def select_optimal_endpoint(self, prompt: str, 
                                     optimize_for: str = "balanced") -> Optional[ModelEndpoint]:
        """
        Select the optimal endpoint from the 3x3 matrix based on prompt analysis
        optimize_for: "speed", "cost", "quality", "balanced"
        """
        # Classify domain
        domain, confidence = self.classify_domain(prompt)
        logger.info(f"Classified prompt as {domain.value} with confidence {confidence:.2f}")
        
        # Get model based on domain
        model_map = {
            DomainType.BIOLOGY: "biogpt",
            DomainType.CHEMISTRY: "chemberta",
            DomainType.MATERIALS: "matscibert",
            DomainType.GENERAL: "biogpt"  # Default to biogpt for general
        }
        
        model = model_map.get(domain, "biogpt")
        
        # Backend priority based on optimization preference
        backend_priorities = {
            "speed": {"tensorrt": 1, "vllm": 2, "tgi": 3},
            "cost": {"tgi": 1, "vllm": 2, "tensorrt": 3},
            "quality": {"vllm": 1, "tensorrt": 2, "tgi": 3},
            "balanced": {"vllm": 1, "tgi": 2, "tensorrt": 3}
        }
        
        priorities = backend_priorities.get(optimize_for, backend_priorities["balanced"])
        
        # Try to find best endpoint
        best_endpoint = None
        best_score = float('inf')
        
        for endpoint in self.endpoints[model]:
            # Calculate score based on priority and performance metrics
            priority_score = priorities.get(endpoint.backend, 99)
            perf_score = (
                endpoint.performance_metrics.get("latency_ms", 1000) / 1000 +
                (100 - endpoint.performance_metrics.get("success_rate", 0)) / 100
            )
            total_score = priority_score + perf_score
            
            if total_score < best_score:
                best_score = total_score
                best_endpoint = endpoint
        
        # Deploy the endpoint if not already deployed
        if best_endpoint and best_endpoint.deployment_status != BackendStatus.DEPLOYED:
            logger.info(f"Deploying optimal endpoint: {best_endpoint.helm_release}")
            deployed = await self.deploy_endpoint(best_endpoint)
            if not deployed:
                logger.error(f"Failed to deploy {best_endpoint.helm_release}")
                return None
            
            # Wait for health check
            await asyncio.sleep(5)
            await self.check_health(best_endpoint)
        
        return best_endpoint
    
    def get_best_endpoint(self, model: str, preferred_backend: str = None) -> Optional[ModelEndpoint]:
        """Get the best available endpoint for a model"""
        healthy_endpoints = self.get_healthy_endpoints(model)
        
        if not healthy_endpoints:
            return None
        
        # If preferred backend is specified and healthy, use it
        if preferred_backend:
            for ep in healthy_endpoints:
                if ep.backend == preferred_backend:
                    return ep
        
        # Otherwise, return first healthy endpoint (priority: VLLM > TGI > TensorRT)
        backend_priority = {"vllm": 1, "tgi": 2, "tensorrt": 3}
        return min(healthy_endpoints, key=lambda x: backend_priority.get(x.backend, 99))
    
    async def generate_completion_with_routing(self, prompt: str, 
                                              optimize_for: str = "balanced",
                                              **kwargs) -> Dict[str, Any]:
        """Generate completion using intelligent routing to select optimal endpoint"""
        start_time = time.time()
        
        # Select optimal endpoint based on prompt
        endpoint = await self.select_optimal_endpoint(prompt, optimize_for)
        
        if not endpoint:
            raise RuntimeError("No suitable endpoint could be deployed")
        
        # Track metrics
        selection_time = (time.time() - start_time) * 1000
        
        # Generate completion
        result = await self._generate_from_endpoint(endpoint, prompt, **kwargs)
        
        # Update performance metrics
        total_time = (time.time() - start_time) * 1000
        endpoint.performance_metrics["latency_ms"] = total_time
        
        result["selection_time_ms"] = selection_time
        result["total_time_ms"] = total_time
        result["domain"] = endpoint.domain.value
        
        return result
    
    async def generate_completion(self, model: str, prompt: str, 
                                backend: str = None, **kwargs) -> Dict[str, Any]:
        """Generate completion using the specified model and backend"""
        endpoint = self.get_best_endpoint(model, backend)
        
        if not endpoint:
            # Try to deploy if not available
            for ep in self.endpoints.get(model, []):
                if backend and ep.backend != backend:
                    continue
                if await self.deploy_endpoint(ep):
                    await asyncio.sleep(5)
                    if await self.check_health(ep) == BackendStatus.HEALTHY:
                        endpoint = ep
                        break
            
            if not endpoint:
                raise RuntimeError(f"No healthy endpoints available for model: {model}")
        
        return await self._generate_from_endpoint(endpoint, prompt, **kwargs)
    
    async def _generate_from_endpoint(self, endpoint: ModelEndpoint, prompt: str, **kwargs) -> Dict[str, Any]:
        """Internal method to generate from a specific endpoint"""
        # Prepare request based on backend type
        if endpoint.backend == "vllm":
            payload = {
                "model": endpoint.model,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 150),
                "temperature": kwargs.get("temperature", 0.7),
            }
        elif endpoint.backend == "tgi":
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get("max_tokens", 150),
                    "temperature": kwargs.get("temperature", 0.7),
                }
            }
        elif endpoint.backend == "tensorrt":
            payload = {
                "inputs": [
                    {
                        "name": "input_text",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [prompt]
                    }
                ],
                "parameters": {
                    "max_tokens": kwargs.get("max_tokens", 150),
                    "temperature": kwargs.get("temperature", 0.7),
                }
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint.generate_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "model": endpoint.model,
                            "backend": endpoint.backend,
                            "response": result,
                            "endpoint": f"{endpoint.host}:{endpoint.port}"
                        }
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Generation failed with status {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Generation failed for {endpoint.model} on {endpoint.backend}: {e}")
            endpoint.performance_metrics["success_rate"] = max(0, endpoint.performance_metrics["success_rate"] - 10)
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status of the 3x3 matrix"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "matrix": {},
            "summary": {
                "total_endpoints": 9,
                "deployed": 0,
                "healthy": 0,
                "unhealthy": 0,
                "not_deployed": 0
            }
        }
        
        for model, endpoints in self.endpoints.items():
            model_status = {
                "domain": endpoints[0].domain.value if endpoints else "unknown",
                "backends": {}
            }
            
            for endpoint in endpoints:
                model_status["backends"][endpoint.backend] = {
                    "status": endpoint.status.value,
                    "deployment_status": endpoint.deployment_status.value,
                    "endpoint": f"{endpoint.host}:{endpoint.port}",
                    "performance": endpoint.performance_metrics
                }
                
                # Update summary
                if endpoint.deployment_status == BackendStatus.DEPLOYED:
                    status["summary"]["deployed"] += 1
                    if endpoint.status == BackendStatus.HEALTHY:
                        status["summary"]["healthy"] += 1
                    else:
                        status["summary"]["unhealthy"] += 1
                else:
                    status["summary"]["not_deployed"] += 1
            
            status["matrix"][model] = model_status
        
        return status
    

# Global instance
backend_manager = BackendManager()