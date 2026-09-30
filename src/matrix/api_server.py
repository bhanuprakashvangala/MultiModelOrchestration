"""
Intelligent API Server for 3x3 Domain-Specific LLM Matrix
Provides REST API endpoints for BioGPT, ChemBERTa, and MatSciBERT
across vLLM, TGI, and TensorRT backends with intelligent routing
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add core components to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import backend manager and orchestrator
from backend_manager import backend_manager, BackendStatus
from orchestrator import get_orchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="3x3 Domain-Specific LLM Matrix API",
    description="Intelligent routing system for domain-specific LLMs (BioGPT, ChemBERTa, MatSciBERT) x (vLLM, TGI, TensorRT)",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    model: str = Field(default=None, description="Model to use: biogpt, chemberta, matscibert (optional - will auto-select based on domain)")
    backend: str = Field(default=None, description="Backend preference: vllm, tgi, tensorrt (optional)")
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

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the specified model and backend"""
    
    if request.model and request.model not in backend_manager.endpoints:
        available_models = list(backend_manager.endpoints.keys())
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{request.model}' not available. Available models: {available_models}"
        )
    
    start_time = datetime.now()
    
    try:
        # If model not specified, use intelligent routing
        if not request.model:
            orchestrator = get_orchestrator()
            result = await orchestrator.process_request(
                prompt=request.prompt,
                optimize_for=request.optimize_for
            )
        else:
            # Use specific model/backend
            result = await backend_manager.generate_completion(
                model=request.model,
                prompt=request.prompt,
                backend=request.backend,
                max_tokens=request.max_tokens,
                temperature=request.temperature
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
            selection_time_ms=result.get("selection_time_ms", 0)
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/api/v1/status")
async def get_system_status():
    """Get system status for all models and backends"""
    await backend_manager.check_all_health()
    return backend_manager.get_system_status()

@app.get("/api/v1/models")
async def list_models():
    """List available models"""
    return {
        "models": [
            {
                "name": "biogpt",
                "description": "Microsoft BioGPT for biomedical text generation",
                "domain": "biology",
                "backends": ["vllm", "tgi", "tensorrt"]
            },
            {
                "name": "chemberta", 
                "description": "ChemBERTa for chemistry and molecular understanding",
                "domain": "chemistry",
                "backends": ["vllm", "tgi", "tensorrt"]
            },
            {
                "name": "matscibert",
                "description": "MatSciBERT for materials science applications",
                "domain": "materials",
                "backends": ["vllm", "tgi", "tensorrt"]
            }
        ]
    }

@app.post("/api/v1/intelligent-route")
async def intelligent_route(request: IntelligentRoutingRequest):
    """Use intelligent routing to automatically select the best model and backend"""
    orchestrator = get_orchestrator()
    
    try:
        result = await orchestrator.process_request(
            prompt=request.prompt,
            optimize_for=request.optimize_for
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except Exception as e:
        logger.error(f"Intelligent routing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")

@app.get("/api/v1/orchestrator/status")
async def get_orchestrator_status():
    """Get orchestrator deployment status"""
    orchestrator = get_orchestrator()
    return orchestrator.get_deployment_status()

@app.get("/api/v1/orchestrator/matrix")
async def get_matrix_view():
    """Get visual matrix representation"""
    orchestrator = get_orchestrator()
    return {"matrix": orchestrator.get_matrix_view()}

@app.post("/api/v1/orchestrator/deploy")
async def deploy_endpoint(model: str, backend: str):
    """Manually deploy a specific endpoint"""
    orchestrator = get_orchestrator()
    
    success = await orchestrator.deploy_endpoint(model, backend)
    
    if success:
        return {"status": "success", "message": f"Deployed {model}-{backend}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to deploy {model}-{backend}")

@app.post("/api/v1/orchestrator/undeploy")
async def undeploy_endpoint(model: str, backend: str):
    """Manually undeploy a specific endpoint"""
    orchestrator = get_orchestrator()
    
    success = await orchestrator.undeploy_endpoint(model, backend)
    
    if success:
        return {"status": "success", "message": f"Undeployed {model}-{backend}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to undeploy {model}-{backend}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Initialize orchestrator on startup
    orchestrator = get_orchestrator()
    
    # Start the API server
    uvicorn.run(app, host=os.environ.get("API_HOST", "localhost"), port=int(os.environ.get("API_PORT", "8080")), log_level="info")