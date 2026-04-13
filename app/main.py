"""
OpenNVR AI Adapter - Clean Architecture Main Entry Point

This server provides:
- Lazy-loaded plugin adapters and tasks
- Task routing via ModelRouter
- Pipeline execution via PipelineEngine
"""
import logging
from fastapi import FastAPI
from app.api.endpoints import router as api_router, set_global_router
from app.router.model_router import ModelRouter
from app.pipelines.engine import PipelineEngine
import app.config.config as config_module
from app.utils.loader import PluginManager

# Setup standard clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenNVR_Adapter")

app = FastAPI(title="OpenNVR Modular AI Adapter", version="2.0")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Core AI Router and Engine...")
    PluginManager.discover_plugins()

    router = ModelRouter(config_module)
    engine = PipelineEngine(router)

    # Inject dependencies globally for FastAPI endpoints
    set_global_router(router, engine)
    logger.info(
        "Server ready. Discovered tasks=%d adapters=%d",
        len(PluginManager.TASK_REGISTRY),
        len(PluginManager.ADAPTER_REGISTRY),
    )

app.include_router(api_router)
