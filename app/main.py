"""
OpenNVR AI Adapter - Clean Architecture Main Entry Point

This server provides:
- Lazy-loaded plugin adapters and tasks
- Task routing via ModelRouter
- Pipeline execution via PipelineEngine
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.endpoints import router as api_router, public_router, set_global_router
from app.router.model_router import ModelRouter
from app.pipelines.engine import PipelineEngine
import app.config.config as config_module
from app.utils.loader import PluginManager

# Setup standard clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenNVR_Adapter")


async def startup_event() -> None:
    """Discover plugins and wire up the ModelRouter + PipelineEngine.

    Extracted from the lifespan handler so tests can monkey-patch this
    name to inject a fake router/engine without actually loading plugins
    (see ``tests/test_endpoints.py``).
    """
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager. Replaces the legacy
    ``@app.on_event("startup")`` decorator which FastAPI has deprecated.

    Delegates to the ``startup_event`` coroutine above so tests can
    patch the underlying logic without having to monkey-patch the
    lifespan itself.
    """
    await startup_event()
    try:
        yield
    finally:
        # Plugins are lazily loaded and hold no background tasks of
        # their own, but the module-level router/engine references in
        # ``app.api.endpoints`` survive an ``importlib.reload`` and
        # would leak previous instances into subsequent test runs.
        # Clear them so the next lifespan startup re-injects fresh
        # state. Safe to call with (None, None) — the endpoints guard
        # for the unset case.
        set_global_router(None, None)


app = FastAPI(
    title="OpenNVR Modular AI Adapter",
    version="2.0",
    lifespan=lifespan,
)

app.include_router(public_router)
app.include_router(api_router)
