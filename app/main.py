import logging
from fastapi import FastAPI
from app.api.endpoints import router as api_router, set_global_router
from app.router.model_router import ModelRouter
from app.pipelines.engine import PipelineEngine
import app.config.config as config_module

from app.adapters.vision.yolo_adapter import YOLOAdapter
from app.adapters.llm.llm_adapter import LocalLLMAdapter

# Setup standard clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OpenNVR_Adapter")

app = FastAPI(title="OpenNVR Modular AI Adapter", version="2.0")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing Core AI Router and Engine...")
    router = ModelRouter(config_module)
    engine = PipelineEngine(router)
    
    # We maintain a list of all potential available classes that could be enabled
    available_classes = [YOLOAdapter, LocalLLMAdapter]
    
    # Strictly config-driven instantiation
    for adapter_cls in available_classes:
        adapter_name = adapter_cls.name
        if config_module.is_adapter_enabled(adapter_name):
            try:
                # Instantiate with config - does NOT load model yet
                adapter_config = config_module.get_adapter_config(adapter_name)
                adapter_instance = adapter_cls(config=adapter_config)
                router.register_adapter(adapter_instance)
            except Exception as e:
                logger.error(f"Failed to instantiate {adapter_name}: {e}")
        else:
            logger.debug(f"Adapter {adapter_name} disabled in config. Skipping instantiation entirely.")
            
    # Process Warmup Hooks
    warmup_list = config_module.get_warmup_adapters()
    for w_adapter in warmup_list:
        adapter = router.adapters.get(w_adapter)
        if adapter:
            logger.info(f"Targeted Warmup: ensuring {w_adapter} model is loaded into memory.")
            adapter.ensure_model_loaded()

    # Inject dependencies globally for FastAPI endpoints
    set_global_router(router, engine)
    logger.info(f"Server ready. Enabled Active Adapters: {router.get_available_adapters()}")

app.include_router(api_router)
