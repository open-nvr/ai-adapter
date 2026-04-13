import pytest
import asyncio
from app.router.model_router import ModelRouter
from app.utils.loader import PluginManager

class DummyAdapter:
    def __init__(self, config=None):
        self.config = config
        self.loaded = False
        self.predictions = []

    def load_model(self):
        import time
        # Simulate blocking I/O
        time.sleep(0.1)
        self.loaded = True

    def predict(self, input_data):
        self.predictions.append(input_data)
        return {"predictions": [{"label": "dummy", "confidence": 0.99}], "task": input_data.get("task")}

class DummyTask:
    def __init__(self):
        self.processed = False

    def process(self, input_data, adapter):
        # The task applies business logic to the adapter response
        self.processed = True
        raw_res = adapter.predict(input_data)
        return {"task_processed": True, "raw": raw_res}

class DummyConfig:
    def is_adapter_enabled(self, adapter_name: str) -> bool:
        return True

    def get_adapter_config(self, adapter_name: str) -> dict:
        return {"fake": "config"}

    def get_adapter_for_task(self, task_name: str) -> str:
        return "dummy_adapter"

@pytest.fixture(autouse=True)
def wipe_registries():
    # Clear registries before each test
    PluginManager.ADAPTER_REGISTRY.clear()
    PluginManager.TASK_REGISTRY.clear()
    yield

@pytest.mark.asyncio
async def test_lazy_load_adapter():
    PluginManager.ADAPTER_REGISTRY["dummy_adapter"] = DummyAdapter

    config = DummyConfig()
    router = ModelRouter(config)

    assert "dummy_adapter" not in router.adapters

    adapter = await router.get_or_create_adapter("dummy_adapter")
    
    assert "dummy_adapter" in router.adapters
    assert adapter is not None
    assert type(adapter) == DummyAdapter
    assert adapter.config == {"fake": "config"}

    adapter_cached = await router.get_or_create_adapter("dummy_adapter")
    assert adapter is adapter_cached  # Same instance

@pytest.mark.asyncio
async def test_route_task_executes_fallback_adapter():
    PluginManager.ADAPTER_REGISTRY["dummy_adapter"] = DummyAdapter

    config = DummyConfig()
    router = ModelRouter(config)

    res = await router.route_task("random_task", {"image": "base64", "task": "random_task"})
    
    assert res["task"] == "random_task"
    assert "predictions" in res

    # Adapter should have been lazily loaded
    assert "dummy_adapter" in router.adapters
    assert router.adapters["dummy_adapter"].loaded is True

@pytest.mark.asyncio
async def test_route_task_executes_task_plugin():
    PluginManager.ADAPTER_REGISTRY["dummy_adapter"] = DummyAdapter
    PluginManager.TASK_REGISTRY["smart_task"] = DummyTask

    config = DummyConfig()
    router = ModelRouter(config)
    
    res = await router.route_task("smart_task", {"image": "base64", "task": "smart_task"})
    
    assert res["task_processed"] is True
    assert res["raw"]["task"] == "smart_task"
    
    assert "smart_task" in router.tasks
    assert router.tasks["smart_task"].processed is True