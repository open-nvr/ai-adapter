import asyncio
from typing import Any, Dict

from app.interfaces.adapter import ResponsePayload
from app.utils.loader import PluginManager


class ModelRouter:
    """
    Routes tasks using plugin registries and lazily-instantiated adapter instances.
    """

    def __init__(self, config_module):
        if not PluginManager.ADAPTER_REGISTRY and not PluginManager.TASK_REGISTRY:
            PluginManager.discover_plugins()
        self.config_module = config_module
        self.adapters: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}

    async def get_or_create_adapter(self, adapter_name: str):
        if adapter_name in self.adapters:
            return self.adapters[adapter_name]

        if not self.config_module.is_adapter_enabled(adapter_name):
            raise ValueError(f"Adapter '{adapter_name}' is disabled in config")

        adapter_cls = PluginManager.ADAPTER_REGISTRY.get(adapter_name)
        if adapter_cls is None:
            raise ValueError(f"Adapter '{adapter_name}' was not discovered")

        adapter_config = self.config_module.get_adapter_config(adapter_name)
        import asyncio
        adapter = await asyncio.to_thread(adapter_cls, config=adapter_config)
        self.adapters[adapter_name] = adapter
        return adapter

    def ensure_adapter_loaded(self, adapter: Any) -> None:
        if hasattr(adapter, "ensure_model_loaded"):
            adapter.ensure_model_loaded()
            return

        if getattr(adapter, "_plugin_model_loaded", False):
            return

        adapter.load_model()
        setattr(adapter, "_plugin_model_loaded", True)

    async def _get_or_create_task(self, task_name: str):
        if task_name in self.tasks:
            return self.tasks[task_name]

        task_cls = PluginManager.TASK_REGISTRY.get(task_name)
        if task_cls is None:
            return None

        import asyncio
        task_instance = await asyncio.to_thread(task_cls)
        self.tasks[task_name] = task_instance
        return task_instance

    @staticmethod
    def _with_task_name(task_name: str, input_data: Any) -> Any:
        if not isinstance(input_data, dict):
            return input_data

        payload = dict(input_data)
        payload.setdefault("task", task_name)
        return payload

    def _predict_with_adapter(
        self, adapter: Any, adapter_name: str, task_name: str, input_data: Any
    ) -> ResponsePayload:
        payload = self._with_task_name(task_name, input_data)

        predict_fn = getattr(adapter, "predict", None)
        if callable(predict_fn):
            return predict_fn(payload)

        infer_fn = getattr(adapter, "infer", None)
        if callable(infer_fn):
            return infer_fn(payload)

        raise ValueError(
            f"Adapter '{adapter_name}' does not implement a callable prediction method"
        )

    async def route_task(self, task_name: str, input_data: Any) -> ResponsePayload:
        import asyncio
        adapter_name = self.config_module.get_adapter_for_task(task_name)
        if not adapter_name:
            raise ValueError(f"No routing rule found for task '{task_name}'")

        adapter = await self.get_or_create_adapter(adapter_name)
        await asyncio.to_thread(self.ensure_adapter_loaded, adapter)

        task = await self._get_or_create_task(task_name)
        if task is not None:
            return await asyncio.to_thread(task.process, input_data, adapter)

        return await asyncio.to_thread(
            self._predict_with_adapter,
            adapter,
            adapter_name,
            task_name,
            input_data,
        )

    def get_available_adapters(self) -> list:
        return list(self.adapters.keys())
