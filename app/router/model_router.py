from typing import Dict, Any

class ModelRouter:
    """
    Routes inference tasks to the correct activated adapter based on configuration.
    """
    def __init__(self, config_module):
        self.config_module = config_module
        self.adapters: Dict[str, Any] = {}

    def register_adapter(self, adapter):
        """Register an adapter if it is enabled in the config."""
        if self.config_module.is_adapter_enabled(adapter.name):
            self.adapters[adapter.name] = adapter

    def route_task(self, task_name: str, input_data: Any) -> Dict[str, Any]:
        """Route a specific task to its mapped adapter and execute."""
        adapter_name = self.config_module.get_adapter_for_task(task_name)
        if not adapter_name:
            raise ValueError(f"No routing rule found for task '{task_name}'")
        
        adapter = self.adapters.get(adapter_name)
        if not adapter:
            raise ValueError(f"Adapter '{adapter_name}' is not registered or enabled")
        if isinstance(input_data, dict) and "task" not in input_data:
            input_data["task"] = task_name
            
        return adapter.infer(input_data)
        
    def get_available_adapters(self) -> list:
        """List all successfully loaded and enabled adapters."""
        return list(self.adapters.keys())
