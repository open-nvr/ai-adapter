# app/config/config.py

# Simulating a parsed YAML config file
CONFIG = {
    "adapters": {
        "yolo_adapter": {
            "enabled": True,
            "weights_path": "yolov8m.pt"
        },
        "mock_llm_adapter": {
            "enabled": True,
            "mode": "api", # Optional configuration (api vs local)
            "api_endpoint": "https://api.openai.com/v1/completions"
        },
        "disabled_vision": {
            "enabled": False
        }
    },
    "warmup": [
        "yolo_adapter"  # E.g., Only YOLO is heavy, so we warm it up at startup
    ],
    "routing": {
        "detect_objects": "yolo_adapter",
        "mock_vision": "yolo_adapter",
        "ask_question": "mock_llm_adapter"
    }
}

def get_adapter_config(adapter_name: str) -> dict:
    return CONFIG["adapters"].get(adapter_name, {})

def is_adapter_enabled(adapter_name: str) -> bool:
    return get_adapter_config(adapter_name).get("enabled", False)

def get_warmup_adapters() -> list:
    return CONFIG.get("warmup", [])

def get_adapter_for_task(task_name: str) -> str:
    """Returns the adapter name mapped to this task"""
    return CONFIG["routing"].get(task_name)
