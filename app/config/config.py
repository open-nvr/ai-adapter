# app/config/config.py

# Simulating a parsed YAML config file
CONFIG = {
    "adapters": {
        "insightface_adapter": {
            "enabled": True,
            "weights_path": "insightface"
        },
        "yolov8_adapter": {
            "enabled": True,
            "weights_path": "yolov8m.onnx"
        },
        "yolov11_adapter": {
            "enabled": True,
            "weights_path": "yolo11m.pt"
        },
        "blip_adapter": {
            "enabled": True
        },
        "huggingface_adapter": {
            "enabled": True
        },
        "disabled_vision": {
            "enabled": False
        }
    },
    "warmup": [
        "yolov8_adapter", "yolov11_adapter"
    ],
    "routing": {
        "face_detection": "insightface_adapter",
        "face_embedding": "insightface_adapter",
        "face_recognition": "insightface_adapter",
        "face_verify": "insightface_adapter",
        "watchlist_check": "insightface_adapter",
        "person_detection": "yolov8_adapter",
        "person_counting": "yolov11_adapter",
        "scene_description": "blip_adapter",
        "zero_shot_detection": "huggingface_adapter"
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
