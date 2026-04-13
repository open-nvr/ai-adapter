# app/config/config.py
# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Configuration for the Clean Architecture AI Adapter.
"""
import os
from typing import Optional

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_FRAMES_DIR = os.path.join(BASE_DIR, "..", "frames")
MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "model_weights")

# Model inference settings
CONFIDENCE_THRESHOLD = 0.20  # Balanced - detects distant people, fewer false positives
INPUT_SIZE = 640

TASK_ADAPTER_MAP = {
    "face_detection": "insightface_adapter",
    "face_embedding": "insightface_adapter",
    "face_recognition": "insightface_adapter",
    "face_verify": "insightface_adapter",
    "watchlist_check": "insightface_adapter",
    "person_detection": "yolov8_adapter",
    "person_counting": "yolov8_adapter",
    "scene_description": "blip_adapter",
    "hf_vision": "huggingface_adapter",
    "object-detection": "huggingface_adapter",
    "image-classification": "huggingface_adapter",
    "image-to-text": "huggingface_adapter",
    "zero_shot_detection": "huggingface_adapter",
}

ENABLED_TASKS = {
    "person_counting": True,
    "person_detection": True,
    "face_detection": True,
    "face_recognition": True,
    "scene_description": True,
}

# Simulating a parsed YAML config file
CONFIG = {
    "adapters": {
        "insightface_adapter": {
            "enabled": True,
            "weights_path": "insightface"
        },
        "yolov8_adapter": {
            "enabled": True,
            "weights_path": "yolov8n.onnx"
        },
        "yolov11_adapter": {
            "enabled": True,
            "weights_path": "yolo11m.pt"
        },
        "blip_adapter": {
            "enabled": True,
            "model_id": "Salesforce/blip-image-captioning-base"
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
    "routing": TASK_ADAPTER_MAP,
}

def get_adapter_config(adapter_name: str) -> dict:
    return CONFIG["adapters"].get(adapter_name, {})

def is_adapter_enabled(adapter_name: str) -> bool:
    return get_adapter_config(adapter_name).get("enabled", False)

def get_warmup_adapters() -> list:
    return CONFIG.get("warmup", [])

def is_task_enabled(task_name: str) -> bool:
    return ENABLED_TASKS.get(task_name, True)

def get_adapter_for_task(task_name: str) -> Optional[str]:
    """Returns the adapter name mapped to this task"""
    if not is_task_enabled(task_name):
        return None
    return TASK_ADAPTER_MAP.get(task_name)
