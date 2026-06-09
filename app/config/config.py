# app/config/config.py
# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bool(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _path(name: str, folder: str) -> str:
    return os.getenv(name, os.path.join(BASE_DIR, "..", folder))


BASE_FRAMES_DIR = _path("FRAMES_DIR", "frames")
BASE_AUDIO_DIR = _path("AUDIO_DIR", "audio")
MODEL_WEIGHTS_DIR = _path("MODEL_WEIGHTS_DIR", "model_weights")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.20"))
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "640"))

TASK_ADAPTER_MAP = {
    "face_detection": "insightface_adapter",
    "face_embedding": "insightface_adapter",
    "face_recognition": "insightface_adapter",
    "face_verify": "insightface_adapter",
    "watchlist_check": "insightface_adapter",
    "person_detection": "yolov8_adapter",
    "person_counting": "yolov11_adapter",
    "scene_description": "blip_adapter",
    "hf_vision": "huggingface_adapter",
    "object-detection": "huggingface_adapter",
    "image-classification": "huggingface_adapter",
    "image-to-text": "huggingface_adapter",
    "zero_shot_detection": "huggingface_adapter",
    "audio_transcription": "whisper_adapter",
    "audio_translation": "whisper_adapter",
    "chat_completion": "ollama_adapter",
    "speech_synthesis": "piper_adapter",
}

ENABLED_TASKS = {
    "person_counting": _bool("TASK_PERSON_COUNTING_ENABLED"),
    "person_detection": _bool("TASK_PERSON_DETECTION_ENABLED"),
    "face_detection": _bool("TASK_FACE_DETECTION_ENABLED"),
    "face_recognition": _bool("TASK_FACE_RECOGNITION_ENABLED"),
    "scene_description": _bool("TASK_SCENE_DESCRIPTION_ENABLED"),
    "audio_transcription": _bool("TASK_AUDIO_TRANSCRIPTION_ENABLED"),
    "audio_translation": _bool("TASK_AUDIO_TRANSLATION_ENABLED"),
    "chat_completion": _bool("TASK_CHAT_COMPLETION_ENABLED"),
    "speech_synthesis": _bool("TASK_SPEECH_SYNTHESIS_ENABLED"),
}

CONFIG = {
    "adapters": {
        "insightface_adapter": {
            "enabled": _bool("INSIGHTFACE_ENABLED"),
            "weights_path": os.getenv("INSIGHTFACE_WEIGHTS", "insightface"),
        },
        "yolov8_adapter": {
            "enabled": _bool("YOLOV8_ENABLED"),
            "weights_path": os.getenv("YOLOV8_WEIGHTS", "yolov8n.onnx"),
        },
        "yolov11_adapter": {
            "enabled": _bool("YOLOV11_ENABLED"),
            "weights_path": os.getenv("YOLOV11_WEIGHTS", "yolo11m.pt"),
        },
        "blip_adapter": {
            "enabled": _bool("BLIP_ENABLED"),
            "model_id": os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base"),
        },
        "huggingface_adapter": {
            "enabled": _bool("HUGGINGFACE_ENABLED"),
        },
        "whisper_adapter": {
            "enabled": _bool("WHISPER_ENABLED"),
            "model_size": os.getenv("WHISPER_MODEL_SIZE", "base"),
            "device": os.getenv("WHISPER_DEVICE", "auto"),
            "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "auto"),
        },
        "ollama_adapter": {
            "enabled": _bool("OLLAMA_ENABLED"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            "timeout_s": float(os.getenv("OLLAMA_TIMEOUT_S", "60")),
        },
        "piper_adapter": {
            "enabled": _bool("PIPER_ENABLED"),
            "voice": os.getenv("PIPER_VOICE", "en_US-libritts-high"),
        },
    },
    "warmup": [
        item.strip()
        for item in os.getenv("WARMUP_ADAPTERS", "yolov8_adapter,yolov11_adapter").split(",")
        if item.strip()
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
    if not is_task_enabled(task_name):
        return None
    return TASK_ADAPTER_MAP.get(task_name)
