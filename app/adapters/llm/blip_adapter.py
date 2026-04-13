# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
BLIP Image Captioning adapter using lazy model loading.
Ported from legacy BLIP handler into the adapter plugin architecture.
"""
import logging
import pathlib
import time
from typing import Any, Dict

from app.adapters.base import BaseAdapter
from app.config import BASE_FRAMES_DIR

logger = logging.getLogger(__name__)


class BLIPAdapter(BaseAdapter):
    name = "blip_adapter"
    type = "llm"

    SUPPORTED_TASKS = ["scene_description"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._processor = None
        self._blip_model = None
        self._device = None
        self._model_id = self.config.get("model_id", "Salesforce/blip-image-captioning-base")

    def load_model(self) -> None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        logger.info("Loading BLIP model: %s", self._model_id)

        self._processor = BlipProcessor.from_pretrained(self._model_id)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(self._model_id)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._blip_model.to(self._device)
        self._blip_model.eval()

        self.model = self._blip_model
        logger.info("BLIP model loaded on %s", self._device)

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from PIL import Image

        task = input_data.get("task", "scene_description")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"BLIPAdapter only supports {self.SUPPORTED_TASKS}, got: {task}")

        start_time = time.time()
        frame_uri = input_data.get("frame", {}).get("uri", "")
        image_path = self._resolve_frame_uri(frame_uri)
        image = Image.open(image_path).convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated_ids = self._blip_model.generate(**inputs, max_new_tokens=50)

        caption = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return {
            "task": task,
            "caption": caption,
            "model_id": self._model_id,
            "executed_at": int(time.time() * 1000),
            "latency_ms": int((time.time() - start_time) * 1000),
        }

    def _resolve_frame_uri(self, uri: str) -> str:
        if not uri.startswith("opennvr://frames/"):
            raise ValueError("Invalid frame URI: only opennvr://frames/ URIs are supported")

        relative_path = uri[len("opennvr://frames/") :]
        base = pathlib.Path(BASE_FRAMES_DIR).resolve()
        frame_path = (base / relative_path).resolve()

        try:
            frame_path.relative_to(base)
        except ValueError as exc:
            raise ValueError("Invalid frame URI: path traversal detected") from exc

        return str(frame_path)

    @property
    def schema(self) -> dict:
        return {
            "task": "scene_description",
            "description": "Generate natural language caption describing image content",
            "response_fields": {
                "task": {"type": "string", "description": "Task name identifier"},
                "caption": {"type": "string", "description": "Generated image caption"},
                "model_id": {"type": "string", "description": "BLIP model identifier"},
                "executed_at": {"type": "integer", "description": "Execution timestamp (ms)"},
                "latency_ms": {"type": "integer", "description": "Inference latency (ms)"},
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": self._model_id,
            "framework": "transformers",
            "tasks": ["scene_description"],
            "device": self._device or "not_loaded",
            "model_loaded": self._blip_model is not None,
        }
