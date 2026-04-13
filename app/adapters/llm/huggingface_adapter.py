# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hugging Face cloud inference adapter using lazy client initialization.
Ported from legacy HuggingFace handler into the adapter plugin architecture.
"""
import ipaddress
import logging
import os
import pathlib
import re
import socket
import time
from typing import Any, Dict

from app.adapters.base import BaseAdapter
from app.config import BASE_FRAMES_DIR

logger = logging.getLogger(__name__)

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-./]{1,200}$")


def _is_safe_url(url: str) -> bool:
    try:
        parsed_host = url.split("//", 1)[1].split("/")[0].split(":")[0]
        addr = ipaddress.ip_address(socket.gethostbyname(parsed_host))
        return not (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        )
    except Exception:
        return False


class HuggingFaceAdapter(BaseAdapter):
    name = "huggingface_adapter"
    type = "llm"

    SUPPORTED_TASKS = [
        "object-detection",
        "image-classification",
        "image-segmentation",
        "image-to-text",
        "text-generation",
        "text-classification",
        "token-classification",
        "question-answering",
        "summarization",
        "translation",
        "fill-mask",
        "zero-shot-classification",
        "automatic-speech-recognition",
        "audio-classification",
        "conversational",
        "feature-extraction",
        "zero_shot_detection",
        "hf_vision",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._client = None

    def load_model(self) -> None:
        from huggingface_hub import InferenceClient

        api_token = os.environ.get("HF_TOKEN")
        if api_token:
            self._client = InferenceClient(token=api_token)
            logger.info("HuggingFace InferenceClient initialized with API token")
        else:
            self._client = InferenceClient()
            logger.warning("HuggingFace client initialized without token; rate limits apply")

        self.model = self._client

    def infer_local(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()

        task = input_data.get("task", "object-detection")
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task '{task}' not supported by HuggingFace adapter")

        model_name = input_data.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for Hugging Face inference")
        if not _MODEL_NAME_RE.match(model_name):
            raise ValueError("Invalid model_name format")

        api_token = os.environ.get("HF_TOKEN") or input_data.get("api_token")
        if not api_token:
            raise ValueError(
                "HuggingFace API token required. Set HF_TOKEN environment variable or pass api_token in request."
            )

        inputs = input_data.get("inputs")
        if inputs is None:
            raise ValueError("inputs are required for Hugging Face inference")

        parameters = input_data.get("parameters", {})
        try:
            result = self._call_hf_api(
                model_name=model_name,
                inputs=inputs,
                parameters=parameters,
                api_token=api_token,
                task=task,
            )

            return {
                "task": task,
                "model_name": model_name,
                "result": result,
                "latency_ms": int((time.time() - start_time) * 1000),
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "success",
            }
        except Exception as exc:
            logger.error("Hugging Face inference failed: %s", exc, exc_info=True)
            return {
                "task": task,
                "model_name": model_name,
                "result": None,
                "error": str(exc),
                "latency_ms": int((time.time() - start_time) * 1000),
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "failed",
            }

    def _call_hf_api(
        self,
        model_name: str,
        inputs: Any,
        parameters: Dict[str, Any],
        api_token: str,
        task: str,
    ) -> Any:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=api_token)
        image_data = None

        if isinstance(inputs, dict) and "image" in inputs:
            image_input = inputs["image"]
            if isinstance(image_input, str) and image_input.startswith("opennvr://frames/"):
                image_data = self._resolve_frame_uri(image_input)
            elif isinstance(image_input, str) and (
                image_input.startswith("http://") or image_input.startswith("https://")
            ):
                if not _is_safe_url(image_input):
                    raise ValueError("Image URL resolves to a private/internal network address")
                image_data = image_input
            else:
                raise ValueError("Image input must be opennvr:// URI or public http(s) URL")
        elif isinstance(inputs, str):
            if inputs.startswith("opennvr://frames/"):
                image_data = self._resolve_frame_uri(inputs)
            elif inputs.startswith("http://") or inputs.startswith("https://"):
                if "image" in task or "object-detection" in task or "segmentation" in task:
                    if not _is_safe_url(inputs):
                        raise ValueError("Image URL resolves to a private/internal network address")
                    image_data = inputs

        if task == "object-detection":
            if image_data is None:
                raise ValueError("No image data provided for object detection task")
            return client.object_detection(image_data, model=model_name)
        if task in ["image-classification", "zero-shot-image-classification"]:
            if image_data is None:
                raise ValueError("No image data provided for classification task")
            return client.image_classification(image_data, model=model_name)
        if task in ["image-to-text", "image-captioning"]:
            if image_data is None:
                raise ValueError("No image data provided for captioning task")
            return client.image_to_text(image_data, model=model_name)
        if task == "image-segmentation":
            if image_data is None:
                raise ValueError("No image data provided for segmentation task")
            return client.image_segmentation(image_data, model=model_name)
        if task == "text-generation":
            text_input = inputs if isinstance(inputs, str) else inputs.get("text", "")
            return client.text_generation(text_input, model=model_name, **parameters)
        if task == "text-classification":
            text_input = inputs if isinstance(inputs, str) else inputs.get("text", "")
            return client.text_classification(text_input, model=model_name)
        if task == "question-answering":
            return client.question_answering(
                question=inputs.get("question", ""),
                context=inputs.get("context", ""),
                model=model_name,
            )

        return client.post(json={"inputs": inputs, "parameters": parameters}, model=model_name)

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

        if not frame_path.exists():
            raise ValueError("Frame not found")

        return str(frame_path)

    @property
    def schema(self) -> dict:
        return {
            "task": "huggingface_inference",
            "description": "Execute inference via Hugging Face cloud API",
            "supported_tasks": self.SUPPORTED_TASKS,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "model": "dynamic (set per-request via model_name)",
            "framework": "huggingface_hub",
            "tasks": self.SUPPORTED_TASKS,
            "inference_mode": "cloud",
            "model_loaded": self._client is not None,
        }
