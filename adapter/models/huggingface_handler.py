# Copyright (c) 2026 OpenNVR
# This file is part of OpenNVR.
# 
# OpenNVR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# OpenNVR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with OpenNVR.  If not, see <https://www.gnu.org/licenses/>.

﻿"""
Hugging Face Cloud Handler - Executes inference via Hugging Face Inference API.

This handler supports multiple tasks by dynamically routing to HF API endpoints.
Unlike local model handlers, this delegates inference to Hugging Face's cloud service.
"""
import ipaddress
import logging
import os
import pathlib
import re
import socket
import time
import requests
from typing import List, Dict, Any
from huggingface_hub import InferenceClient
from pathlib import Path
from PIL import Image
import io

from .base_handler import BaseModelHandler

logger = logging.getLogger(__name__)

# Validated model_name pattern: org/repo or bare repo name, no shell metacharacters
_MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-./]{1,200}$')


def _is_safe_url(url: str) -> bool:
    """
    Return False if the URL resolves to a private/link-local/loopback address.
    Blocks SSRF attempts targeting internal network services.
    """
    try:
        parsed_host = url.split("//", 1)[1].split("/")[0].split(":")[0]
        addr = ipaddress.ip_address(socket.gethostbyname(parsed_host))
        return not (addr.is_private or addr.is_loopback or addr.is_link_local
                    or addr.is_reserved or addr.is_multicast)
    except Exception:
        return False


def _resolve_OpenNVR_uri(uri: str) -> str:
    """
    Convert a opennvr://frames/<...> URI to an absolute local file path.
    Validates path containment to prevent directory traversal.

    Raises:
        ValueError: If the URI is invalid or attempts directory traversal, or file not found.
    """
    from ..config import BASE_FRAMES_DIR

    if not uri.startswith("opennvr://frames/"):
        raise ValueError("Invalid frame URI: only opennvr://frames/ URIs are supported")

    relative_path = uri[len("opennvr://frames/"):]
    base = pathlib.Path(BASE_FRAMES_DIR).resolve()
    frame_path = (base / relative_path).resolve()

    try:
        frame_path.relative_to(base)
    except ValueError:
        raise ValueError("Invalid frame URI: path traversal detected")

    if not frame_path.exists():
        raise ValueError("Frame not found")

    return str(frame_path)


class HuggingFaceHandler(BaseModelHandler):
    """
    Handler for Hugging Face cloud-based AI models.
    
    Supports tasks like:
    - image-classification
    - object-detection
    - image-to-text (image captioning)
    - text-generation
    - And many more HF Inference API tasks
    """
    
    def __init__(self, model_path: str = "huggingface"):
        """
        Initialize Hugging Face handler.
        
        Args:
            model_path: Not used for cloud handler (kept for interface compatibility)
        """
        super().__init__(model_path)
        self.api_base_url = "https://api-inference.huggingface.co/models"
        self.session = None  # Not used for HTTP client
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return HuggingFace cloud handler metadata."""
        return {
            "model": "huggingface-cloud",
            "framework": "huggingface-api",
            "device": "cloud",
            "tasks": self.get_supported_tasks(),
        }

    def get_supported_tasks(self) -> List[str]:
        """
        Return list of supported Hugging Face tasks.
        
        Returns:
            List of HF task names that this handler can execute via API
        """
        return [
            "image-classification",
            "object-detection",
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
            "feature-extraction"
        ]
    
    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference via Hugging Face Inference API.
        
        Args:
            task: Task name (e.g., 'image-classification')
            input_data: Dictionary containing:
                - model_name: HF model identifier (e.g., 'google/vit-base-patch16-224')
                - inputs: Task-specific inputs (image URL, text, etc.)
                - parameters: Optional inference parameters
                - api_token: Hugging Face API token for authentication
        
        Returns:
            Unified response dictionary:
            {
                "task": str,
                "model_name": str,
                "result": any (task-specific result),
                "latency_ms": int,
                "executed_at": str (ISO timestamp)
            }
        """
        start_time = time.time()
        
        # Validate task
        if not self.validate_task(task):
            raise ValueError(f"Task '{task}' not supported by Hugging Face handler")
        
        # Extract required fields
        model_name = input_data.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for Hugging Face inference")
        if not _MODEL_NAME_RE.match(model_name):
            raise ValueError("Invalid model_name format")

        # API token: prefer server-side env var, fall back to per-request token
        api_token = os.environ.get("HF_TOKEN") or input_data.get("api_token")
        if not api_token:
            raise ValueError(
                "HuggingFace API token required. Set the HF_TOKEN environment variable "
                "on the server, or pass api_token in the request."
            )
        
        inputs = input_data.get("inputs")
        if inputs is None:
            raise ValueError("inputs are required for Hugging Face inference")
        
        parameters = input_data.get("parameters", {})
        
        # Call Hugging Face API
        try:
            result = self._call_hf_api(
                model_name=model_name,
                inputs=inputs,
                parameters=parameters,
                api_token=api_token,
                task=task
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "task": task,
                "model_name": model_name,
                "result": result,
                "latency_ms": latency_ms,
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "success"
            }
        
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Hugging Face inference failed: {e}", exc_info=True)
            
            return {
                "task": task,
                "model_name": model_name,
                "result": None,
                "error": str(e),
                "latency_ms": latency_ms,
                "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "failed"
            }
    
    def _call_hf_api(
        self,
        model_name: str,
        inputs: Any,
        parameters: Dict[str, Any],
        api_token: str,
        task: str
    ) -> Any:
        """
        Make request to Hugging Face Inference API using InferenceClient.
        
        Args:
            model_name: HF model identifier
            inputs: Input data (can be text, image URL, dict with image/text, binary data, etc.)
            parameters: Inference parameters
            api_token: HF API token
            task: Task name (for routing to correct API method)
        
        Returns:
            Raw API response (varies by task)
        
        Raises:
            Exception: If API request fails
        """
        # Create InferenceClient with token
        client = InferenceClient(token=api_token)
        
        # Prepare image data if needed
        image_data = None

        # Handle dict inputs (e.g., {"image": "url"} or {"image": "opennvr://..."})
        if isinstance(inputs, dict) and "image" in inputs:
            image_input = inputs["image"]

            # Handle opennvr:// URIs - resolve with path containment check
            if image_input.startswith("opennvr://frames/"):
                image_data = _resolve_OpenNVR_uri(image_input)

            # Handle HTTP(S) URLs - block SSRF to private/internal addresses
            elif image_input.startswith("http://") or image_input.startswith("https://"):
                if not _is_safe_url(image_input):
                    raise ValueError("Image URL resolves to a private/internal network address")
                image_data = image_input

            else:
                raise ValueError("Image input must be a opennvr:// URI or a public http(s):// URL")

        # Handle string inputs
        elif isinstance(inputs, str):
            # Handle opennvr:// URIs - resolve with path containment check
            if inputs.startswith("opennvr://frames/"):
                image_data = _resolve_OpenNVR_uri(inputs)

            # Handle HTTP(S) URLs for images
            elif inputs.startswith("http://") or inputs.startswith("https://"):
                if "image" in task or "object-detection" in task or "segmentation" in task:
                    if not _is_safe_url(inputs):
                        raise ValueError("Image URL resolves to a private/internal network address")
                    image_data = inputs
                # else: leave as text input for non-image tasks

            # For text tasks, keep as string (no-op)


        # Route to appropriate InferenceClient method based on task
        try:
            if task == "object-detection":
                if image_data is None:
                    raise ValueError("No image data provided for object detection task")
                logger.info(f"Calling HF object detection: model={model_name}, image_type={type(image_data)}")
                result = client.object_detection(image_data, model=model_name)
                logger.info(f"HF object detection successful: model={model_name}")
                return result
            
            elif task in ["image-classification", "zero-shot-image-classification"]:
                if image_data is None:
                    raise ValueError("No image data provided for classification task")
                result = client.image_classification(image_data, model=model_name)
                logger.info(f"HF image classification successful: model={model_name}")
                return result
            
            elif task in ["image-to-text", "image-captioning"]:
                if image_data is None:
                    raise ValueError("No image data provided for captioning task")
                result = client.image_to_text(image_data, model=model_name)
                logger.info(f"HF image-to-text successful: model={model_name}")
                return result
            
            elif task == "image-segmentation":
                if image_data is None:
                    raise ValueError("No image data provided for segmentation task")
                result = client.image_segmentation(image_data, model=model_name)
                logger.info(f"HF image segmentation successful: model={model_name}")
                return result
            
            elif task == "text-generation":
                text_input = inputs if isinstance(inputs, str) else inputs.get("text", "")
                result = client.text_generation(text_input, model=model_name, **parameters)
                logger.info(f"HF text generation successful: model={model_name}")
                return result
            
            elif task == "text-classification":
                text_input = inputs if isinstance(inputs, str) else inputs.get("text", "")
                result = client.text_classification(text_input, model=model_name)
                logger.info(f"HF text classification successful: model={model_name}")
                return result
            
            elif task == "question-answering":
                result = client.question_answering(
                    question=inputs.get("question", ""),
                    context=inputs.get("context", ""),
                    model=model_name
                )
                logger.info(f"HF question answering successful: model={model_name}")
                return result
            
            else:
                # Fallback: use post method for unsupported tasks
                logger.warning(f"Task '{task}' not directly supported, using generic post")
                result = client.post(
                    json={"inputs": inputs, "parameters": parameters},
                    model=model_name
                )
                return result
                
        except Exception as e:
            logger.error(f"HF API error: model={model_name}, task={task}, error={e}")
            raise
    
    def _download_image(self, image_url: str) -> bytes:
        """
        Download image from a public URL.

        Args:
            image_url: HTTP(S) URL of the image (must resolve to a public address)

        Returns:
            Image binary data

        Raises:
            ValueError: If the URL targets a private/internal address
            requests.HTTPError: If the download fails
        """
        if not _is_safe_url(image_url):
            raise ValueError("Image URL resolves to a private/internal network address")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content
