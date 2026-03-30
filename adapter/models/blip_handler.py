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
BLIP Image Captioning Handler - Generate natural language descriptions of images.

This handler uses the Salesforce BLIP (Bootstrapping Language-Image Pre-training)
model to generate captions describing the content of images.

Model: Salesforce/blip-image-captioning-base
Task: scene_description - Generates text caption for image content
"""
import os
import pathlib
import time
import torch
from PIL import Image
from typing import List, Dict, Any
from transformers import BlipProcessor, BlipForConditionalGeneration

from .base_handler import BaseModelHandler
from ..config import BASE_FRAMES_DIR, MODEL_WEIGHTS_DIR


class BLIPHandler(BaseModelHandler):
    """
    Handler for BLIP image captioning model.
    
    Generates natural language descriptions of image content.
    Uses Salesforce/blip-image-captioning-base model.
    """
    
    def __init__(self):
        """
        Initialize BLIP handler.
        Models are loaded lazily on first use.
        """
        self.model_path = None
        self.session = None
        
        # Model components (loaded on first use)
        self.processor = None
        self.model = None
        self.device = None
        self.model_id = "Salesforce/blip-image-captioning-base"
        
        print(f"  âœ“ BLIPHandler initialized (model loads on first use)")
    
    def get_supported_tasks(self) -> List[str]:
        """Return list of tasks supported by this handler."""
        return ["scene_description"]

    def get_model_info(self) -> Dict[str, Any]:
        """Return BLIP model metadata."""
        return {
            "model": "blip-image-captioning-base",
            "framework": "pytorch",
            "device": self.device or "cpu",
            "tasks": self.get_supported_tasks(),
        }

    def _ensure_model_loaded(self):
        """Load model if not already loaded (lazy loading)."""
        if self.model is not None:
            return
        
        print(f"  ðŸ“¦ Loading BLIP model: {self.model_id}")
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  âœ“ BLIP model loaded on {self.device}")
    
    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference for scene description task.
        
        Args:
            task: Task name (should be "scene_description")
            input_data: Dictionary with frame URI
            
        Returns:
            Dictionary with generated caption
        """
        start_time = time.time()
        
        # Validate task
        if task != "scene_description":
            raise ValueError(f"Unknown task: {task}. BLIPHandler only supports 'scene_description'")
        
        # Ensure model is loaded
        self._ensure_model_loaded()
        
        # Get image path from frame URI
        frame_uri = input_data.get("frame", {}).get("uri", "")
        image_path = self._resolve_frame_uri(frame_uri)
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
        # Generate caption
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "task": task,
            "caption": caption,
            "model_id": self.model_id,
            "executed_at": int(time.time() * 1000),
            "latency_ms": latency_ms
        }
    
    def _resolve_frame_uri(self, uri: str) -> str:
        """
        Convert opennvr:// URI to actual file path.

        Example: opennvr://frames/camera_0/latest.jpg
              -> /app/frames/camera_0/latest.jpg

        Raises:
            ValueError: If the URI is invalid or attempts directory traversal
        """
        if not uri.startswith("opennvr://frames/"):
            raise ValueError("Invalid frame URI: only opennvr://frames/ URIs are supported")

        relative_path = uri[len("opennvr://frames/"):]
        base = pathlib.Path(BASE_FRAMES_DIR).resolve()
        frame_path = (base / relative_path).resolve()

        # Path containment check â€” prevent directory traversal
        try:
            frame_path.relative_to(base)
        except ValueError:
            raise ValueError("Invalid frame URI: path traversal detected")

        return str(frame_path)
