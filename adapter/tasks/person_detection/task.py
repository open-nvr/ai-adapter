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
Person Detection Task Plugin

Detects the person with highest confidence in a camera frame.
Uses YOLOv8 (ONNX) as the underlying model.
"""
import os
from typing import Dict, Any

from adapter.interfaces import BaseTask
from adapter.models.yolov8_handler import YOLOv8Handler
from adapter.config import MODEL_CONFIGS


class Task(BaseTask):

    name = "person_detection"
    description = "Detect the person with highest confidence in the image"

    def setup(self):
        """Load YOLOv8 ONNX model."""
        config = MODEL_CONFIGS["yolov8n"]
        model_path = config["path"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOv8 model not found at {model_path}. "
                "Download it or set the correct path in config.py"
            )

        self._handler = YOLOv8Handler(model_path)

    def run(self, image, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run person detection.

        Args:
            image: Not used directly â€” handler reads from URI in params.
            params: Must contain {"frame": {"uri": "opennvr://..."}}

        Returns:
            {
                "label": "person",
                "confidence": 0.85,
                "bbox": [100, 150, 200, 300],
                "annotated_image_uri": "opennvr://...",
                "executed_at": 1735546430000,
                "latency_ms": 150,
            }
        """
        return self._handler.infer("person_detection", params)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": "yolov8n",
            "framework": "onnx",
            "device": "cpu",
            "tasks": [self.name],
        }
