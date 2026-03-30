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
Person Counting Task Plugin

Counts all persons in a camera frame using ByteTrack tracking.
Uses YOLOv11 (Medium) as the underlying model.
"""
from typing import Dict, Any

from adapter.interfaces import BaseTask
from adapter.models.yolov11_handler import YOLOv11Handler


class Task(BaseTask):

    name = "person_counting"
    description = "Count all persons in the image with tracking IDs"

    def setup(self):
        """Load YOLOv11 model with ByteTrack tracker."""
        self._handler = YOLOv11Handler()

    def run(self, image, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run person counting with tracking.

        Args:
            image: Not used directly â€” handler reads from URI in params.
            params: Must contain {"frame": {"uri": "opennvr://..."}}

        Returns:
            {
                "task": "person_counting",
                "count": 3,
                "confidence": 0.82,
                "detections": [...],
                "annotated_image_uri": "opennvr://...",
                "executed_at": ...,
                "latency_ms": ...,
            }
        """
        return self._handler.infer("person_counting", params)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": "yolo11m",
            "framework": "pytorch",
            "device": "cpu",
            "tasks": [self.name],
        }
