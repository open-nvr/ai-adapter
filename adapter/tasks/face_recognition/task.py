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

"""
Face Recognition Task Plugin

Identifies a person from registered faces using embedding comparison.
Uses InsightFace Buffalo-L as the underlying model.
"""
import os
from typing import Dict, Any

from adapter.interfaces import BaseTask
from adapter.models.insightface_handler import InsightFaceHandler
from adapter.config import MODEL_CONFIGS


class Task(BaseTask):

    name = "face_recognition"
    description = "Identify person from registered faces (1:N matching)"

    def setup(self):
        """Initialize InsightFace handler (model lazy-loads on first inference)."""
        config = MODEL_CONFIGS["insightface"]
        self._handler = InsightFaceHandler(config["path"])

    def run(self, image, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._handler.infer("face_recognition", params)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": "buffalo_l",
            "framework": "onnx",
            "device": "cpu",
            "tasks": [self.name],
        }
