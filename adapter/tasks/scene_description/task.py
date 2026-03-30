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
Scene Description Task Plugin

Generates natural language captions describing image content.
Uses Salesforce BLIP as the underlying model.
"""
from typing import Dict, Any

from adapter.interfaces import BaseTask
from adapter.models.blip_handler import BLIPHandler


class Task(BaseTask):

    name = "scene_description"
    description = "Generate natural language caption describing image content"

    def setup(self):
        """Initialize BLIP handler (model lazy-loads on first inference)."""
        self._handler = BLIPHandler()

    def run(self, image, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run scene captioning.

        Args:
            image: Not used directly â€” handler reads from URI in params.
            params: Must contain {"frame": {"uri": "opennvr://..."}}

        Returns:
            {
                "task": "scene_description",
                "caption": "a person sitting at a desk ...",
                "model_id": "Salesforce/blip-image-captioning-base",
                "executed_at": ...,
                "latency_ms": ...,
            }
        """
        return self._handler.infer("scene_description", params)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": "blip-image-captioning-base",
            "framework": "pytorch",
            "device": "cpu",
            "tasks": [self.name],
        }
