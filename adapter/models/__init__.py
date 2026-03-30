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
Model handlers for AI adapter.
"""
from .base_handler import BaseModelHandler
from .yolov8_handler import YOLOv8Handler
from .yolov11_handler import YOLOv11Handler
from .blip_handler import BLIPHandler
from .insightface_handler import InsightFaceHandler
from .huggingface_handler import HuggingFaceHandler

__all__ = ["BaseModelHandler", "YOLOv8Handler", "YOLOv11Handler", "BLIPHandler", "InsightFaceHandler", "HuggingFaceHandler"]
