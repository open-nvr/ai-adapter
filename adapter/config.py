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
Configuration for AI Adapter models and tasks.
"""
import os

# Base directory for the adapter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FRAMES_DIR = os.path.join(BASE_DIR, "..", "frames")
MODEL_WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "model_weights")

# Model configurations
MODEL_CONFIGS = {
    "yolov8n": {
        "path": os.path.join(MODEL_WEIGHTS_DIR, "yolov8n.onnx"),
        "handler_class": "YOLOv8Handler",
    },
    "yolov11m": {
        "path": os.path.join(MODEL_WEIGHTS_DIR, "yolo11m.pt"),
        "handler_class": "YOLOv11Handler",
    },
    "blip": {
        "path": "Salesforce/blip-image-captioning-base",  # HuggingFace model ID
        "handler_class": "BLIPHandler",
    },
    "insightface": {
        "path": os.path.join(MODEL_WEIGHTS_DIR, "insightface"),
        "handler_class": "InsightFaceHandler",
        "model_name": "buffalo_l",
    }
}

# Task enablement - control which tasks are active
# Set to False to disable a task without removing code
ENABLED_TASKS = {
    "person_detection": True, 
    "person_counting": True,
    "scene_description": True,  # BLIP image captioning
    # InsightFace tasks
    "face_detection": True,
    "face_embedding": True,
    "face_recognition": True,
    "face_verify": True,
    "watchlist_check": True,
}

# Model inference settings
CONFIDENCE_THRESHOLD = 0.20  # Balanced - detects distant people, fewer false positives
INPUT_SIZE = 640
