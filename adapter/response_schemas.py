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
Response schema definitions for AI Adapter models.

This module provides schema definitions that describe the structure
of responses returned by each model/task.
"""

from typing import Dict, Any, List


def get_person_detection_schema() -> Dict[str, Any]:
    """
    Returns the response schema for person_detection task.
    
    Returns:
        Dictionary describing the response structure with field types and descriptions
    """
    return {
        "task": "person_detection",
        "description": "Detects the person with highest confidence in the image",
        "response_fields": {
            "label": {
                "type": "string",
                "description": "Object class label (always 'person')",
                "example": "person"
            },
            "confidence": {
                "type": "float",
                "description": "Confidence score of the best detection (0.0 to 1.0)",
                "example": 0.85
            },
            "bbox": {
                "type": "array[int]",
                "description": "Bounding box in [left, top, width, height] format",
                "example": [100, 150, 200, 300]
            },
            "annotated_image_uri": {
                "type": "string",
                "description": "URI to the annotated image with bounding box drawn",
                "example": "opennvr://frames/camera_0/person_detection_annotated.jpg",
                "optional": True
            },
            "executed_at": {
                "type": "integer",
                "description": "Timestamp in milliseconds when inference was executed",
                "example": 1735546430000
            },
            "latency_ms": {
                "type": "integer",
                "description": "Inference latency in milliseconds",
                "example": 150
            }
        },
        "example_response": {
            "label": "person",
            "confidence": 0.85,
            "bbox": [100, 150, 200, 300],
            "annotated_image_uri": "opennvr://frames/camera_0/person_detection_annotated.jpg",
            "executed_at": 1735546430000,
            "latency_ms": 150
        }
    }


def get_person_counting_schema() -> Dict[str, Any]:
    """
    Returns the response schema for person_counting task.
    
    Returns:
        Dictionary describing the response structure with field types and descriptions
    """
    return {
        "task": "person_counting",
        "description": "Counts all persons in the image with tracking IDs",
        "response_fields": {
            "task": {
                "type": "string",
                "description": "Task name identifier",
                "example": "person_counting"
            },
            "count": {
                "type": "integer",
                "description": "Total number of persons detected in the frame",
                "example": 3
            },
            "confidence": {
                "type": "float",
                "description": "Average confidence score across all detections (0.0 to 1.0)",
                "example": 0.82
            },
            "detections": {
                "type": "array[object]",
                "description": "List of all detected persons with bounding boxes and tracking IDs",
                "item_schema": {
                    "bbox": {
                        "type": "array[int]",
                        "description": "Bounding box [left, top, width, height]",
                        "example": [100, 150, 200, 300]
                    },
                    "confidence": {
                        "type": "float",
                        "description": "Detection confidence score (0.0 to 1.0)",
                        "example": 0.85
                    },
                    "track_id": {
                        "type": "integer",
                        "description": "Unique tracking ID for this person",
                        "example": 1
                    }
                },
                "example": [
                    {
                        "bbox": [100, 150, 200, 300],
                        "confidence": 0.85,
                        "track_id": 1
                    },
                    {
                        "bbox": [400, 200, 180, 280],
                        "confidence": 0.78,
                        "track_id": 2
                    }
                ]
            },
            "annotated_image_uri": {
                "type": "string",
                "description": "URI to the annotated image with all detections drawn",
                "example": "opennvr://frames/camera_0/person_counting_annotated.jpg",
                "optional": True
            },
            "executed_at": {
                "type": "integer",
                "description": "Timestamp in milliseconds when inference was executed",
                "example": 1735546430000
            },
            "latency_ms": {
                "type": "integer",
                "description": "Inference latency in milliseconds",
                "example": 180
            }
        },
        "example_response": {
            "task": "person_counting",
            "count": 3,
            "confidence": 0.82,
            "detections": [
                {
                    "bbox": [100, 150, 200, 300],
                    "confidence": 0.85,
                    "track_id": 1
                },
                {
                    "bbox": [400, 200, 180, 280],
                    "confidence": 0.78,
                    "track_id": 2
                },
                {
                    "bbox": [700, 100, 150, 250],
                    "confidence": 0.92,
                    "track_id": 3
                }
            ],
            "annotated_image_uri": "opennvr://frames/camera_0/person_counting_annotated.jpg",
            "executed_at": 1735546430000,
            "latency_ms": 180
        }
    }


def get_object_detection_detr_schema() -> Dict[str, Any]:
    """Returns the response schema for object_detection_detr task (HuggingFace DETR)."""
    return {
        "task": "object_detection_detr",
        "description": "Detects all objects in image using DETR model (80 COCO classes)",
        "response_fields": {
            "count": {"type": "integer", "description": "Number of objects detected"},
            "confidence": {"type": "float", "description": "Average confidence score"},
            "detections": {
                "type": "array[object]",
                "description": "List of detected objects",
                "item_schema": {
                    "label": {"type": "string", "description": "Object class (cat, car, person, etc.)"},
                    "confidence": {"type": "float", "description": "Detection confidence"},
                    "bbox": {"type": "array[int]", "description": "[x, y, width, height]"}
                }
            },
            "model_id": {"type": "string", "description": "HuggingFace model ID used"},
            "executed_at": {"type": "integer", "description": "Timestamp in ms"},
            "latency_ms": {"type": "integer", "description": "Inference latency in ms"}
        },
        "example_response": {
            "task": "object_detection_detr",
            "count": 3,
            "confidence": 0.85,
            "detections": [
                {"label": "cat", "confidence": 0.98, "bbox": [100, 150, 200, 180]},
                {"label": "couch", "confidence": 0.89, "bbox": [0, 200, 400, 250]}
            ],
            "model_id": "facebook/detr-resnet-50",
            "executed_at": 1735546430000,
            "latency_ms": 2500
        }
    }


def get_image_classification_schema() -> Dict[str, Any]:
    """Returns the response schema for image_classification task (HuggingFace ViT)."""
    return {
        "task": "image_classification",
        "description": "Classifies image into ImageNet categories using ViT model",
        "response_fields": {
            "top_label": {"type": "string", "description": "Top predicted label"},
            "top_confidence": {"type": "float", "description": "Confidence of top prediction"},
            "classifications": {
                "type": "array[object]",
                "description": "Top 5 classifications",
                "item_schema": {
                    "label": {"type": "string", "description": "Class label"},
                    "confidence": {"type": "float", "description": "Confidence score"}
                }
            },
            "model_id": {"type": "string", "description": "HuggingFace model ID used"},
            "executed_at": {"type": "integer", "description": "Timestamp in ms"},
            "latency_ms": {"type": "integer", "description": "Inference latency in ms"}
        },
        "example_response": {
            "task": "image_classification",
            "top_label": "tabby, tabby cat",
            "top_confidence": 0.62,
            "classifications": [
                {"label": "tabby, tabby cat", "confidence": 0.62},
                {"label": "tiger cat", "confidence": 0.18}
            ],
            "model_id": "google/vit-base-patch16-224",
            "executed_at": 1735546430000,
            "latency_ms": 1800
        }
    }


def get_scene_description_schema() -> Dict[str, Any]:
    """Returns the response schema for scene_description task (BLIP captioning)."""
    return {
        "task": "scene_description",
        "description": "Generates natural language caption describing image content",
        "response_fields": {
            "caption": {"type": "string", "description": "Generated text describing the image"},
            "model_id": {"type": "string", "description": "Model ID used"},
            "executed_at": {"type": "integer", "description": "Timestamp in ms"},
            "latency_ms": {"type": "integer", "description": "Inference latency in ms"}
        },
        "example_response": {
            "task": "scene_description",
            "caption": "a person sitting at a desk with a computer",
            "model_id": "Salesforce/blip-image-captioning-base",
            "executed_at": 1735546430000,
            "latency_ms": 2000
        }
    }


# Registry mapping task names to their schema functions
SCHEMA_REGISTRY = {
    "person_detection": get_person_detection_schema,
    "person_counting": get_person_counting_schema,
    "scene_description": get_scene_description_schema
}


def get_schema_for_task(task: str) -> Dict[str, Any]:
    """
    Get the response schema for a specific task.
    
    Args:
        task: Task name (e.g., "person_detection", "person_counting")
        
    Returns:
        Schema dictionary for the task
        
    Raises:
        KeyError: If task is not found in registry
    """
    if task not in SCHEMA_REGISTRY:
        raise KeyError(f"No schema defined for task: {task}")
    
    return SCHEMA_REGISTRY[task]()


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get schemas for all available tasks.
    
    Returns:
        Dictionary mapping task names to their schemas
    """
    return {
        task: schema_func()
        for task, schema_func in SCHEMA_REGISTRY.items()
    }
