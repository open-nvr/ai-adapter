# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
COCO class labels for YOLOv8.

YOLOv8n (and variants) are pre-trained on Microsoft COCO 2017 with 80
object classes. Mapping the integer ``class_id`` returned by the model
to a human-readable label is what lets §5.1's
``DetectionItem.label = "person"`` work; without it the contract just
sees an opaque number.

The list is the standard COCO-80 ordering used by Ultralytics and
matches what yolov8n.onnx outputs. Source:
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
"""
from __future__ import annotations

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

assert len(COCO_CLASSES) == 80, "COCO-80 must have exactly 80 classes"


def class_id_to_label(class_id: int) -> str:
    """Return the COCO label for an integer class id, or 'class_<id>'
    if the id is out of range (defensive — shouldn't happen for a
    correctly-trained model)."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"
