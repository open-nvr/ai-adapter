# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
COCO-17 keypoint names for YOLO11-pose.

YOLO11n-pose (and every other Ultralytics pose variant) is trained on
COCO Keypoints 2017, which annotates 17 body joints per person. The
model emits them as a flat ``[x, y, conf] * 17`` block in a FIXED
order — index 9 is always the left wrist, whatever the scene looks
like. Consumers therefore index by position, and this table is what
turns position 9 into the string ``"left_wrist"`` for the `keypoint_
names` field on every response (and for the per-joint domain metric
registered in ``load()``).

Same role ``adapters/yolov8/coco_classes.py`` plays for COCO-80
class ids: without it the contract output is a wall of anonymous
numbers.

The order below is the standard COCO ordering used by Ultralytics.
Source:
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml
"""
from __future__ import annotations

COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

assert len(COCO_KEYPOINTS) == 17, "COCO-17 must have exactly 17 keypoints"

#: Values per keypoint in the model's flat output block: x, y, conf.
KEYPOINT_STRIDE: int = 3

#: Bone list for consumers that want to draw a skeleton (pairs of
#: indices into ``COCO_KEYPOINTS``). Not part of the wire response —
#: an overlay renderer would otherwise have to hard-code it, and
#: hard-coding it twice is how left/right ends up mirrored.
COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms + shoulders
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
)


def keypoint_index_to_name(index: int) -> str:
    """Return the COCO keypoint name for a slot index, or
    ``'keypoint_<index>'`` if the index is out of range (defensive —
    shouldn't happen for a correctly-exported 17-keypoint model)."""
    if 0 <= index < len(COCO_KEYPOINTS):
        return COCO_KEYPOINTS[index]
    return f"keypoint_{index}"
