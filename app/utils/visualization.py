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
Visualization utilities for annotating images with bounding boxes.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional


# Define colorful palette (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
    (128, 0, 128),    # Purple
    (0, 165, 255),    # Orange
]


def draw_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict],
    count: Optional[int] = None,
    show_labels: bool = True,
    show_count: bool = True
) -> np.ndarray:
    """
    Draw colorful bounding boxes on image with detection information.
    
    Args:
        image: Original image as numpy array (BGR format)
        detections: List of detection dictionaries with 'bbox' and 'confidence'
                   bbox format: [left, top, width, height]
        count: Optional overall count to display in top-left corner
        show_labels: Whether to show individual labels on each box
        show_count: Whether to show overall count in corner
    
    Returns:
        Annotated image with bounding boxes drawn
    
    Example:
        >>> detections = [
        ...     {"bbox": [100, 50, 200, 400], "confidence": 0.87},
        ...     {"bbox": [350, 60, 180, 390], "confidence": 0.82}
        ... ]
        >>> annotated = draw_bounding_boxes(image, detections, count=2)
    """
    # Create a copy to avoid modifying original
    annotated = image.copy()
    
    # Draw bounding box for each detection
    for i, detection in enumerate(detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # Get color for this detection (cycle through palette)
        color = COLORS[i % len(COLORS)]
        
        # Extract bbox coordinates [left, top, width, height]
        x, y, w, h = bbox
        
        # Draw rectangle around detected object
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Draw label if enabled
        if show_labels:
            # Prepare label text
            label = f"Person {i+1}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw colored background for label
            cv2.rectangle(
                annotated,
                (x, y - text_h - 10),
                (x + text_w, y),
                color,
                -1  # Filled rectangle
            )
            
            # Draw white text on colored background
            cv2.putText(
                annotated,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
    
    # Draw overall count in top-left corner if enabled
    if show_count and count is not None:
        count_text = f"Count: {count}"
        cv2.putText(
            annotated,
            count_text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),  # Green
            3
        )
    
    return annotated
