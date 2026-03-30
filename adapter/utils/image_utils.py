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
Image processing utilities for the AI adapter.
"""
import os
import pathlib
import cv2
from fastapi import HTTPException
from ..config import BASE_FRAMES_DIR


def load_image_from_uri(uri: str):
    """
    Load an image from a OpenNVR URI.

    Args:
        uri: OpenNVR URI in format opennvr://frames/<camera_id>/<filename>

    Returns:
        Loaded image as numpy array

    Raises:
        HTTPException: If frame not found or invalid
    """
    if not uri.startswith("opennvr://frames/"):
        raise HTTPException(status_code=400, detail="Invalid frame URI")

    relative_path = uri[len("opennvr://frames/"):]
    base = pathlib.Path(BASE_FRAMES_DIR).resolve()
    frame_path = (base / relative_path).resolve()

    # Path containment check â€” prevent directory traversal outside BASE_FRAMES_DIR
    try:
        frame_path.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid frame URI")

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    img = cv2.imread(str(frame_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    return img


def validate_image(img):
    """
    Validate that an image is properly loaded.
    
    Args:
        img: Image to validate
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If image is invalid
    """
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return True
