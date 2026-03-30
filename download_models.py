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

#!/usr/bin/env python3
"""
Download required AI models on container startup if they don't exist.
This keeps the Docker image small while ensuring models are available.
"""
import os
import sys
import urllib.request
from pathlib import Path

# Model download URLs (using Ultralytics official releases)
MODEL_URLS = {
    "yolov8n.onnx": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    # Add more models as needed:
    # "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
}

def download_file(url: str, dest_path: Path):
    """Download a file with progress indicator."""
    print(f"📥 Downloading {dest_path.name}...")
    print(f"   URL: {url}")
    
    try:
        # Download with progress
        def reporthook(blocknum, blocksize, totalsize):
            if totalsize > 0:
                downloaded = blocknum * blocksize
                percent = min(downloaded * 100.0 / totalsize, 100)
                sys.stdout.write(f"\r   Progress: {percent:.1f}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n✅ Downloaded {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {dest_path.name}: {e}")
        return False

def main():
    """Download models if they don't exist."""
    model_weights_dir = Path("/app/model_weights")
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🔍 Checking for required models...")
    
    download_count = 0
    skip_count = 0
    fail_count = 0
    
    for filename, url in MODEL_URLS.items():
        model_path = model_weights_dir / filename
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            print(f"✓ {filename} already exists ({size_mb:.1f} MB)")
            skip_count += 1
        else:
            if filename.endswith(".onnx") and "yolo" in filename:
                print(f"Generating {filename} via Ultralytics export...")
                try:
                    from ultralytics import YOLO
                    import shutil
                    pt_name = filename.replace(".onnx", ".pt")
                    # YOLO will auto-download the .pt if missing
                    model = YOLO(pt_name)
                    exported_path = model.export(format="onnx")
                    if os.path.exists(exported_path):
                        shutil.move(exported_path, model_path)
                        if os.path.exists(pt_name):
                            os.remove(pt_name)
                        print(f"\n✅ Exported {filename} successfully")
                        download_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"\n❌ Failed to generate {filename}: {e}")
                    fail_count += 1
            else:
                if download_file(url, model_path):
                    download_count += 1
                else:
                    fail_count += 1
