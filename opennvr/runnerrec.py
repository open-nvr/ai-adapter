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
OpenNVR Video File Runner with configurable task selection.

Reads frames from rec.mp4 video file instead of camera.

Usage:
    python runnerrec.py --task person_detection
    python runnerrec.py --task person_counting
    python runnerrec.py --help
"""
import cv2
import time
import httpx
import os
import argparse
import sys
import collections
from statistics import mode, StatisticsError
import numpy as np


def get_available_tasks():
    """Query the adapter for available tasks."""
    try:
        r = httpx.get("http://127.0.0.1:9100/capabilities", timeout=5)
        return r.json()["tasks"]
    except Exception as e:
        print(f"âš ï¸  Could not fetch capabilities from adapter: {e}")
        print("âš ï¸  Make sure the adapter is running: uvicorn adapter.main:app --reload --port 9100")
        return []


def run_video(task: str, video_path: str, camera_id: int = 0, interval: float = 0.5):
    """
    Run video file processing and inference loop.
    
    Args:
        task: Task name(s) to run (e.g., "person_detection" or "person_detection,person_counting")
        video_path: Path to video file (e.g., "rec.mp4")
        camera_id: Virtual camera ID for frame storage (default 0)
        interval: Seconds between frame processing (default 2.0)
    """
    # Parse tasks (support multiple comma-separated tasks)
    tasks = [t.strip() for t in task.split(',')]
    
    # Setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRAME_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frames", f"camera_{camera_id}"))
    os.makedirs(FRAME_DIR, exist_ok=True)
    FRAME_PATH = os.path.normpath(os.path.join(FRAME_DIR, "latest.jpg"))

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"\nâŒ Error: Video file not found: {video_path}")
        print(f"Please make sure 'rec.mp4' exists in the current directory or provide the full path.")
        sys.exit(1)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"\nâŒ Error: Could not open video file: {video_path}")
        sys.exit(1)
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame skip for interval-based processing
    frames_to_skip = int(fps * interval) if fps > 0 else 1
    estimated_frames_to_process = max(1, total_frames // frames_to_skip)
    
    print(f"\nðŸŽ¬ VIDEO MODE: Processing frames from {video_path}")
    print(f"{'='*60}")
    print(f"Video Duration: {duration:.2f}s")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Processing Interval: {interval}s (every {frames_to_skip} frames)")
    print(f"Estimated frames to process: ~{estimated_frames_to_process}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Virtual Camera ID: {camera_id}")
    print(f"Frame Directory: {FRAME_DIR}")
    print(f"Adapter URL: http://127.0.0.1:9100/infer")
    print(f"{'='*60}\n")
    
    
    frame_count = 0
    processed_count = 0
    
    # Temporal smoothing: Track counts over last 5 frames to reduce flickering
    count_history = collections.deque(maxlen=5)
    
    try:
        while True:
            # Read next frame from video
            ret, frame = cap.read()
            
            if not ret:
                # End of video reached - exit gracefully
                print(f"\nâœ… End of video reached. Processed {processed_count} frames successfully.")
                break
            
            frame_count += 1
            
            # Skip frames based on interval (only process every Nth frame)
            if frame_count % frames_to_skip != 0:
                continue
            
            # Save frame with verification
            write_success = cv2.imwrite(FRAME_PATH, frame)
            if write_success:
                file_size = os.path.getsize(FRAME_PATH)
                video_timestamp = frame_count / fps if fps > 0 else 0
                print(f"\nâœ… Frame {frame_count}/{total_frames} saved at {video_timestamp:.1f}s ({file_size} bytes)")
            else:
                print(f"\nâŒ Warning: Failed to save frame to {FRAME_PATH}")
                continue  # Skip inference if frame save failed
            
            processed_count += 1
            
            # Run all tasks on this frame
            print(f"\n[Frame {processed_count:04d}/{total_frames}] [VIDEO] {'='*40}")
            
            for task_name in tasks:
                # Prepare payload
                payload = {
                    "task": task_name,
                    "input": {
                        "frame": {
                            "uri": f"opennvr://frames/camera_{camera_id}/latest.jpg"
                        }
                    }
                }
                
                # Call adapter
                try:
                    start_time = time.time()
                    r = httpx.post(
                        "http://127.0.0.1:9100/infer",
                        json=payload,
                        timeout=30  # Increased for lazy loading
                    )
                    elapsed = int((time.time() - start_time) * 1000)
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        # Format output based on task
                        if task_name == "person_detection":
                            conf = result.get("confidence", 0)
                            bbox = result.get("bbox", [0,0,0,0])
                            print(f"  âœ… Detection | Conf: {conf:.2f} | BBox: {bbox} | Latency: {elapsed}ms")
                        
                        elif task_name == "person_counting":
                            count = result.get("count", 0)
                            conf = result.get("confidence", 0)
                            
                            # Add to history for temporal smoothing
                            count_history.append(count)
                            
                            # Calculate smoothed count (mode of last 5 frames)
                            try:
                                smoothed_count = mode(count_history)
                            except StatisticsError:
                                # If no unique mode, use current count
                                smoothed_count = count
                            
                            # Display both raw and smoothed counts
                            if len(count_history) >= 3:  # Show smoothed after 3 frames
                                print(f"  âœ… Counting  | Count: {smoothed_count} (smoothed) | Raw: {count} | Avg Conf: {conf:.2f} | Latency: {elapsed}ms")
                            else:
                                print(f"  âœ… Counting  | Count: {count} | Avg Conf: {conf:.2f} | Latency: {elapsed}ms")
                            
                            if count > 0 and count <= 3:  # Only show details for small counts
                                for i, det in enumerate(result.get('detections', [])[:3]):
                                    print(f"     Person {i+1}: Conf={det['confidence']}, BBox={det['bbox']}")
                            
                            # Show annotated image path if available
                            annotated_uri = result.get('annotated_image_uri')
                            if annotated_uri:
                                print(f"  ðŸŽ¨ Annotated image available at: {annotated_uri}")
                        
                        elif task_name == "scene_description":
                            caption = result.get("caption", "")
                            print(f"  âœ… Caption   | {caption} | Latency: {elapsed}ms")
                        
                        else:
                            print(f"  âœ… {task_name}: {result}")
                    
                    else:
                        print(f"  âŒ {task_name} Error {r.status_code}: {r.json()}")
                    
                except httpx.TimeoutException:
                    print(f"  â±ï¸  {task_name} Timeout - adapter took too long")
                except httpx.ConnectError:
                    print(f"  âŒ {task_name} Connection Error - is adapter running?")
                except Exception as e:
                    print(f"  âŒ {task_name} Error: {e}")
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print(f"ðŸ›‘ Video processing stopped by user")
        print(f"Processed frames: {processed_count}/{total_frames}")
        print(f"{'='*60}\n")
    
    finally:
        cap.release()
        print(f"\nâœ… Video processing complete. Total frames processed: {processed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenNVR Video File Runner - Run inference tasks on video file frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runnerrec.py --task person_detection --video rec.mp4
  python runnerrec.py --task person_counting --interval 1.0
  python runnerrec.py --task person_detection --camera 1
  python runnerrec.py --list-tasks
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="Task(s) to run. Use comma-separated for multiple (e.g., person_detection or person_detection,person_counting)"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default="rec.mp4",
        help="Path to video file (default: rec.mp4)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Virtual camera device ID for frame storage (default: 0)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between frame processing (default: 0.5)"
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks from the adapter"
    )
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nðŸ” Fetching available tasks from adapter...\n")
        tasks = get_available_tasks()
        if tasks:
            print("Available tasks:")
            for task in tasks:
                print(f"  - {task}")
            print(f"\nTo run a task: python runnerrec.py --task {tasks[0]} --video rec.mp4")
        else:
            print("No tasks available or adapter not running.")
        print()
        return
    
    # Validate task is provided
    if not args.task:
        print("\nâŒ Error: --task is required")
        print("Use --list-tasks to see available tasks")
        print("Example: python runnerrec.py --task person_detection --video rec.mp4\n")
        parser.print_help()
        sys.exit(1)
    
    # Verify task is available
    available_tasks = get_available_tasks()
    if available_tasks:
        # Split comma-separated tasks and validate each one
        requested_tasks = [t.strip() for t in args.task.split(',')]
        invalid_tasks = [t for t in requested_tasks if t not in available_tasks]
        
        if invalid_tasks:
            print(f"\nâš ï¸  Warning: Task(s) '{', '.join(invalid_tasks)}' not found in adapter capabilities")
            print(f"Available tasks: {available_tasks}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Run video processing
    run_video(args.task, args.video, args.camera, args.interval)


if __name__ == "__main__":
    main()
