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
OpenNVR Camera Runner with configurable task selection.

Usage:
    python runner.py --task person_detection
    python runner.py --task person_counting
    python runner.py --task person_detection --debug  # Show GUI
    python runner.py --help
"""
import cv2
import time
import httpx
import os
import argparse
import sys
import numpy as np


def get_available_tasks():
    """Query the adapter for available tasks."""
    try:
        r = httpx.get("http://127.0.0.1:9100/capabilities", timeout=5)
        return r.json()["tasks"]
    except Exception as e:
        print(f"WARNING: Could not fetch capabilities from adapter: {e}")
        print("WARNING: Make sure the adapter is running: uv run uvicorn app.main:app --reload --port 9100")
        return []


def run_camera(task: str, camera_id: int = 0, interval: float = 2.0, debug_gui: bool = False, rtsp_url: str = None):
    """
    Run camera capture and inference loop.
    
    Args:
        task: Task name(s) to run (e.g., "person_detection" or "person_detection,person_counting")
        camera_id: Camera device ID (default 0) - ignored if rtsp_url is provided
        interval: Seconds between captures (default 2.0)
        debug_gui: Show live GUI window with bounding boxes (default False)
        rtsp_url: RTSP stream URL (e.g., "rtsp://user:pass@ip:port/stream") - overrides camera_id
    """
    # Parse tasks (support multiple comma-separated tasks)
    tasks = [t.strip() for t in task.split(',')]
    
    # Determine camera source
    camera_source = rtsp_url if rtsp_url else camera_id
    camera_name = "rtsp" if rtsp_url else f"camera_{camera_id}"
    
    # Setup
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRAME_DIR = os.path.join(BASE_DIR, "..", "frames", camera_name)
    os.makedirs(FRAME_DIR, exist_ok=True)
    FRAME_PATH = os.path.join(FRAME_DIR, "latest.jpg")

    # Try to open camera/stream
    print(f"\nConnecting to {'RTSP stream' if rtsp_url else f'Camera {camera_id}'}...")
    if rtsp_url:
        print(f"RTSP URL: {rtsp_url}")
    
    cap = cv2.VideoCapture(camera_source)
    
    # Set RTSP-specific options for better performance
    if rtsp_url:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    ret, _ = cap.read()
    
    is_passive = False
    if not ret:
        print(f"\nWARNING: {'RTSP stream' if rtsp_url else f'Camera {camera_id}'} is unavailable or connection failed.")
        if rtsp_url:
            print("Tips for RTSP:")
            print("   - Check network connectivity")
            print("   - Verify credentials and IP address")
            print("   - Ensure camera supports RTSP on the specified port")
        print(f"Switching to PASSIVE MODE: Watching {FRAME_PATH} for updates from another runner...")
        is_passive = True
        cap.release()
    else:
        print(f"\nACTIVE MODE: {'Streaming from RTSP' if rtsp_url else f'Capturing from Camera {camera_id}'}")
        # Set camera properties for better performance
        if debug_gui:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    
    # Setup GUI window if debug mode enabled
    if debug_gui:
        source_name = "RTSP Stream" if rtsp_url else f"Camera {camera_id}"
        window_name = f"OpenNVR Debug - {source_name} - {', '.join(tasks)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        print("Debug GUI: Enabled (Live Stream Mode)")
    else:
        window_name = None
        print("Debug GUI: Disabled (use --debug to enable)")
    
    print(f"{'='*60}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Camera ID: {camera_id}")
    print(f"Frame Directory: {FRAME_DIR}")
    print(f"Interval: {interval}s")
    print(f"Adapter URL: http://127.0.0.1:9100/infer")
    print(f"{'='*60}\n")
    
    frame_count = 0
    last_file_mtime = 0
    current_annotated_frame = None  # Store annotated frame for GUI display
    last_detections = []  # Store latest detections
    last_count = 0
    inference_in_progress = False
    last_inference_time = 0
    
    try:
        while True:
            loop_start = time.time()
            
            # In debug GUI mode, show live video feed
            if debug_gui and window_name and not is_passive:
                ret, live_frame = cap.read()
                if ret:
                    # Show live frame with latest detections overlaid
                    if last_detections or last_count > 0:
                        display_frame = draw_debug_frame(
                            live_frame, 
                            last_detections, 
                            last_count if last_count > 0 else len(last_detections),
                            frame_count,
                            inference_in_progress
                        )
                    else:
                        display_frame = draw_debug_frame(
                            live_frame, 
                            [], 
                            0,
                            frame_count,
                            inference_in_progress
                        )
                    
                    cv2.imshow(window_name, display_frame)
                    
                    # Check for quit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nGUI closed by user (pressed 'q')")
                        break
            
            # Check if it's time to run inference
            if time.time() - last_inference_time < interval:
                continue
            
            inference_in_progress = True
            
            if not is_passive:
                # ACTIVE MODE: Capture and Save
                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Stream/Camera lost. Reconnecting...")
                    if rtsp_url:
                        # Try to reconnect to RTSP stream
                        cap.release()
                        time.sleep(2)
                        cap = cv2.VideoCapture(camera_source)
                        if rtsp_url:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FPS, 30)
                        continue
                    else:
                        break
                
                # Save frame
                cv2.imwrite(FRAME_PATH, frame)
            
            else:
                # PASSIVE MODE: Watch File
                if not os.path.exists(FRAME_PATH):
                    print(f"Waiting for {FRAME_PATH}...", end='\r')
                    time.sleep(1)
                    continue
                
                # Check if file updated
                try:
                    current_mtime = os.path.getmtime(FRAME_PATH)
                    if current_mtime <= last_file_mtime:
                        # File not changed yet, wait and retry
                        time.sleep(0.1)
                        continue
                    
                    last_file_mtime = current_mtime
                    # Allow writer to finish writing
                    time.sleep(0.1) 
                except OSError:
                    continue

            frame_count += 1
            
            # Read frame for GUI display
            display_frame = cv2.imread(FRAME_PATH)
            if display_frame is None:
                print("WARNING: Could not read frame for display")
                continue
            
            # Run all tasks on this frame
            print(f"\n[Frame {frame_count:04d}] [{'PASSIVE' if is_passive else 'ACTIVE'}] {'='*40}")
            
            # Store all detections for visualization
            all_detections = []
            total_count = 0
            
            for task_name in tasks:
                # Prepare payload
                payload = {
                    "task": task_name,
                    "input": {
                        "frame": {
                            "uri": f"opennvr://frames/{camera_name}/latest.jpg"
                        }
                    }
                }
                
                # Call adapter
                try:
                    start_time = time.time()
                    r = httpx.post(
                        "http://127.0.0.1:9100/infer",
                        json=payload,
                        timeout=120  # Allow time for first-time model loading (BLIP etc.)
                    )
                    elapsed = int((time.time() - start_time) * 1000)
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        # Format output based on task
                        if task_name == "person_detection":
                            conf = result.get("confidence", 0)
                            bbox = result.get("bbox", [0,0,0,0])
                            print(f"  OK  Detection | Conf: {conf:.2f} | BBox: {bbox} | Latency: {elapsed}ms")
                            
                            # Add to detections for GUI
                            if conf > 0:
                                all_detections.append({"bbox": bbox, "confidence": conf})
                        
                        elif task_name == "person_counting":
                            count = result.get("count", 0)
                            conf = result.get("confidence", 0)
                            total_count = count
                            print(f"  OK  Counting  | Count: {count} | Avg Conf: {conf:.2f} | Latency: {elapsed}ms")
                            
                            # Add all detections for GUI
                            detections_list = result.get('detections', [])
                            all_detections.extend(detections_list)
                            
                            if count > 0 and count <= 3:  # Only show details for small counts
                                for i, det in enumerate(detections_list[:3]):
                                    print(f"     Person {i+1}: Conf={det['confidence']}, BBox={det['bbox']}")
                        
                        elif task_name == "scene_description":
                            caption = result.get("caption", "")
                            print(f"  OK  Caption   | {caption} | Latency: {elapsed}ms")
                        
                        else:
                            print(f"  OK  {task_name}: {result}")
                    
                    else:
                        print(f"  ERROR {task_name} Error {r.status_code}: {r.json()}")
                    
                except httpx.TimeoutException:
                    print(f"  TIMEOUT {task_name} - adapter took too long")
                except httpx.ConnectError:
                    print(f"  ERROR {task_name} Connection Error - is adapter running?")
                except Exception as e:
                    print(f"  ERROR {task_name} Error: {e}")
            
            # Update stored detections for live overlay
            last_detections = all_detections
            last_count = total_count
            inference_in_progress = False
            last_inference_time = time.time()
            
            # In non-debug mode or passive mode, wait the full interval
            if not debug_gui or is_passive:
                time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("Runner stopped by user")
        print(f"Total frames processed: {frame_count}")
        print(f"{'='*60}\n")
    
    finally:
        cap.release()
        if debug_gui:
            cv2.destroyAllWindows()


def draw_debug_frame(frame: np.ndarray, detections: list, count: int, frame_num: int, processing: bool = False) -> np.ndarray:
    """
    Draw bounding boxes and info overlay on frame for debug GUI.
    
    Args:
        frame: Original frame
        detections: List of detection dicts with 'bbox' and 'confidence'
        count: Total count of detections
        frame_num: Current frame number
        processing: Whether inference is currently running
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    # Color palette (BGR)
    COLORS = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (0, 255, 255),    # Yellow
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 165, 255),    # Orange
        (128, 0, 128),    # Purple
    ]
    
    # Draw each detection
    for i, det in enumerate(detections):
        bbox = det['bbox']
        conf = det['confidence']
        color = COLORS[i % len(COLORS)]
        
        # Extract bbox [x, y, w, h]
        x, y, w, h = bbox
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"Person {i+1}: {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Background for label
        cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w, y), color, -1)
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw info overlay at top
    overlay_height = 80
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    
    # Frame number
    cv2.putText(annotated, f"Frame: {frame_num:04d}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Count
    cv2.putText(annotated, f"Count: {count}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Instructions
    cv2.putText(annotated, "Press 'q' to quit", (annotated.shape[1] - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Processing indicator
    if processing:
        cv2.putText(annotated, "Processing...", (annotated.shape[1] - 250, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(annotated, "LIVE", (annotated.shape[1] - 250, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="OpenNVR Camera Runner - Run inference tasks on camera feed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py --task person_detection
  python runner.py --task person_counting --interval 1.0
  python runner.py --task person_detection --camera 1
  python runner.py --list-tasks
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        help="Task(s) to run. Use comma-separated for multiple (e.g., person_detection or person_detection,person_counting)"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0) - ignored if --rtsp is used"
    )
    
    parser.add_argument(
        "--rtsp",
        type=str,
        default=None,
        help='RTSP stream URL (e.g., "rtsp://user:pass@192.168.1.100:554/stream")'
    )
    
    parser.add_argument(
        "--interval", 
        type=float,
        default=2.0,
        help="Seconds between captures (default: 2.0)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show live GUI window with bounding boxes"
    )
    
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks from the adapter"
    )
    
    args = parser.parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nFetching available tasks from adapter...\n")
        tasks = get_available_tasks()
        if tasks:
            print("Available tasks:")
            for task in tasks:
                print(f"  - {task}")
            print(f"\nTo run a task: python runner.py --task {tasks[0]}")
        else:
            print("No tasks available or adapter not running.")
        print()
        return
    
    # Validate task is provided
    if not args.task:
        print("\nERROR: --task is required")
        print("Use --list-tasks to see available tasks")
        print("Example: python runner.py --task person_detection\n")
        parser.print_help()
        sys.exit(1)
    
    # Verify task is available
    available_tasks = get_available_tasks()
    if available_tasks and args.task not in available_tasks:
        print(f"\nWARNING: Task '{args.task}' not found in adapter capabilities")
        print(f"Available tasks: {available_tasks}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run camera
    run_camera(args.task, args.camera, args.interval, args.debug, args.rtsp)


if __name__ == "__main__":
    main()
