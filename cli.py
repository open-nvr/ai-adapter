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

#!/usr/bin/env python
"""
OpenNVR AI Adapter CLI
Developer tooling to test and interact with task plugins.

Usage:
    python cli.py list-tasks
    python cli.py schema <task_name>
    python cli.py infer <task_name> <uri_or_path>
"""

import argparse
import json
import sys
import os
import shutil
import cv2

# Ensure adapter is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapter.loader import load_tasks
from adapter.config import BASE_FRAMES_DIR


def list_tasks(args):
    print("Loading tasks...")
    tasks = load_tasks()
    print("\n--- Available Tasks ---")
    for name, task in tasks.items():
        print(f"\nTask: {name}")
        print(f"Description: {task.description}")
        print("Model Info: ")
        print(json.dumps(task.get_model_info(), indent=2))


def show_schema(args):
    tasks = load_tasks()
    task = tasks.get(args.task)
    if not task:
        print(f"Error: Task '{args.task}' not found.")
        sys.exit(1)
    
    print(f"\n--- Schema for {args.task} ---")
    print(json.dumps(task.schema(), indent=2))


def run_infer(args):
    tasks = load_tasks()
    task = tasks.get(args.task)
    if not task:
        print(f"Error: Task '{args.task}' not found.")
        sys.exit(1)
    
    uri = args.uri
    # If the user passed a local file path instead of a opennvr URI, let's copy it
    # to a temp location in the frames folder to mock the camera behavior.
    if not uri.startswith("opennvr://"):
        if not os.path.exists(uri):
            print(f"Error: File '{uri}' does not exist.")
            sys.exit(1)
        
        # Create a temp test directory
        test_dir = os.path.join(BASE_FRAMES_DIR, "cli_test")
        os.makedirs(test_dir, exist_ok=True)
        
        dest_path = os.path.join(test_dir, os.path.basename(uri))
        shutil.copy(uri, dest_path)
        uri = f"opennvr://frames/cli_test/{os.path.basename(uri)}"
        print(f"File copied for testing. Mocking URI: {uri}")
    
    params = {
        "frame": {"uri": uri}
    }
    
    print(f"\nRunning inference for '{args.task}'...")
    
    try:
        # Load image array just to pass it in case any modern task uses it (though most handlers pull via URI)
        img = None
        
        result = task.run(image=img, params=params)
        print("\n--- Inference Result ---")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\nInference failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="OpenNVR AI Adapter CLI Tooling")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # list-tasks
    list_parser = subparsers.add_parser("list-tasks", help="List all available tasks")
    list_parser.set_defaults(func=list_tasks)

    # schema
    schema_parser = subparsers.add_parser("schema", help="Show response schema for a task")
    schema_parser.add_argument("task", type=str, help="Name of the task")
    schema_parser.set_defaults(func=show_schema)

    # infer
    infer_parser = subparsers.add_parser("infer", help="Run inference for a task")
    infer_parser.add_argument("task", type=str, help="Name of the task")
    infer_parser.add_argument("uri", type=str, help="OpenNVR URI (e.g. opennvr://frames/...) or local file path")
    infer_parser.set_defaults(func=run_infer)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
