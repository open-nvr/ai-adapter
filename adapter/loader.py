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
Dynamic Task Plugin Loader

Auto-discovers and loads task plugins from adapter/tasks/ at startup.
No manual imports or registration needed — just drop a folder and restart.

USAGE:
    from adapter.loader import load_tasks

    # Returns dict of {task_name: TaskInstance}
    tasks = load_tasks()

HOW IT WORKS:
    1. Scans adapter/tasks/ for subdirectories
    2. Each subdirectory must contain a task.py with a class named Task
    3. Task class must extend BaseTask (from adapter.interfaces)
    4. Only loads tasks that are enabled in config.ENABLED_TASKS
    5. Catches errors per-plugin so one broken plugin doesn't crash the server
"""

import importlib
import pkgutil
import os
from typing import Dict

from adapter.interfaces import BaseTask
from adapter.config import ENABLED_TASKS


# Path to the tasks package
_TASKS_PACKAGE = "adapter.tasks"
_TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


def load_tasks() -> Dict[str, BaseTask]:
    """
    Discover and instantiate all task plugins from adapter/tasks/.

    Returns:
        Dictionary mapping task name → BaseTask instance.
        Only includes tasks that are enabled in ENABLED_TASKS.

    Example:
        {
            "person_detection": <PersonDetectionTask>,
            "person_counting": <PersonCountingTask>,
            "scene_description": <SceneDescriptionTask>,
        }
    """
    tasks: Dict[str, BaseTask] = {}

    # Iterate over all sub-packages inside adapter/tasks/
    for importer, module_name, is_pkg in pkgutil.iter_modules([_TASKS_DIR]):
        if not is_pkg:
            # Only process directories (packages), skip loose .py files
            continue

        # Check if this task is enabled in config
        # If the task isn't listed in ENABLED_TASKS, default to True
        # (so new plugins work out-of-the-box without config changes)
        if not ENABLED_TASKS.get(module_name, True):
            print(f"  ⊘ Skipped: {module_name} (disabled in config)")
            continue

        try:
            # Import adapter.tasks.<module_name>.task
            task_module = importlib.import_module(
                f"{_TASKS_PACKAGE}.{module_name}.task"
            )

            # Get the Task class from the module
            task_class = getattr(task_module, "Task", None)
            if task_class is None:
                print(f"  ⚠ {module_name}/task.py has no 'Task' class — skipping")
                continue

            # Verify it extends BaseTask
            if not issubclass(task_class, BaseTask):
                print(f"  ⚠ {module_name}.Task does not extend BaseTask — skipping")
                continue

            # Instantiate the task (calls setup() internally)
            task_instance = task_class()
            tasks[task_instance.name] = task_instance
            print(f"  ✓ Loaded: {task_instance.name} ({task_class.__name__})")

        except FileNotFoundError as e:
            # Model file missing — expected in some deployments
            print(f"  ⊘ {module_name}: model not available ({e})")
        except Exception as e:
            # Log error but don't crash — other plugins should still load
            print(f"  ⚠ Failed to load {module_name}: {e}")

    return tasks
