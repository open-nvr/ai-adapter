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
Task plugins package — auto-discovered at startup.

Each subdirectory is a task plugin containing:
    task.py      — Task class extending BaseTask
    schema.json  — Response schema (optional, can also be in task.py)

To add a new task, create a new folder here with a task.py file.
See adapter/interfaces/task.py for the BaseTask contract.
"""
