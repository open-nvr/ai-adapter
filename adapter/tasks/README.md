# adapter/tasks/ -- Task Plugins

This is the **plugin folder**. Each subdirectory is a self-contained AI task that the system auto-discovers at startup.

## Current Tasks

```
tasks/
├── person_detection/    # Detect the person with highest confidence
│   ├── task.py          # Uses YOLOv8Handler (ONNX)
│   ├── schema.json      # Response: label, confidence, bbox
│   └── __init__.py
│
├── person_counting/     # Count all people with tracking IDs
│   ├── task.py          # Uses YOLOv11Handler (PyTorch + ByteTrack)
│   ├── schema.json      # Response: count, detections with track_id
│   └── __init__.py
│
├── scene_description/   # Generate text caption for the image
│   ├── task.py          # Uses BLIPHandler (lazy-loaded)
│   ├── schema.json      # Response: caption string
│   └── __init__.py
│
├── face_detection/      # Detect faces with landmarks, age, gender
│   ├── task.py          # Uses InsightFaceHandler
│   ├── schema.json      # Response: faces array with bbox, age, gender
│   └── __init__.py
│
└── face_recognition/    # Identify person from registered faces
    ├── task.py          # Uses InsightFaceHandler
    ├── schema.json      # Response: person_id, name, similarity
    └── __init__.py
```

## How a Task Plugin Works

Every task folder must contain a `task.py` file with a class named `Task` that extends `BaseTask`:

```python
from adapter.interfaces import BaseTask

class Task(BaseTask):
    name = "your_task_name"             # unique identifier
    description = "What this task does"  # human-readable

    def setup(self):
        # Load model (called once at startup)
        self._handler = SomeHandler(...)

    def run(self, image, params):
        # Run inference, return result dict
        return self._handler.infer(self.name, params)

    def get_model_info(self):
        return {"model": "...", "framework": "...", "device": "cpu", "tasks": [self.name]}
```

## How Discovery Works

At startup, `adapter/loader.py`:

1. Scans this directory for subdirectories
2. Imports `<folder>.task` and looks for a `Task` class
3. Verifies it extends `BaseTask`
4. Calls `Task()` which triggers `setup()` to load the model
5. Registers the task in `task_registry`

If a plugin fails to load (missing model, bad code), it's skipped and other plugins still load normally.

## How to Add a New Task

1. Create a new folder here: `mkdir tasks/fire_detection`
2. Add `task.py` with `class Task(BaseTask)` and required methods
3. Add `schema.json` with response field definitions
4. Restart the server

Full guide: [../docs/PLUGIN_DEVELOPMENT.md](../../docs/PLUGIN_DEVELOPMENT.md)

## What Each File Does

| File | Purpose |
|------|---------|
| `task.py` | Required. Contains `class Task(BaseTask)` with `setup()`, `run()`, `get_model_info()` |
| `schema.json` | Required. Defines response fields, types, descriptions, and example response |
| `__init__.py` | Optional. Makes the folder a proper Python package |

## How Tasks Use Model Handlers

Tasks are thin wrappers. The actual ML work is done by handlers in `adapter/models/`:

| Task | Handler | Model |
|------|---------|-------|
| `person_detection` | `YOLOv8Handler` | YOLOv8n ONNX |
| `person_counting` | `YOLOv11Handler` | YOLOv11m PyTorch |
| `scene_description` | `BLIPHandler` | BLIP base |
| `face_detection` | `InsightFaceHandler` | Buffalo-L |
| `face_recognition` | `InsightFaceHandler` | Buffalo-L |

Multiple tasks can share the same handler (e.g., face_detection and face_recognition both use InsightFaceHandler).
