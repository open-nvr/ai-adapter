# 🛠️ Plugin Development Guide: Build Your Own AI Adapter!

Welcome to the AIAdapters Developer Guide! 🎉 Whether you want to drop in the latest Hugging Face Vision-Language Model, a custom PyTorch network you trained yourself, or a lightning-fast ONNX model for edge computing, you are in the right place.

Open-NVR uses a **zero-friction plugin architecture**. You don't need to learn our massive backend codebase. You just create a tiny folder, drop in two files, and restart the server. Your new AI capability will magically appear in the NVR dashboard!

Here is exactly how you do it. 🚀

---

## 🏗️ What You'll Create

Every AI task in our system lives in its own isolated folder inside `adapter/tasks/`. All you need to create is:

```text
adapter/tasks/my_awesome_model/
├── task.py         # Your Python logic (required)
└── schema.json     # What your model outputs (required)
```

That's literally it. The engine auto-discovers your folder on boot.

---

## 🚀 Step 1: Create Your Plugin Folder

Pick a `snake_case` name that describes your amazing new task! Let's build a Fire Detector for this example.

```bash
mkdir adapter/tasks/fire_detection
```

## 📝 Step 2: Define Your Schema (`schema.json`)

The UI developers (who build the React dashboard) need to know what Data Shape your AI model spits out so they can draw bounding boxes or display text. 

Create `adapter/tasks/fire_detection/schema.json`:

```json
{
  "task": "fire_detection",
  "description": "Detects active flames in a camera stream to trigger emergency alerts.",
  "response_fields": {
    "label": {
      "type": "string",
      "description": "What was detected (e.g., 'fire')"
    },
    "confidence": {
      "type": "float",
      "description": "Detection confidence from 0.0 to 1.0"
    },
    "bbox": {
      "type": "array[int]",
      "description": "Bounding box array exactly like: [left, top, width, height]"
    }
  },
  "example_response": {
    "label": "fire",
    "confidence": 0.98,
    "bbox": [100, 150, 200, 300]
  }
}
```

## 🧠 Step 3: Write the Code (`task.py`)

This is where the magic happens. You just need to create a class named `Task` that inherits from our `BaseTask` contract.

Create `adapter/tasks/fire_detection/task.py`:

```python
from typing import Dict, Any
from adapter.interfaces import BaseTask

class Task(BaseTask):

    # 1. Define your task details
    name = "fire_detection"
    description = "Detects active flames in a camera stream."

    def setup(self):
        """
        Runs exactly ONE time when the server boots!
        Load your heavy Neural Networks into VRAM here so they are ready to go.
        """
        print(f"🔥 Booting up {self.name}...")
        
        # Example: Loading a PyTorch model
        # from ultralytics import YOLO
        # self.model = YOLO("fire_yolov8.pt")
        pass

    def run(self, image, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs every time KAI-C sends a frame to your model!
        
        Args:
            image: A numpy array (BGR format, directly from OpenCV)
            params: Metadata like the camera URI.

        Returns:
            A Dictionary that perfectly matches the schema.json you wrote!
        """
        
        # Example Inference:
        # results = self.model.predict(image)
        # return {
        #     "label": "fire",
        #     "confidence": float(results[0].boxes.conf[0]),
        #     "bbox": results[0].boxes.xywh[0].tolist(),
        # }

        # Placeholder return for now:
        return {"label": "fire", "confidence": 0.0, "bbox": [0, 0, 0, 0]}

    def get_model_info(self) -> Dict[str, Any]:
        """Tell the system what hardware you are running on!"""
        return {
            "model": "fire_vision_v1",
            "framework": "pytorch",
            "device": "gpu",
            "tasks": [self.name],
        }
```

## 🎉 Step 4: Add `__init__.py` and Restart!

Just drop an empty file named `__init__.py` inside your folder so Python knows it's a module:

```bash
touch adapter/tasks/fire_detection/__init__.py
```

Now, **restart the inference server**:

```bash
uvicorn adapter.main:app --reload --port 9100
```

Watch the terminal logs. You will see:
`✓ Successfully Loaded Plugin: fire_detection`

You can verify it instantly by calling the API:
```bash
curl http://localhost:9100/capabilities
```

---

## ⚡ Advanced Topics & API Details

If you're building a highly complex AI task, here is the full technical deep dive into the plugin architecture.

### The BaseTask Contract

Every task plugin extends `BaseTask`. Here's what you **must** and **can** implement:

**Required:**
| Attribute/Method | What it does |
|-----------------|-------------|
| `name` | Unique task identifier (string). Must match your folder name. |
| `description` | Human-readable description (string). |
| `run(image, params)` | Execute inference. Return a dict matching your schema. |
| `get_model_info()` | Return model metadata dict. |

**Optional (Override if needed):**
| Method | What it does | Default |
|--------|-------------|---------|
| `setup()` | Load models/weights at startup. | Does nothing. |
| `cleanup()` | Release GPU memory at shutdown. | Does nothing. |
| `schema()` | Return response schema. | Reads from `schema.json` in your task's folder. |
| `validate_params(params)` | Validate incoming params. | Returns params as-is. |

### Using an Existing Model Handler

If your task uses a model that already has a handler in the code (e.g., YOLO, BLIP, InsightFace), you don't need to write new ML loader code! Just wire the existing handler in `setup()`:

```python
def setup(self):
    from adapter.models.yolov8_handler import YOLOv8Handler
    from adapter.config import MODEL_CONFIGS

    config = MODEL_CONFIGS["yolov8n"]
    self._handler = YOLOv8Handler(config["path"])

def run(self, image, params):
    return self._handler.infer(self.name, params)
```

The `_handler` attribute automatically registers your model correctly!

### Writing a New Model Handler

If you're bringing an entirely new architecture (e.g., a custom transformer), you have two choices for integrating your weights:

**Choice A: Inline (small models)**
Load and run the model directly inside your `task.py`:

```python
def setup(self):
    import onnxruntime as ort
    self.session = ort.InferenceSession("model_weights/fire.onnx")

def run(self, image, params):
    blob = preprocess(image)
    output = self.session.run(None, {"input": blob})
    return postprocess(output)
```

**Choice B: Separate handler class (shared/complex models)**
Create a new file in `adapter/models/` extending `BaseModelHandler`:

```python
# adapter/models/fire_handler.py
from .base_handler import BaseModelHandler

class FireHandler(BaseModelHandler):
    def __init__(self, model_path):
        super().__init__(model_path)
        # Load model...

    def get_supported_tasks(self):
        return ["fire_detection"]

    def infer(self, task, input_data):
        # Run inference...
        return result

    def get_model_info(self):
        return {"model": "fire_v1", "framework": "onnx", "device": "cpu", "tasks": ["fire_detection"]}
```

Then register it in `adapter/models/__init__.py` and use it in your task's `setup()`.

### Testing Your Plugin

Before deploying your plugin, test it using our built-in Developer CLI tools:

#### Testing Without the Web Server (CLI)
```bash
# Check that your plugin loads successfully without crashing
python cli.py list-tasks

# View the schema your task generates
python cli.py schema fire_detection

# Run an immediate inference pass on a local image
python cli.py infer fire_detection path/to/test_image.jpg
```

#### Testing With the Live Server (HTTP)
```bash
curl -X POST http://localhost:9100/infer \
  -H "Content-Type: application/json" \
  -d '{
    "task": "fire_detection",
    "input": {
      "frame": {"uri": "opennvr://frames/camera_0/latest.jpg"}
    }
  }'
```

---

## ⚙️ Administrative Features

### Enabling/Disabling Tasks 

In `adapter/config.py`, the `ENABLED_TASKS` dictionary allows administrators to selectively turn off heavy tasks:

```python
ENABLED_TASKS = {
    "person_detection": True,
    "fire_detection": False,  # Disabled! Will NOT load even if the folder exists.
}
```
*Note: Any new plugin folder not explicitly listed in this dict defaults to **enabled**.*

### Adding Model Weights to Production

If your model requires `.onnx` or `.pt` weight files:
1. Place them in the `model_weights/` directory.
2. Reference them cleanly via `adapter/config.py` in the `MODEL_CONFIGS` dictionary.
3. Add the public download URL to the `download_models.py` script so that when other developers run Docker builds, your weights are fetched automatically!

---

## 🏆 Submitting Your Plugin to Open-NVR

We want the community to use your model! Once you have thoroughly tested your plugin locally:

1. **Update API_REFERENCE.md**: Scroll down to the bottom of the API docs and add your new `schema.json` example so frontend developers know how to use your data.
2. **Submit a Pull Request**: Push your new folder to GitHub and open a PR!
3. **Include Your Weights Script**: Guarantee you updated `download_models.py` in your PR so users can fetch your weights seamlessly.

Welcome to the team! We can't wait to see what you build. 🌍
