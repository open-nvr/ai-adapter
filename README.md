# OpenNVR AI Adapter 🧠🔌

Welcome to the **OpenNVR Modular AI Adapter**! This repository is fundamentally designed as a dynamic, infinitely scalable engine that connects OpenNVR to cutting-edge AI models (Vision, LLMs, Audio, and Quantum).

We rebuilt this entirely around the **Clean Architecture** patterns to strip out massive dependencies and empower community contributions. 

## 📐 Architecture Design Patterns

To ensure this repository easily scales to hundreds of community adapters without becoming bloated, we have natively embraced three core Gang of Four design patterns:

1. **The Strategy Pattern (Core Router)**
   The `ModelRouter` delegates tasks based on the `task_name` routed from the config. It doesn't care *how* a person is calculated; it simply switches to whichever ML Algorithm (Strategy) matches the task.
2. **The Adapter Pattern (Base Interface)**
   Machine learning frameworks are chaotic (PyTorch vs ONNX vs Ultralytics). The `BaseAdapter` strictly encapsulates these frameworks into a single unified API (`load_model()` and `infer_local()`), meaning the FastAPI server is completely insulated from ML pipeline breaks.
3. **The Abstract Factory Pattern (Lazy Loading Registry)**
   During startup (`main.py`), the system uses `config.py` as a factory registry. Even if there are 500 adapters in the codebase, it will only instantiate the specific classes you enable—guaranteeing that unused ML models consume **zero** RAM, VRAM, or CPU.

## 🌟 Why OpenNVR AI? (Community First)
OpenNVR is built to **democratize AI for digital sovereignty**. You aren't locked into our models. The architecture provides a universal, language-agnostic interface (`BaseAdapter`) allowing anyone to plug in essentially *any* HuggingFace, PyTorch, ONNX, or REST-based model in under **30 minutes**.

## 🏎️ Lean Docker Containers
This primary Docker image is *incredibly lean (<100MB)*. Out of the box, it only installs `FastAPI` and its core web routing utilities.
It **will not install heavy gigabyte-sized ML libraries (like Torch/CUDA)** unless you specifically instruct it to!

```bash
# Start lean core (API routing, Mock models, Remote LLMs)
docker build -t opennvr-ai-adapter .

# Build with a thick Adapter's requirements (e.g. YOLO/Torch)
docker build --build-arg ADAPTER_REQ="vision/requirements-yolo.txt" -t opennvr-ai-adapter-heavy .
```

## ⏱️ The 30-Minute Adapter Guide (For Contributors)

Want to add a new model to the ecosystem? Follow these 3 incredibly simple steps:

### Step 1: Copy the Template
Take a look at `app/adapters/example_adapter.py`. 
Copy this file into `app/adapters/vision/` (or `llm/`):
```bash
cp app/adapters/example_adapter.py app/adapters/vision/super_detector_adapter.py
```

### Step 2: Inherit from `BaseAdapter`
Write a class that inherits from `BaseAdapter`. You only need to fulfill three simple requirements:

```python
from app.adapters.base import BaseAdapter
from typing import Any, Dict

class SuperDetectorAdapter(BaseAdapter):
    name: str = "super_detector"
    type: str = "vision"
    
    def load_model(self):
        # We enforce lazy loading! Do your heavy CUDA/ONNX loading here.
        # This will ONLY run if the adapter is literally triggered, saving massive VRAM memory.
        self.model = load_my_massive_model("weights.pt")
        
    def infer_local(self, input_data: Any) -> Dict[str, Any]:
        result = self.model.detect(input_data)
        return {"objects": result}
        
    def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "type": self.type}
```

### Step 3: Enable it in Config
Open `app/config/config.py` and activate your adapter!

```python
CONFIG = {
    "adapters": {
        "super_detector": {
            "enabled": True
        }
    },
    "routing": {
        "detect_objects": "super_detector"
    }
}
```

**That's it!** The `ModelRouter` and FastAPI endpoints automatically scrape your config, initialize the class, map the HTTP REST endpoints to your `infer_local` logic, and manage your memory lifecycle perfectly without crashing the host Node!

## Contributing
We welcome massive ML PRs and tiny configuration tweaks! Add your own models, expand the `PipelineEngine`, or submit entirely new LLM architectures. Be sure to include an adapter-specific `requirements-[name].txt` file so users can selectively build your module via the Docker pipeline!

For more detailed step-by-step instructions on making an adapter and interacting with the independent AI-Adapter FastAPI server, view our [CONTRIBUTING.md](CONTRIBUTING.md).
