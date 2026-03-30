# 🏛️ Architecture Overview: The AIAdapters Engine

Welcome to the architectural deep dive of Open-NVR's AI engine. This document explains how the system is designed, how data flows securely, and why we decoupled the AI from the core NVR.

## 🌟 The Big Picture: Task-Centric Plugins

AIAdapters uses a **Task-Centric Plugin Architecture**. Instead of hardcoding a massive, brittle monolith of neural networks, every AI capability (e.g., person detection, fire detection) is a minimal, self-contained "task plugin" that the system organically auto-discovers at startup.

This allows you to add complex Hugging Face models or custom PyTorch networks without ever touching the core server code!

```text
                         ┌─────────────────────────────────────────┐
                         │           AIAdapters Server             │
                         │                                         │
  Camera Frame           │   ┌──────────┐     ┌────────────────┐   │
  (opennvr://...)  ────> │   │  FastAPI │────>│  Task Registry │   │
                         │   │  Router  │     │                │   │
                         │   │ main.py  │     │ person_detect  │───┼──> YOLOv8Handler
                         │   │          │     │ face_recog     │───┼──> InsightFaceHandler
  JSON Result      <──── │   │  POST    │<────│ hf_vision      │───┼──> HuggingFaceHandler
                         │   │  /infer  │     └────────────────┘   │
                         │   └──────────┘                          │
                         └─────────────────────────────────────────┘
```

---

## 🚀 The Boot Sequence

When you run `uvicorn adapter.main:app`, the engine spring to life:

1. **FastAPI Initializes**: Web server boots up.
2. **Auto-Discovery**: `loader.load_tasks()` fires, scanning the `adapter/tasks/` directory.
3. **Plugin Registration**: Every folder containing a valid `Task` class is instantly mounted into memory.
4. **Model Loading**: Heavy GPU models are loaded into VRAM only once.
5. **Ready!**: The server opens port `9100` and awaits frames.

---

## ⚡ The Inference Flow

Because the Open-NVR core values zero-trust security, video frames never leave the camera VLAN unnecessarily. KAI-C (the connector) simply passes a unified URI reference to the AI engine.

Here is what happens when KAI-C requests an inference:

1. **The Request**: Client sends `POST /infer` with `{"task": "person_counting", "input": {"frame": {"uri": "opennvr://camera_0/latest.jpg"}}}`.
2. **The Route**: The engine looks up the `person_counting` task in the Task Registry.
3. **The Execution**: The associated Model Handler (e.g., YOLOv11) is triggered.
4. **The heavy lifting**:
   - The image is loaded securely from the `opennvr://` URI.
   - Preprocessed (resizing, tensor conversions).
   - Inferenced on the GPU/CPU.
   - Post-processed (bounding boxes, NMS).
5. **The Return**: A clean, structured JSON response is routed back to KAI-C.

---

## 🥞 The Three-Layer Separation

To ensure absolute modularity, the codebase is split into three strict layers:

### Layer 1: Task Plugins (`adapter/tasks/`)
- **Lightweight**: Just routing definitions and JSON schemas.
- **Rule**: One folder per specific capability.

### Layer 2: Model Handlers (`adapter/models/`)
- **Heavyweight**: Neural network operations, PyTorch/ONNX loading.
- **Rule**: One class per physical model. A single model (like InsightFace) can easily serve 5 different tasks!

### Layer 3: Core Utilities (`adapter/utils/`)
- **Shared Helpers**: URI decoding, bounding box drawing, file I/O.

**Why separate tasks from models?**
It means you can invent a brand-new task (e.g., "Hard Hat Detection") that utilizes an existing model handler (e.g., YOLO) **without writing any Machine Learning code!**

---

## ☁️ The Cloud Gateway (Hugging Face)

If a task requires massive compute resources (like generating text descriptions of a scene), the engine gracefully routes the request locally to the `HuggingFaceHandler`, which proxies the request to the cloud.

The cloud requests include the `model_name` and the `api_token`, letting you access over **100,000+ open-source models** instantly.

---

## 🗂️ File-by-File Reference

*Navigating the codebase like a pro:*

| File Profile | Purpose |
|--------------|---------|
| `adapter/main.py` | The main router and core FastAPI backend. |
| `adapter/config.py` | Your master control file (toggle tasks, set thresholds). |
| `adapter/loader.py` | The magical auto-discovery engine. |
| `adapter/interfaces/task.py` | The `BaseTask` contract for plugin developers. |
| `adapter/models/base_handler.py` | The abstract layout for writing new Model Handlers. |
| `adapter/utils/image_utils.py` | Securely reads images from `opennvr://` virtual paths. |
| `download_models.py` | Fetches massive weight files automatically for Docker builds. |

Ready to build your own piece of the architecture? Head over to the [Plugin Development Guide](PLUGIN_DEVELOPMENT.md)!
