# Contributing to OpenNVR AI Adapter

Welcome! The AI Adapter is designed as a standalone, loosely-coupled FastAPI microservice. It runs independently from the main OpenNVR server (Kai-C). Kai-C seamlessly communicates with this server over HTTP to route incoming multimedia streams into analytical tasks.

Because of our **Clean Architecture** mapping, adding a completely new AI model (PyTorch, ONNX, HuggingFace, etc.) takes **under 30 minutes**. You don't need to touch the FastAPI routing layer at all!

## 🚀 30-Minute Guide to Adding a New AI Model

Here is how to add a completely new handler without modifying any core plumbing.

### Step 1: Copy the Template
Take a look at `app/adapters/example_adapter.py`. 
Copy this file into the appropriate domain directory (`app/adapters/vision/` or `app/adapters/llm/`).
```bash
cp app/adapters/example_adapter.py app/adapters/vision/my_custom_adapter.py
```

### Step 2: Implement Your Logic
Open your new file. You only need to touch two methods:
1. **`load_model(self)`**: This runs *only once*. Load your model weights into RAM/VRAM here. (e.g., `self.model = torch.load(...)`). It is intentionally kept out of `__init__` to enforce Lazy Loading, keeping memory usage extremely low.
2. **`infer_local(self, input_data)`**: This runs on *every inference request*. Parse your incoming `input_data`, pass it to `self.model`, and return a clean dictionary containing your outputs (`boxes`, `captions`, `confidence` scores, etc.).

### Step 3: Register Your Adapter
Open `app/config/config.py` and register your new adapter in the `CONFIG` dictionary.

1. Add it to the `adapters` list to enable it:
```python
"adapters": {
    "my_custom_adapter": {
        "enabled": True,
        "weights_path": "path_to_my_weights.pt"
    }
}
```

2. Route a specific **task** from Kai-C to your adapter:
```python
"routing": {
    "custom_task": "my_custom_adapter"
}
```

**And you're done!** 
When Kai-C sends a `POST /infer` request to this FastAPI application asking to process `"task": "custom_task"`, it will dynamically locate `my_custom_adapter`, lazy-load its heavy PyTorch weights into memory instantly, and fire your `infer_local` logic.

## Developing & Testing
Run the FastAPI development server:
```bash
poetry run uvicorn app.main:app --reload
```
Test your new component instantly using `curl`:
```bash
curl -X POST http://localhost:8000/infer \
    -H "Content-Type: application/json" \
    -d '{"task": "custom_task", "data": {"frame": {"uri": "opennvr://test_video"}}}'
```
