# 🧠 Models Library: Batteries Included & Infinite Expansion

The Open-NVR `AIAdapters` repository comes with a series of pre-configured, highly optimized Model Handlers. These handlers manage the heavy lifting: neural network loading, tensor preprocessing, inference, and memory cleanup.

Task plugins use these handlers so that developers don't have to rewrite ML code.

---

## 🔋 Batteries Included: Core Handlers

We ship with 5 core handlers out of the box. They cover 90% of standard NVR use cases.

| Handler Class | Framework | Weights | Included Tasks |
|---------------|-----------|---------|----------------|
| **`YOLOv8Handler`** | ONNX Runtime | 6MB (Local) | `person_detection` |
| **`YOLOv11Handler`** | PyTorch | 40MB (Local) | `person_counting` (with ByteTrack) |
| **`InsightFaceHandler`** | ONNX Runtime | 183MB (Local Download) | `face_detection`, `face_recognition`, `face_verify`, `face_embedding`, `watchlist_check` |
| **`BLIPHandler`** | PyTorch (Transformers)| Auto-Downloaded | `scene_description` (VLM) |
| **`HuggingFaceHandler`**| Cloud REST API | Zero | *16+ Auto-Generated tasks* |

---

## 🌍 The Hugging Face Gateway: Bring Your Own Model (BYOM)

**This is the superpower of Open-NVR.** 

The `HuggingFaceHandler` bypasses local hardware constraints entirely. By passing a task request through this handler, you instantly gain access to the **entire Hugging Face Hub**. 

You do not need to download weights or configure Docker GPUs to use world-class models!

**Supported Auto-Routed Tasks Include:**
* `image-classification`
* `object-detection`
* `image-to-text` (VQA & Captioning)
* `zero-shot-classification`
* *And a dozen more!*

You simply pass the `model_name` and your `api_token` in the POST request, and the Handler securely proxies the inference to Hugging Face's global edge network.

---

## ⚙️ Deep Dive: How the Local Edge Models Work

If you are running compute locally, here is exactly what our embedded edge models are doing:

### 🚀 YOLOv8 (High-Speed Person Detection)
- Runs at blistering speeds on ONNX architecture.
- Specifically trained or filtered for Class 0 (Persons).
- Utilizes adaptive confidence thresholds: it lowers the threshold for distant, tiny people, and raises it for people standing right in front of the camera, virtually eliminating false-positives!

### 🎯 YOLOv11 w/ ByteTrack (Persistent Counting)
- Loads via the Ultralytics PyTorch wrapper.
- Uses `bytetrack.yaml` to remember who is who across multiple video frames.
- This prevents the NVR from counting the same person 500 times if they stand still!

### 👤 InsightFace (Biometric Engine)
- A mammoth 5-in-1 handler. It takes ~180MB of memory and lazy-loads on first request to keep server boot times extremely fast.
- Generates 512-dimensional vector embeddings for faces.
- It interfaces directly with our local `face_db` module for instant, offline VIP or Watchlist recognition.

### 🖼️ BLIP (Vision-Language)
- A heavy PyTorch Transformer model that "looks" at a frame and describes it in plain English.
- Example output: *"A person dropping a suspicious package near a doorway."*
- Fully offline and privacy-respecting!

---

## 🛠️ Adding Your Own Custom Weights

Have your own `.onnx` or `.pt` file?
Building your own Model Handler is trivial. Extend our `BaseModelHandler`, map it in `config.py`, and your custom weights will instantly supercharge the ecosystem! See the [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) for a step-by-step tutorial.
