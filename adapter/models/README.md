# adapter/models/ -- Model Handlers

This directory contains the neural network handler classes. Each handler wraps one ML model: it loads weights, preprocesses images, runs inference, and formats results.

## Files

```
models/
├── base_handler.py          # Abstract base class (BaseModelHandler)
├── yolov8_handler.py        # YOLOv8 Nano (ONNX) -- person detection
├── yolov11_handler.py       # YOLOv11 Medium (PyTorch) -- person counting + tracking
├── blip_handler.py          # BLIP (PyTorch) -- image captioning
├── insightface_handler.py   # InsightFace Buffalo-L (ONNX) -- face analysis
├── huggingface_handler.py   # Cloud inference via HuggingFace API
└── __init__.py              # Exports all handler classes
```

## base_handler.py

Defines `BaseModelHandler`, the abstract class all handlers extend. Forces every handler to implement:

- `get_supported_tasks()` -- return list of task names this handler can serve
- `infer(task, input_data)` -- run inference and return result dict

Also provides:
- `get_model_info()` -- return model metadata (has default, override for better info)
- `validate_task(task)` -- check if a task is supported (helper, not abstract)

## yolov8_handler.py

**Model:** YOLOv8 Nano | **Framework:** ONNX Runtime | **Task:** person_detection

Loads `yolov8n.onnx` from `model_weights/`. Uses adaptive confidence thresholds -- lower threshold for distant (small) people, higher for close (large) people. Includes NMS (Non-Maximum Suppression) to remove duplicate detections. Generates annotated images with bounding boxes.

**Key internals:**
- `_preprocess()` -- blob from image at 640x640
- `_run_inference()` -- ONNX session run
- `_filter_person_detections()` -- adaptive threshold by bbox size
- `_apply_nms()` -- removes overlapping boxes
- `_detect_persons()` -- full pipeline, returns result dict

## yolov11_handler.py

**Model:** YOLOv11 Medium | **Framework:** Ultralytics PyTorch | **Task:** person_counting

Uses the Ultralytics YOLO API with ByteTrack temporal tracking. Each detected person gets a persistent tracking ID across frames. Draws colorful bounding boxes (12-color palette). Auto-downloads the weight file if missing.

**Key internals:**
- `_count_persons()` -- full pipeline with tracking
- Uses `model.track()` with `persist=True` to maintain IDs across frames
- Confidence: 0.42, IOU: 0.35

## blip_handler.py

**Model:** Salesforce/blip-image-captioning-base | **Framework:** HuggingFace Transformers | **Task:** scene_description

Lazy-loaded (only loads when first called, not at startup). Generates natural language descriptions of images. Auto-detects GPU availability.

**Key internals:**
- `_ensure_model_loaded()` -- lazy load check
- `_resolve_frame_uri()` -- converts `opennvr://` URIs to file paths

## insightface_handler.py

**Model:** Buffalo-L | **Framework:** ONNX (via InsightFace) | **Tasks:** face_detection, face_embedding, face_recognition, face_verify, watchlist_check

The most feature-rich handler. Lazy-loaded (~183MB). Serves 5 different tasks from one model. Uses `FaceDatabase` for recognition and watchlist features.

**Key internals:**
- `_detect_faces()` -- bounding boxes, landmarks, age, gender
- `_get_embedding()` -- 512-dim face vector
- `_recognize_face()` -- match against registered faces
- `_verify_faces()` -- compare two faces via cosine similarity
- `_check_watchlist()` -- match against watchlist category only
- `register_face()` -- add face to database

## huggingface_handler.py

**Model:** Any HuggingFace Hub model | **Framework:** HuggingFace Inference API | **Tasks:** 16+ cloud tasks

No local weights -- sends requests to HuggingFace's cloud API. Requires `HF_TOKEN` env var or per-request `api_token`. Has SSRF protection (blocks private IPs) and path-traversal prevention.

**Key internals:**
- `_call_hf_api()` -- routes to correct `InferenceClient` method based on task
- `_is_safe_url()` -- SSRF check for image URLs
- `_resolve_OpenNVR_uri()` -- local URI resolution with containment check

## Adding a New Handler

1. Create `adapter/models/your_handler.py`
2. Extend `BaseModelHandler`
3. Implement `get_supported_tasks()` and `infer()`
4. Add import to `__init__.py`
5. Add model config to `config.py`

See [../docs/MODELS.md](../../docs/MODELS.md) for full details.
