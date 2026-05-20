"""
YOLOv8 person/object-detection adapter — A2.2 reference implementation
of the AI Adapter Contract v1.

This package is the contract-compliant HTTP+WebSocket service that
wraps the legacy ``YOLOv8Adapter`` ONNX runner. It mirrors the shape
of ``adapters/piper/`` and adds:

* multipart with real binary image upload on /infer
* the full §6 WebSocket streaming protocol (inline frames)
* §5.1-normalized detection output (bbox in [0, 1] coordinates,
  class_id → COCO label translation)

Shared-memory fast path (§6.2 frame_ref) is intentionally NOT in v1 —
adapter advertises ``supports_shared_memory: false`` and the
handshake_ack falls back to websocket transport. A follow-up commit
will land shm support.

Run with:
    python -m uvicorn adapters.yolov8.main:app --host 0.0.0.0 --port 9002
"""
