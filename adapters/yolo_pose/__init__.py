"""
YOLO-pose human-keypoint adapter — implementation of the
AI Adapter Contract v1.

This package is the contract-compliant HTTP+WebSocket service around a
YOLO11n-pose ONNX model. It returns COCO-17 body keypoints per person
so downstream apps can reason about body GEOMETRY — the
wand-compliance app (is the guard's detector tracking the visitor's
arms?), fall detection, queue-orientation analytics.

It mirrors the shape of ``adapters/yolov8/`` — the canonical
``BodyShape.IMAGE`` adapter — and carries the same surface:

* multipart with real binary image upload on /infer
* the §6 WebSocket streaming protocol with inline frames — which is
  the path a 10 fps camera uses. Inline frames only: §6.2's
  shared-memory ``frame_ref`` and §6.3's NATS ``result_sink`` are both
  answered by downgrading to websocket in the handshake_ack.
* pixel-coordinate keypoint output plus ``frame_dimensions``, rather
  than §5.1's normalized detection shape (see ``service.py`` for why)

Shared-memory fast path (§6.2 frame_ref) is intentionally NOT here —
the adapter advertises ``supports_shared_memory: false`` and the
handshake_ack falls back to websocket transport, exactly as the
yolov8 adapter does.

Run with:
    python -m uvicorn adapters.yolo_pose.main:app --host 0.0.0.0 --port 9009
"""
