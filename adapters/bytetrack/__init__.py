"""
ByteTrack multi-object tracking adapter — reference implementation
of the AI Adapter Contract v1.

Takes a frame's detections in and returns the same detections with
``track_id`` populated. Stateful per-camera: each camera has its own
ByteTrack instance so tracks don't bleed across cameras.

This is the canonical *post-processing* adapter — it doesn't run a
model itself, it composes with an upstream detector (YOLOv8, YOLOv11,
fast-plate-ocr, or any other detection-style adapter) by chaining
through KAI-C. The license-plate-recognition example demonstrates the
chaining pattern; ByteTrack slots into the same shape.

Run with:
    python -m uvicorn adapters.bytetrack.main:app --host 0.0.0.0 --port 9007
"""
