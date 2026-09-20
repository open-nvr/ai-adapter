"""
Package-detection adapter — reference implementation of the AI
Adapter Contract v1 for the ``package_detection`` task.

Wraps a YOLOv8n fine-tuned on doorstep parcels (the MIT-licensed
*package at front door* set plus Roboflow's CC0 *packages* set) on
``onnxruntime``, and returns §5.1 detections with the single label
``package``. COCO, which the general object detector runs, has no
package class at all; this adapter is the skill that fills that gap for
apps such as ``package-delivery``, which asks KAI-C for the best
parcel-capable skill on the box and takes this one over VQA and over
the COCO bag-class stand-in.

Self-contained like ``adapters/yolo_pose/``: it owns its ONNX session
and post-processing, so the image contains the SDK and this package.

Run with:
    python -m uvicorn adapters.package_detection.main:app --host 0.0.0.0 --port 9010
"""
