# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The YOLOv8 decode path: letterbox geometry, vectorised NMS decode, and
the service's box shaping. Real numpy + cv2 (both are hard dependencies
of the adapter); no model, no onnxruntime."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.adapters.vision.yolov8_adapter import (
    LetterboxGeometry,
    box_to_source,
    decode_predictions,
    letterbox,
)
from app.config import INPUT_SIZE
from opennvr_adapter_sdk.contract import DetectionResult, InferResponse
from tests._yolov8_service_fixtures import (  # noqa: F401
    sample_jpeg,
    yolov8_app,
    yolov8_environment,
)


def _pred(cx, cy, w, h, class_id, score, nc=80):
    row = np.zeros(4 + nc, dtype=np.float32)
    row[:4] = (cx, cy, w, h)
    row[4 + class_id] = score
    return row


# ── letterbox ───────────────────────────────────────────────────────


def test_letterbox_preserves_aspect_and_pads_vertically():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    padded, geo = letterbox(img, INPUT_SIZE)
    assert padded.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert geo.scale == pytest.approx(1.0)
    assert geo.pad_x == 0
    assert geo.pad_y == pytest.approx((INPUT_SIZE - 360) / 2, abs=1)
    # The padding rows are grey, the image rows are black.
    assert padded[0, 0].tolist() == [114, 114, 114]
    assert padded[INPUT_SIZE // 2, INPUT_SIZE // 2].tolist() == [0, 0, 0]


def test_letterbox_square_input_is_a_plain_resize():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, geo = letterbox(img, INPUT_SIZE)
    assert geo.scale == pytest.approx(10.0)
    assert (geo.pad_x, geo.pad_y) == (0, 0)


# ── box_to_source ───────────────────────────────────────────────────


def test_box_to_source_undoes_letterbox_for_a_wide_frame():
    # 1920x1080 → scale 1/3, 640x360 inside 640x640, 140px top pad.
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, geo = letterbox(img, INPUT_SIZE)
    # A box spanning the whole visible image in model space.
    x1, y1, x2, y2 = box_to_source((0, geo.pad_y, 640, geo.pad_y + 360), geo)
    assert (x1, y1) == pytest.approx((0, 0), abs=1)
    assert (x2, y2) == pytest.approx((1920, 1080), abs=1)


def test_box_to_source_clips_to_the_frame():
    geo = LetterboxGeometry(scale=1.0, pad_x=0, pad_y=0, src_w=100, src_h=100, input_size=640)
    x1, y1, x2, y2 = box_to_source((-20, -20, 50, 50), geo)
    assert (x1, y1, x2, y2) == (0, 0, 50, 50)


def test_box_to_source_takes_model_pixels_at_face_value():
    # A sub-pixel box near the origin is a tiny box, not a whole-frame one.
    geo = LetterboxGeometry(scale=2.0, pad_x=0, pad_y=0, src_w=320, src_h=320, input_size=640)
    x1, y1, x2, y2 = box_to_source((0.0, 0.0, 0.9, 1.0), geo)
    assert (x1, y1, x2, y2) == pytest.approx((0, 0, 0.45, 0.5))


# ── decode_predictions ──────────────────────────────────────────────


def test_decode_suppresses_duplicate_boxes_of_one_object():
    """Three anchors on the same person → one detection, the strongest."""
    raw = np.stack([
        _pred(320, 320, 100, 200, 0, 0.90),
        _pred(322, 318, 102, 198, 0, 0.85),
        _pred(318, 321, 98, 203, 0, 0.60),
    ])
    boxes, class_ids, confs = decode_predictions(raw)
    assert len(boxes) == 1
    assert class_ids.tolist() == [0]
    assert confs[0] == pytest.approx(0.90)
    assert boxes[0].tolist() == pytest.approx([270, 220, 370, 420])


def test_decode_keeps_different_classes_on_the_same_spot():
    raw = np.stack([
        _pred(320, 320, 100, 200, 0, 0.90),   # person
        _pred(320, 320, 100, 200, 2, 0.80),   # car, same box — per-class NMS keeps it
    ])
    boxes, class_ids, confs = decode_predictions(raw)
    assert sorted(class_ids.tolist()) == [0, 2]


def test_decode_keeps_separate_objects():
    raw = np.stack([
        _pred(100, 100, 50, 50, 0, 0.9),
        _pred(500, 500, 50, 50, 0, 0.8),
    ])
    boxes, _, confs = decode_predictions(raw)
    assert len(boxes) == 2
    assert confs.tolist() == pytest.approx([0.9, 0.8])   # sorted descending


def test_decode_applies_confidence_threshold():
    raw = np.stack([_pred(100, 100, 50, 50, 0, 0.3), _pred(500, 500, 50, 50, 1, 0.2)])
    boxes, class_ids, _ = decode_predictions(raw, confidence_threshold=0.25)
    assert class_ids.tolist() == [0]
    boxes, _, _ = decode_predictions(raw, confidence_threshold=0.5)
    assert len(boxes) == 0


def test_decode_keeps_a_box_scoring_exactly_the_threshold():
    raw = np.stack([_pred(100, 100, 50, 50, 0, 0.25)])
    _, class_ids, confs = decode_predictions(raw, confidence_threshold=0.25)
    assert class_ids.tolist() == [0]
    assert confs[0] == pytest.approx(0.25)


def test_decode_handles_empty_and_malformed_output():
    for raw in (np.zeros((0, 84), np.float32), np.zeros((3,), np.float32), np.zeros((3, 2))):
        boxes, class_ids, confs = decode_predictions(raw)
        assert len(boxes) == len(class_ids) == len(confs) == 0


# ── service: boxes, NMS and params through /infer ───────────────────


def _install_predictions(app, rows: np.ndarray):
    """Swap the fake session's fixed output for ``rows`` (N, 84)."""
    from adapters.yolov8 import main as yolov8_main

    service = yolov8_main._service
    preds = rows.T[None, :, :].astype(np.float32)   # (1, 84, N)
    service._adapter.session._preds = preds


def test_infer_collapses_a_cluster_into_one_box(yolov8_app, sample_jpeg):
    _install_predictions(yolov8_app, np.stack([
        _pred(320, 320, 200, 300, 0, 0.9),
        _pred(325, 315, 205, 295, 0, 0.7),
        _pred(315, 325, 195, 305, 0, 0.5),
    ]))
    response = yolov8_app.post(
        "/infer", files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    infer = InferResponse.model_validate(response.json())
    result = DetectionResult.model_validate(infer.result)
    assert len(result.detections) == 1
    assert result.detections[0].confidence == pytest.approx(0.9)
    assert infer.result["raw_prediction_count"] == 3


def test_infer_iou_threshold_of_one_disables_nms(yolov8_app, sample_jpeg):
    _install_predictions(yolov8_app, np.stack([
        _pred(320, 320, 200, 300, 0, 0.9),
        _pred(325, 315, 205, 295, 0, 0.7),
    ]))
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        data={"params": json.dumps({"iou_threshold": 1.0})},
    )
    assert response.status_code == 200, response.text
    result = DetectionResult.model_validate(InferResponse.model_validate(response.json()).result)
    assert len(result.detections) == 2


def test_infer_accepts_the_sibling_adapters_spelling_of_iou(yolov8_app, sample_jpeg):
    _install_predictions(yolov8_app, np.stack([
        _pred(320, 320, 200, 300, 0, 0.9),
        _pred(325, 315, 205, 295, 0, 0.7),
    ]))
    for body in ({"iou": 1.0}, {"nms_threshold": 1.0}):
        response = yolov8_app.post(
            "/infer",
            files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
            data={"params": json.dumps(body)},
        )
        assert response.status_code == 200, response.text
        result = DetectionResult.model_validate(
            InferResponse.model_validate(response.json()).result)
        assert len(result.detections) == 2, body


@pytest.mark.parametrize("bad", ["abc", 1.5, -0.1, True])
def test_infer_rejects_bad_iou_threshold(yolov8_app, sample_jpeg, bad):
    response = yolov8_app.post(
        "/infer",
        files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
        data={"params": json.dumps({"iou_threshold": bad})},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_input"


def test_infer_box_at_the_frame_edge_is_cropped_not_shifted(yolov8_app, sample_jpeg):
    # The 64x64 fixture frame letterboxes to scale 10, no padding. A box
    # centred at model x=0 with width 200 spans -100..100 → source -10..10,
    # so the visible part is x∈[0, 10] of 64: x=0, w≈0.156. The old clamp
    # kept the full width (w≈0.31) and pushed the box inward.
    _install_predictions(yolov8_app, np.stack([_pred(0, 320, 200, 200, 0, 0.9)]))
    response = yolov8_app.post(
        "/infer", files={"frame": ("frame.jpg", sample_jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    result = DetectionResult.model_validate(InferResponse.model_validate(response.json()).result)
    assert len(result.detections) == 1
    bbox = result.detections[0].bbox
    assert bbox.x == 0.0
    assert bbox.w == pytest.approx(10 / 64, abs=0.02)
