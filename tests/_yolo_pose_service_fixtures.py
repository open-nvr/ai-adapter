# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the YOLO-pose contract-service tests.

Stubs ``onnxruntime`` so the tests run without real ONNX weights (and
without a network round-trip to fetch them). The fake session returns
a deterministic 3-prediction array in the YOLO11-pose layout — 56
features per prediction: 4 box + 1 person score + 17 × (x, y, conf) —
chosen to exercise every branch of the post-processor at once:

  * prediction 0 — a strong person (score 0.92) at the centre of the
    model canvas, with a high-confidence left wrist and a
    low-confidence right wrist so the per-joint visibility metric has
    something to distinguish.
  * prediction 1 — a near-duplicate of 0 (score 0.80) that NMS must
    suppress at the default IoU, and must NOT suppress at iou=1.0.
  * prediction 2 — a small, distant person (score 0.30) that falls
    below the default conf of 0.4 and reappears when the caller lowers
    it.

The numbers are picked so the arithmetic is checkable by hand: the
nose of prediction 0 sits at the exact centre of the model canvas, so
after un-letterboxing it must land at the exact centre of the source
frame whatever that frame's aspect ratio is.

Same flat-module pattern as ``tests/_yolov8_service_fixtures.py`` to
sidestep the namespace-package collision (``tests/adapters/``).
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

#: Model canvas the fake predictions are expressed in — matches
#: ``service.DEFAULT_IMGSZ``. The fake session ignores the blob it is
#: handed (it has no weights), so the tests must feed the service its
#: default imgsz for the coordinates below to mean anything.
FAKE_IMGSZ: int = 448

#: Slot indices under test, from the COCO-17 ordering.
NOSE: int = 0
LEFT_WRIST: int = 9
RIGHT_WRIST: int = 10

#: Confidence assigned to every keypoint that isn't called out below.
#: Above service.KEYPOINT_VISIBLE_CONF, so the "visible" metric counts
#: it.
FILLER_KEYPOINT_CONF: float = 0.6
#: Right wrist is deliberately BELOW the visibility floor — occluded
#: joints are the normal case in a real frame and the response must
#: still carry all 17 slots.
RIGHT_WRIST_CONF: float = 0.2


def _pose_predictions(features: int = 56):
    """Build the (1, features, 3) array the fake session returns.

    ``features`` is a parameter so a test can hand the service a
    detection-shaped (84-feature) export and assert the typed
    "wrong model" failure.
    """
    import numpy as np

    preds = np.zeros((1, features, 3), dtype=np.float32)

    # ── prediction 0: strong person, centred on the model canvas ──
    preds[0, 0, 0] = 224.0   # cx
    preds[0, 1, 0] = 224.0   # cy
    preds[0, 2, 0] = 100.0   # w
    preds[0, 3, 0] = 200.0   # h
    preds[0, 4, 0] = 0.92    # person score

    # ── prediction 1: near-duplicate of 0, NMS fodder ─────────────
    preds[0, 0, 1] = 230.0
    preds[0, 1, 1] = 228.0
    preds[0, 2, 1] = 100.0
    preds[0, 3, 1] = 200.0
    preds[0, 4, 1] = 0.80

    # ── prediction 2: small distant person, below default conf ────
    preds[0, 0, 2] = 60.0
    preds[0, 1, 2] = 300.0
    preds[0, 2, 2] = 40.0
    preds[0, 3, 2] = 80.0
    preds[0, 4, 2] = 0.30

    if features < 56:
        return preds

    for slot in range(17):
        base = 5 + slot * 3
        # Filler joint for every prediction, overridden below for the
        # three slots the tests actually assert on.
        for pred in range(3):
            preds[0, base, pred] = 224.0
            preds[0, base + 1, pred] = 250.0
            preds[0, base + 2, pred] = FILLER_KEYPOINT_CONF

    # Prediction 0's diagnostic joints.
    preds[0, 5 + NOSE * 3, 0] = 224.0          # dead centre of the canvas
    preds[0, 5 + NOSE * 3 + 1, 0] = 224.0
    preds[0, 5 + NOSE * 3 + 2, 0] = 0.90
    preds[0, 5 + LEFT_WRIST * 3, 0] = 200.0
    preds[0, 5 + LEFT_WRIST * 3 + 1, 0] = 300.0
    preds[0, 5 + LEFT_WRIST * 3 + 2, 0] = 0.95
    preds[0, 5 + RIGHT_WRIST * 3, 0] = 250.0
    preds[0, 5 + RIGHT_WRIST * 3 + 1, 0] = 300.0
    preds[0, 5 + RIGHT_WRIST * 3 + 2, 0] = RIGHT_WRIST_CONF
    return preds


def install_fake_onnxruntime(
    features: int = 56,
    providers: list[str] | None = None,
    input_shape: list | None = None,
):
    """Inject a stub ``onnxruntime`` module.

    ``features`` lets a test simulate the wrong ONNX file being
    mounted; ``providers`` lets one simulate a CUDA build;
    ``input_shape`` lets one simulate a fixed-size export (the default
    is the dynamic-axes signature, where the spatial dims come back as
    symbolic names).
    """
    provider_list = list(providers or ["CPUExecutionProvider"])
    shape = list(input_shape or [1, 3, "height", "width"])

    class _FakeInputMeta:
        def __init__(self, name: str = "images") -> None:
            self.name = name
            self.shape = list(shape)

    class _FakeInferenceSession:
        """Returns the fixed prediction array above, and records the
        blobs it was handed so tests can assert on preprocessing
        (shape, dtype, value range) without a real model."""

        def __init__(self, *_args, **_kwargs) -> None:
            self._providers = provider_list
            self._preds = _pose_predictions(features)
            self.received_blobs: list = []

        def get_inputs(self):
            return [_FakeInputMeta()]

        def get_providers(self):
            return list(self._providers)

        def run(self, _outputs, inputs):
            self.received_blobs.append(next(iter(inputs.values())))
            return [self._preds]

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = _FakeInferenceSession
    module.get_available_providers = lambda: list(provider_list)
    sys.modules["onnxruntime"] = module
    return module


def _jpeg(width: int, height: int) -> bytes:
    """A real, decodable JPEG of the requested size. cv2 is a core
    project dep (the yolov8 tests decode with it too), so we use the
    real thing rather than stubbing the decode path."""
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed"
    return bytes(buf.tobytes())


@pytest.fixture
def yolo_pose_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: tmp weights dir + fake onnxruntime + fake weights
    file. ``YOLO_POSE_MODEL_URL`` is forced empty so a bug that lets
    ``ensure_model_file`` reach the network fails the test instead of
    downloading 12 MB."""
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    weights_path = weights_dir / "yolo11n-pose.onnx"
    # Real bytes so the sha256 fingerprint has something to hash.
    weights_path.write_bytes(b"YOLO11N_POSE_TEST_WEIGHTS_PAYLOAD")

    monkeypatch.setenv("YOLO_POSE_WEIGHTS_DIR", str(weights_dir))
    monkeypatch.setenv("YOLO_POSE_MODEL_URL", "")

    install_fake_onnxruntime()

    return {"weights_dir": weights_dir, "weights_path": weights_path}


def _boot_app(monkeypatch: pytest.MonkeyPatch):
    # Reload the two adapter-local modules so this test's patched
    # YOLO_POSE_WEIGHTS_DIR and stubbed onnxruntime take effect on
    # module import (``main`` probes the ORT providers at import time
    # to declare the gpu permission).
    for mod_name in ("adapters.yolo_pose.service", "adapters.yolo_pose.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.yolo_pose.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def yolo_pose_app(yolo_pose_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client


@pytest.fixture
def yolo_pose_app_with_auth(yolo_pose_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def service(yolo_pose_environment):
    """A freshly loaded ``YoloPoseService`` for the tests that drive
    the service directly instead of going through HTTP."""
    import adapters.yolo_pose.service as service_module
    importlib.reload(service_module)

    svc = service_module.YoloPoseService()
    svc.load()
    assert svc.is_ready(), f"YoloPoseService failed to load: {svc._load_error}"
    return svc


@pytest.fixture
def square_jpeg() -> bytes:
    """448×448 — the model canvas size, so model coordinates and
    pixel coordinates coincide and the expected output is readable by
    hand."""
    return _jpeg(FAKE_IMGSZ, FAKE_IMGSZ)


@pytest.fixture
def wide_jpeg() -> bytes:
    """640×360 (16:9) — the realistic camera aspect ratio, and the one
    that proves the letterbox padding is unmapped correctly."""
    return _jpeg(640, 360)
