# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the package-detection contract-service tests.

Stubs ``onnxruntime`` so the tests run without the real ONNX weights
(and without a network fetch). The fake session returns a
deterministic 4-prediction array in the single-class YOLOv8 detection
layout — 5 features per prediction: cx, cy, w, h, package score —
chosen to exercise every branch of the post-processor at once:

  * prediction 0 — a strong parcel (0.91) centred on the model canvas,
    100×60 px. After un-letterboxing it must sit at the exact centre
    of the source frame whatever that frame's aspect ratio is.
  * prediction 1 — a near-duplicate of 0 (0.70) that NMS suppresses at
    the default IoU and must NOT suppress at iou=1.0.
  * prediction 2 — a small parcel at the top-left (0.55), a genuine
    second parcel that survives NMS: the count must be 2.
  * prediction 3 — a faint box (0.20) below the default conf of 0.35,
    which reappears when the caller lowers it.

Same flat-module pattern as ``tests/_yolo_pose_service_fixtures.py``.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

#: Model canvas the fake predictions are expressed in — matches
#: ``service.DEFAULT_IMGSZ``.
FAKE_IMGSZ: int = 416
#: Features of one row for a single-class detection export.
SINGLE_CLASS_FEATURES: int = 5


def _detection_predictions(features: int = SINGLE_CLASS_FEATURES):
    """Build the ``(1, features, 4)`` array the fake session returns.

    ``features`` lets a test hand the service a pose-shaped export
    (56) and assert the row is still parsed as 4 box + 52 "classes" —
    or a 3-feature nonsense export and assert the typed failure.
    """
    import numpy as np

    preds = np.zeros((1, features, 4), dtype=np.float32)
    if features < 5:
        return preds

    # prediction 0: strong parcel, centred.
    preds[0, 0, 0] = 208.0   # cx
    preds[0, 1, 0] = 208.0   # cy
    preds[0, 2, 0] = 100.0   # w
    preds[0, 3, 0] = 60.0    # h
    preds[0, 4, 0] = 0.91

    # prediction 1: near-duplicate of 0 — NMS fodder.
    preds[0, 0, 1] = 212.0
    preds[0, 1, 1] = 210.0
    preds[0, 2, 1] = 100.0
    preds[0, 3, 1] = 60.0
    preds[0, 4, 1] = 0.70

    # prediction 2: a second, smaller parcel at the top-left.
    preds[0, 0, 2] = 60.0
    preds[0, 1, 2] = 60.0
    preds[0, 2, 2] = 40.0
    preds[0, 3, 2] = 30.0
    preds[0, 4, 2] = 0.55

    # prediction 3: faint, below the default floor.
    preds[0, 0, 3] = 300.0
    preds[0, 1, 3] = 300.0
    preds[0, 2, 3] = 50.0
    preds[0, 3, 3] = 50.0
    preds[0, 4, 3] = 0.20
    return preds


def install_fake_onnxruntime(
    features: int = SINGLE_CLASS_FEATURES,
    providers: list[str] | None = None,
    input_shape: list | None = None,
):
    """Inject a stub ``onnxruntime`` module (see the pose fixture for
    the rationale behind each knob)."""
    provider_list = list(providers or ["CPUExecutionProvider"])
    shape = list(input_shape or [1, 3, "height", "width"])

    class _FakeInputMeta:
        def __init__(self, name: str = "images") -> None:
            self.name = name
            self.shape = list(shape)

    class _FakeInferenceSession:
        def __init__(self, *_args, **_kwargs) -> None:
            self._providers = provider_list
            self._preds = _detection_predictions(features)
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
    import cv2
    import numpy as np

    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed"
    return bytes(buf.tobytes())


@pytest.fixture
def package_detection_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: tmp weights dir + fake onnxruntime + fake weights
    file. The model URL is forced empty so a bug that lets
    ``ensure_model_file`` reach the network fails the test."""
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    weights_path = weights_dir / "yolov8n-package.onnx"
    weights_path.write_bytes(b"YOLOV8N_PACKAGE_TEST_WEIGHTS_PAYLOAD")

    monkeypatch.setenv("PACKAGE_DETECTION_WEIGHTS_DIR", str(weights_dir))
    monkeypatch.setenv("PACKAGE_DETECTION_MODEL_URL", "")
    monkeypatch.delenv("PACKAGE_DETECTION_LABELS", raising=False)

    install_fake_onnxruntime()

    return {"weights_dir": weights_dir, "weights_path": weights_path}


def _boot_app(monkeypatch: pytest.MonkeyPatch):
    for mod_name in ("adapters.package_detection.service", "adapters.package_detection.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.package_detection.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def package_detection_app(package_detection_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client


@pytest.fixture
def package_detection_app_with_auth(package_detection_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def service(package_detection_environment):
    """A freshly loaded ``PackageDetectionService`` for tests that drive
    the service directly instead of going through HTTP."""
    import adapters.package_detection.service as service_module
    importlib.reload(service_module)

    svc = service_module.PackageDetectionService()
    svc.load()
    assert svc.is_ready(), f"PackageDetectionService failed to load: {svc._load_error}"
    return svc


@pytest.fixture
def square_jpeg() -> bytes:
    """416×416 — model canvas size, so model and pixel coordinates
    coincide and the expected output is readable by hand."""
    return _jpeg(FAKE_IMGSZ, FAKE_IMGSZ)


@pytest.fixture
def wide_jpeg() -> bytes:
    """640×360 (16:9) — the realistic doorbell/porch aspect ratio, and
    the one that proves the letterbox padding is unmapped correctly."""
    return _jpeg(640, 360)
