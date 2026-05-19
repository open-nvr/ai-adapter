# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the YOLOv8 contract-service tests.

Stubs ``onnxruntime`` so tests run without real ONNX weights. The fake
session returns a deterministic 3-detection array: a high-confidence
"person" plus two low-confidence boxes that get filtered out at the
default confidence threshold.

Same flat-module pattern as ``tests/_piper_service_fixtures.py`` to
sidestep the namespace-package collision (``tests/adapters/``).
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def install_fake_onnxruntime() -> None:
    """Inject a stub ``onnxruntime`` module."""

    import numpy as np

    class _FakeInputMeta:
        def __init__(self, name: str = "images") -> None:
            self.name = name

    class _FakeInferenceSession:
        """
        Returns a fixed-shape prediction array. YOLOv8 raw output shape
        is (1, 84, 8400) for COCO-80 — we ship 3 predictions to exercise
        the confidence filter:
          - person (class 0) at high confidence → kept
          - cat (class 15) at confidence below default 0.25 → dropped
          - dog (class 16) at confidence near 0 → dropped
        """

        def __init__(self, *_args, **_kwargs) -> None:
            self._providers = ["CPUExecutionProvider"]
            # Shape (1, 84, 3): 4 box coords + 80 class scores per detection
            preds = np.zeros((1, 84, 3), dtype=np.float32)
            # detection 0: center (0.5, 0.5) normalized, size (0.3, 0.4); person score 0.92
            preds[0, 0, 0] = 0.5
            preds[0, 1, 0] = 0.5
            preds[0, 2, 0] = 0.3
            preds[0, 3, 0] = 0.4
            preds[0, 4 + 0, 0] = 0.92  # class_id 0 = person
            # detection 1: cat (class 15) below threshold
            preds[0, 4 + 15, 1] = 0.10
            # detection 2: dog (class 16) negligible
            preds[0, 4 + 16, 2] = 0.001
            self._preds = preds

        def get_inputs(self):
            return [_FakeInputMeta()]

        def get_providers(self):
            return list(self._providers)

        def run(self, _outputs, _inputs):
            return [self._preds]

    def _available_providers():
        return ["CPUExecutionProvider"]

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = _FakeInferenceSession
    module.get_available_providers = _available_providers
    sys.modules["onnxruntime"] = module


@pytest.fixture
def yolov8_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: tmp weights dir + fake onnxruntime + fake weights file."""
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()
    weights_path = weights_dir / "yolov8n.onnx"
    weights_path.write_bytes(b"YOLOV8_TEST_WEIGHTS_PAYLOAD")  # bytes for sha256 fingerprint

    import app.config.config as config_module
    monkeypatch.setattr(config_module, "MODEL_WEIGHTS_DIR", str(weights_dir))
    import app.config as config_pkg
    monkeypatch.setattr(config_pkg, "MODEL_WEIGHTS_DIR", str(weights_dir), raising=False)

    install_fake_onnxruntime()

    # Reload service modules so they pick up the new MODEL_WEIGHTS_DIR
    # and the freshly-stubbed onnxruntime.
    import app.adapters.vision.yolov8_adapter as yolo_module
    importlib.reload(yolo_module)

    return {"weights_dir": weights_dir, "weights_path": weights_path}


def _boot_app(yolov8_env, monkeypatch: pytest.MonkeyPatch):
    # Post-A2.3b the YOLOv8 adapter is just ``service`` + ``main``;
    # auth + metrics moved to ``opennvr_adapter_sdk``. We reload only
    # the two adapter-local modules so the patched MODEL_WEIGHTS_DIR /
    # stubbed onnxruntime take effect on this test's module-import.
    for mod_name in ("adapters.yolov8.service", "adapters.yolov8.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    import adapters.yolov8.service as service_module
    original_init = service_module.YoloV8Service.__init__

    def patched_init(self, weights_path: str | None = None):
        original_init(self, weights_path=str(yolov8_env["weights_path"]))

    monkeypatch.setattr(service_module.YoloV8Service, "__init__", patched_init)

    from fastapi.testclient import TestClient
    import adapters.yolov8.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def yolov8_app(yolov8_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(yolov8_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def yolov8_app_with_auth(yolov8_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(yolov8_environment, monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def sample_jpeg() -> bytes:
    """A tiny but valid JPEG image (64x64 black) for tests."""
    import cv2
    import numpy as np

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed"
    return bytes(buf.tobytes())
