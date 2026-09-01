# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the fast-plate-ocr contract-service tests.

Stubs out ``fast_plate_ocr`` so tests run without downloading the
real ONNX weights. The fake LicensePlateRecognizer returns
deterministic per-call output so the §5 wire shape has concrete
inputs to validate.

Same flat-module pattern as ``tests/_piper_service_fixtures.py`` and
``tests/_whisper_service_fixtures.py`` to sidestep the
``tests/adapters/`` namespace-collision.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path

import pytest

# Default fake output. Tests that need a specific return value set
# _FakeLicensePlateRecognizer.next_output before calling /infer.
_DEFAULT_FAKE_OUTPUT: tuple[str, float] = ("ABC1234", 0.93)


def install_fake_fast_plate_ocr(model_file_path: Path | None = None):
    """Inject a stub ``fast_plate_ocr`` module providing
    ``LicensePlateRecognizer``.

    Optional ``model_file_path`` is exposed on the fake recognizer as
    ``model_path`` so the service's fingerprint helper finds an on-disk
    file to hash — that exercises the live-fingerprint path. When
    omitted, fingerprint falls back to the identifier-derived
    synthetic.
    """

    class _FakeLicensePlateRecognizer:
        last_init_args: tuple = ()
        last_init_kwargs: dict = {}
        # Allow tests to mutate the next return value before /infer.
        next_output: tuple[str, float] = _DEFAULT_FAKE_OUTPUT

        def __init__(self, model_id, *args, **kwargs):
            _FakeLicensePlateRecognizer.last_init_args = (model_id,) + args
            _FakeLicensePlateRecognizer.last_init_kwargs = kwargs
            self.model_id = model_id
            if model_file_path is not None:
                # Service looks for ``model_path`` first; honour that.
                self.model_path = str(model_file_path)

        # Match the real LicensePlateRecognizer.run signature in
        # fast-plate-ocr 1.x: ``source`` is ndarray-or-path, NOT
        # bytes. The adapter decodes request bytes via cv2.imdecode
        # before invoking run(), so what arrives here is a numpy
        # array. We stash it for tests that want to assert on
        # input shape; the canned return value is independent.
        def run(self, source, *, return_confidence: bool = False):
            _FakeLicensePlateRecognizer.last_run_source = source
            text, conf = _FakeLicensePlateRecognizer.next_output
            if return_confidence:
                return [(text, conf)]
            return [text]

    module = types.ModuleType("fast_plate_ocr")
    module.LicensePlateRecognizer = _FakeLicensePlateRecognizer
    sys.modules["fast_plate_ocr"] = module
    return _FakeLicensePlateRecognizer


def install_fake_open_image_models():
    """Inject a stub ``open_image_models`` providing ``create_detector``.

    The fake detector's class-level ``next_detections`` is a list of
    ``(confidence, (x1, y1, x2, y2))`` tuples tests can set before an
    /infer call. Default: one high-confidence detection covering the
    WHOLE test frame — margins clamp, so the OCR input equals the full
    image and every pre-localization test keeps its exact expectations
    (caller's floor, (32, 64, 3) ndarray) while still exercising the
    detection-found code path.

    Set ``raise_on_create = True`` to simulate a detector that cannot
    load (the degraded mode)."""

    class _FakeBox:
        def __init__(self, x1, y1, x2, y2):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    class _FakeDetection:
        def __init__(self, conf, box):
            self.label = "License Plate"
            self.confidence = conf
            self.bounding_box = _FakeBox(*box)

    class _FakeDetector:
        next_detections: list = [(0.9, (0, 0, 64, 32))]
        last_predict_input = None
        last_create_kwargs: dict = {}
        raise_on_create = False

        def predict(self, image):
            _FakeDetector.last_predict_input = image
            return [_FakeDetection(c, b)
                    for c, b in _FakeDetector.next_detections]

    def create_detector(model, **kwargs):
        _FakeDetector.last_create_kwargs = {"model": model, **kwargs}
        if _FakeDetector.raise_on_create:
            raise RuntimeError("fake detector load failure")
        return _FakeDetector()

    module = types.ModuleType("open_image_models")
    module.create_detector = create_detector
    sys.modules["open_image_models"] = module
    return _FakeDetector


def _tiny_jpeg_bytes() -> bytes:
    """Generate a real-decodable JPEG via Pillow.

    The fake recognizer ignores image contents, but the service path
    now decodes the request body via ``cv2.imdecode`` BEFORE calling
    the recognizer (since fast-plate-ocr 1.x's API takes ndarrays,
    not bytes). A garbage JPEG-marker-prefix would fail the decode
    and return TRANSPORT_ERROR before the recognizer ever ran.
    Synthesising a tiny valid JPEG via Pillow keeps the decode path
    exercised and keeps the test focused on the recognizer flow.
    """
    from PIL import Image  # core project dep; opencv-python's installed too

    img = Image.new("RGB", (64, 32), (180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def fast_plate_ocr_environment(tmp_path: Path):
    """Sandboxed env: a tmp ONNX-shaped file the fake recognizer
    points at so the service's live-fingerprint path has something to
    hash, plus the fake fast_plate_ocr module installed."""
    model_file = tmp_path / "cct-xs-v1-global-model.onnx"
    # 4 KB of deterministic bytes so the sha256 is stable across test
    # runs. Real ONNX files are much larger; this is just enough for
    # the fingerprint helper to read.
    model_file.write_bytes(b"FAKE-FAST-PLATE-OCR-ONNX-" * 200)

    fake_cls = install_fake_fast_plate_ocr(model_file)
    fake_detector_cls = install_fake_open_image_models()
    # Reset class-level knobs so test order can't leak state.
    fake_detector_cls.next_detections = [(0.9, (0, 0, 64, 32))]
    fake_detector_cls.raise_on_create = False
    fake_detector_cls.last_predict_input = None

    return {
        "model_file": model_file,
        "fake_recognizer_cls": fake_cls,
        "fake_detector_cls": fake_detector_cls,
        "sample_jpeg": _tiny_jpeg_bytes(),
    }


def _boot_app(env, monkeypatch: pytest.MonkeyPatch):
    """Build a TestClient against the adapter app with the fake
    recognizer wired up. Reloads the service + main modules so the
    stubbed ``fast_plate_ocr`` import takes effect."""
    for mod_name in (
        "adapters.fast_plate_ocr.service",
        "adapters.fast_plate_ocr.main",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.fast_plate_ocr.main as main_module

    return TestClient(main_module.app), main_module


@pytest.fixture
def fast_plate_ocr_app(
    fast_plate_ocr_environment, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(fast_plate_ocr_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def fast_plate_ocr_app_with_auth(
    fast_plate_ocr_environment, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(fast_plate_ocr_environment, monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def sample_jpeg(fast_plate_ocr_environment) -> bytes:
    return fast_plate_ocr_environment["sample_jpeg"]
