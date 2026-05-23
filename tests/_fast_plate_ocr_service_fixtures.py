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

        def run(self, image_bytes, *, return_confidence: bool = False):
            text, conf = _FakeLicensePlateRecognizer.next_output
            if return_confidence:
                return [(text, conf)]
            return [text]

    module = types.ModuleType("fast_plate_ocr")
    module.LicensePlateRecognizer = _FakeLicensePlateRecognizer
    sys.modules["fast_plate_ocr"] = module
    return _FakeLicensePlateRecognizer


def _tiny_jpeg_bytes() -> bytes:
    """Generate a small valid-ish JPEG byte string for tests.

    The fake recognizer ignores image contents, but request validation
    (multipart parsing, body-size checks) may inspect leading bytes,
    so we ship a well-formed JPEG marker prefix instead of garbage.
    """
    # JFIF header + minimal SOI/EOI markers. Not a renderable image,
    # but the bytes parse as a JPEG container far enough for HTTP
    # bodies to round-trip.
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00\x43\x00" + b"\x08" * 64
        + b"\xff\xd9"
    )


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

    return {
        "model_file": model_file,
        "fake_recognizer_cls": fake_cls,
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
