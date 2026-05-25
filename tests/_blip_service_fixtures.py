# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the SDK-based BLIP adapter tests.

Stubs ``transformers`` + ``torch`` so the tests don't pull the
~990 MB BLIP weights or the half-gigabyte torch/transformers wheels.
The fake processor + model return deterministic captions so the §5
result wire shape has concrete inputs to validate.

Same flat-module pattern as ``_insightface_service_fixtures.py`` —
keeps tests/ flat to dodge the tests/adapters/ namespace collision.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path

import pytest


_DEFAULT_FAKE_CAPTION: str = "a box on the porch"


def install_fake_transformers(caption: str = _DEFAULT_FAKE_CAPTION):
    """Inject stubs for ``transformers`` + ``torch`` so importing
    them in BlipService.load() succeeds without the real wheels."""

    # ── torch stub ────────────────────────────────────────────────
    torch = types.ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.cuda = _FakeCuda()  # type: ignore[attr-defined]

    class _FakeNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    torch.no_grad = lambda: _FakeNoGrad()  # type: ignore[attr-defined]

    sys.modules["torch"] = torch

    # ── transformers stub ─────────────────────────────────────────
    class _FakeInputs:
        """Stand-in for the processor output. The real shape is a
        dict-of-tensors (BatchEncoding from transformers); we only
        need ``.to()`` to no-op and ``**inputs`` unpacking in
        ``_run_blip`` to yield something ``generate()`` ignores.

        Note: ``keys()`` returns an empty list deliberately. Python's
        ``**`` unpacking semantics iterate keys() FIRST to discover
        argument names, then look up each via ``__getitem__``. With
        keys() empty, __getitem__ is never invoked — so its raising
        KeyError is intentionally unreachable in the test path. The
        real BatchEncoding returns real keys (pixel_values, etc.)
        and the unpacking pulls real tensors out; that production
        path is exercised when the adapter is deployed against the
        actual transformers wheel.
        """

        def to(self, device):
            return self

        def keys(self):
            return []

        def __iter__(self):
            return iter([])

        def __getitem__(self, key):
            raise KeyError(key)

    class _FakeProcessor:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def __call__(self, images=None, return_tensors=None, **kwargs):
            return _FakeInputs()

        def batch_decode(self, ids, skip_special_tokens=True):
            return [caption]

        @classmethod
        def from_pretrained(cls, model_id: str):
            return cls(model_id)

    class _FakeModel:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            # Return a sentinel — batch_decode ignores it and returns
            # the canned caption.
            return [[0]]

        @classmethod
        def from_pretrained(cls, model_id: str):
            return cls(model_id)

    transformers = types.ModuleType("transformers")
    transformers.BlipProcessor = _FakeProcessor  # type: ignore[attr-defined]
    transformers.BlipForConditionalGeneration = _FakeModel  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers

    return {"processor_cls": _FakeProcessor, "model_cls": _FakeModel}


def _solid_jpeg(width: int = 320, height: int = 240) -> bytes:
    """Generate a real decodable JPEG via Pillow. BLIP's image path
    uses Pillow under the hood (via ``PIL.Image.open`` in
    ``_decode_image``), so we need a valid image — a marker-prefix
    stub would fail the decode step."""
    from PIL import Image

    img = Image.new("RGB", (width, height), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def blip_environment(monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: stubbed transformers + torch + a canned JPEG."""
    fakes = install_fake_transformers()
    return {
        "fake_processor_cls": fakes["processor_cls"],
        "fake_model_cls": fakes["model_cls"],
        "sample_jpeg": _solid_jpeg(),
    }


def _boot_app(env, monkeypatch: pytest.MonkeyPatch):
    """Reload service + main modules so the stubbed imports take
    effect, then return a TestClient against the FastAPI app."""
    for mod_name in (
        "adapters.blip.service",
        "adapters.blip.main",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.blip.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def blip_app(blip_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(blip_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def blip_app_with_auth(blip_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(blip_environment, monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def sample_jpeg(blip_environment) -> bytes:
    return blip_environment["sample_jpeg"]
