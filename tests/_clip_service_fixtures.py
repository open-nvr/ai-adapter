# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the CLIP embedding adapter's contract tests.

Stubs ``open_clip`` and ``torch`` so the tests run without a 600 MB
weights download and without a network. The fake encoders return
deterministic vectors that are NOT unit length, which is the point —
normalisation is this adapter's promise to its consumer, and a fixture
that handed back unit vectors would let a broken ``_unit`` pass.

Two things the fake deliberately does NOT do:

* It does not return the same vector for text and image. Half of this
  adapter's job is that the two modalities land in one comparable
  space, and a fixture where they are identical could not tell a
  working cross-modal path from one that ignored its input.
* It does not pretend to be semantic. Whether CLIP actually puts "a red
  truck" near a picture of one is a property of the WEIGHTS, not of
  this code, and asserting it against a stub would be theatre. That
  claim is checked against the real model in
  ``test_clip_real_model.py``, which is skipped unless the weights are
  present.
"""
from __future__ import annotations

import importlib
import io
import sys
import types

import pytest

#: Width of the fake embedding. Not 512 on purpose — the service must
#: read the dimension from the model rather than assuming ViT-B-32, and
#: a fixture at 512 would hide a hard-coded constant.
FAKE_DIM = 8


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A torch stub with just enough tensor behaviour for the service.

    ``_unit`` needs ``.norm()``, ``/``, ``.tolist()`` and indexing;
    ``_weights_fingerprint`` needs ``state_dict()`` and numpy bytes.
    """
    import numpy as np

    class FakeTensor:
        def __init__(self, data):
            self._a = np.asarray(data, dtype=np.float32)

        def __getitem__(self, idx):
            return FakeTensor(self._a[idx])

        def __truediv__(self, scalar):
            return FakeTensor(self._a / scalar)

        def norm(self):
            return FakeScalar(float(np.linalg.norm(self._a)))

        def tolist(self):
            return self._a.tolist()

        def unsqueeze(self, _dim):
            return FakeTensor(self._a[None, ...])

        def to(self, _device):
            return self

        def detach(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self._a

    class FakeScalar:
        def __init__(self, v):
            self._v = v

        def item(self):
            return self._v

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = _NoGrad
    torch.set_num_threads = lambda _n: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.Tensor = FakeTensor
    torch._fake_tensor = FakeTensor
    monkeypatch.setitem(sys.modules, "torch", torch)


def _install_fake_open_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    FakeTensor = sys.modules["torch"]._fake_tensor

    class FakeModel:
        def __init__(self):
            self.text_projection = np.zeros((4, FAKE_DIM), dtype=np.float32)
            self.seen_text: list[list[str]] = []
            self.seen_images = 0

        def eval(self):
            return self

        def to(self, _device):
            return self

        def state_dict(self):
            return {"w": FakeTensor(np.arange(4, dtype=np.float32))}

        def encode_text(self, tokens):
            self.seen_text.append(tokens)
            # Deliberately not unit length, and distinct from the image
            # vector below.
            v = np.zeros((1, FAKE_DIM), dtype=np.float32)
            v[0, 0] = 3.0
            v[0, 1] = 4.0
            return FakeTensor(v)

        def encode_image(self, batch):
            self.seen_images += 1
            v = np.zeros((1, FAKE_DIM), dtype=np.float32)
            v[0, 2] = 6.0
            v[0, 3] = 8.0
            return FakeTensor(v)

    def _preprocess(image):
        # Asserts the service handed over a real PIL image in RGB.
        assert image.mode == "RGB", f"preprocess got mode {image.mode}"
        return FakeTensor(np.zeros((3, 4, 4), dtype=np.float32))

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = (
        lambda arch, pretrained=None, cache_dir=None: (
            FakeModel(), None, _preprocess))
    open_clip.get_tokenizer = lambda arch: (lambda texts: list(texts))
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)


@pytest.fixture
def clip_environment(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_open_clip(monkeypatch)
    monkeypatch.setenv("CLIP_CACHE_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("CLIP_OFFLINE", "1")
    yield


def _boot_app(monkeypatch: pytest.MonkeyPatch):
    for mod_name in ("adapters.clip.service", "adapters.clip.main"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.clip.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def clip_app(clip_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client


@pytest.fixture
def clip_app_with_auth(clip_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def service(clip_environment):
    """A freshly loaded ``ClipEmbeddingService`` for tests that drive it
    directly rather than over HTTP."""
    import adapters.clip.service as service_module
    importlib.reload(service_module)

    svc = service_module.ClipEmbeddingService()
    svc.load()
    assert svc.is_ready(), f"ClipEmbeddingService failed to load: {svc._load_error}"
    return svc


@pytest.fixture
def small_jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()
