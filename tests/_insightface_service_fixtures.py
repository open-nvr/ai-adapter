# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared fixtures for the SDK-based InsightFace adapter tests.

Stubs the ``insightface`` package so the tests don't pull in
onnxruntime or download model weights. The fake ``FaceAnalysis``
returns deterministic face objects so the recognition + embedding
paths have concrete inputs to validate.

Same flat-module pattern as the other adapter fixtures in this
directory to sidestep the ``tests/adapters/`` namespace collision.
"""
from __future__ import annotations

import importlib
import io
import sys
import types
from pathlib import Path

import pytest


# ── Deterministic fake faces ────────────────────────────────────────


# Two distinct 512-d embeddings — one for "Alice", one for "Bob".
# Constructed so the cosine similarity between them is below 0.5
# (so they're unambiguously different people in tests) and the
# similarity of each with itself is exactly 1.0.
_FAKE_EMBEDDING_ALICE: list[float] = [0.1 if i < 256 else 0.0 for i in range(512)]
_FAKE_EMBEDDING_BOB:   list[float] = [0.0 if i < 256 else 0.1 for i in range(512)]


class _FakeFace:
    """Stand-in for the ``insightface.app.common.Face`` object."""

    def __init__(
        self,
        *,
        bbox: tuple[int, int, int, int] = (100, 120, 200, 240),
        det_score: float = 0.95,
        embedding: list[float] | None = None,
        age: int = 30,
        gender: int = 1,
    ) -> None:
        # Mimic the numpy.ndarray .tolist() API by wrapping in a
        # tiny shim — the service code calls ``.tolist()`` on bbox /
        # landmarks / embedding.
        self.bbox = _ToList(list(bbox))
        self.det_score = det_score
        self.kps = _ToList([[110, 130], [140, 130], [125, 160], [115, 180], [135, 180]])
        self.age = age
        self.gender = gender
        emb = embedding if embedding is not None else _FAKE_EMBEDDING_ALICE
        self.normed_embedding = _ToList(list(emb))


class _ToList:
    """Cheap stand-in for numpy arrays — only provides .tolist()."""

    def __init__(self, value: list) -> None:
        self._value = value

    def tolist(self) -> list:
        return list(self._value)


def install_fake_insightface(faces: list[_FakeFace] | None = None):
    """Inject a stub ``insightface.app`` providing ``FaceAnalysis``.

    The fake's ``get(image)`` returns whatever ``faces`` was set to at
    install time. Tests that need a specific scenario (no face, two
    faces, low-confidence face) override by re-installing.
    """

    class _FakeFaceAnalysis:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs
            self.models = {
                "recognition_w600k_r50": _FakeModel(),
                "detection_retina": _FakeModel(),
            }

        def prepare(self, *args, **kwargs):  # noqa: D401
            self.prepared_with = kwargs
            return None

        def get(self, image):  # noqa: D401
            # Default: return the canned face list. Tests that need
            # to vary by call can monkey-patch this method.
            return list(faces) if faces is not None else [
                _FakeFace(embedding=_FAKE_EMBEDDING_ALICE)
            ]

    pkg = types.ModuleType("insightface")
    app_mod = types.ModuleType("insightface.app")
    app_mod.FaceAnalysis = _FakeFaceAnalysis
    pkg.app = app_mod
    sys.modules["insightface"] = pkg
    sys.modules["insightface.app"] = app_mod
    return _FakeFaceAnalysis


class _FakeModel:
    """Stand-in for an InsightFace internal model object — exposes
    ``model_file`` so the fingerprint helper has a path to hash."""

    def __init__(self):
        self.model_file = None  # tests that need a real path override


def _solid_jpeg(width: int = 320, height: int = 240) -> bytes:
    """Generate a real decodable JPEG via Pillow. The fake
    FaceAnalysis ignores the pixel content but cv2.imdecode in the
    service path needs a valid image — using a synthesised JPEG
    avoids stubbing cv2 (which the YOLOv8 tests use for real)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def insightface_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sandboxed env: tmp face DB path, fake insightface package, a
    canned JPEG byte string."""
    db_path = tmp_path / "faces.json"
    monkeypatch.setenv("OPENNVR_INSIGHTFACE_FACE_DB", str(db_path))

    fake_cls = install_fake_insightface(
        [_FakeFace(embedding=_FAKE_EMBEDDING_ALICE)]
    )

    # numpy + cv2 + PIL are core project deps (used by YOLOv8 +
    # license-plate tests too), so we use the real packages here.
    # The fake FaceAnalysis ignores the decoded image, but cv2 must
    # be able to imdecode a real JPEG — hence ``_solid_jpeg()``
    # produces a valid Pillow-generated image, not a garbage byte
    # prefix.
    return {
        "db_path": db_path,
        "fake_face_analysis_cls": fake_cls,
        "sample_jpeg": _solid_jpeg(),
        "embedding_alice": list(_FAKE_EMBEDDING_ALICE),
        "embedding_bob": list(_FAKE_EMBEDDING_BOB),
        "FakeFace": _FakeFace,
    }


def _boot_app(env, monkeypatch: pytest.MonkeyPatch):
    for mod_name in (
        "adapters.insightface.service",
        "adapters.insightface.main",
    ):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

    from fastapi.testclient import TestClient
    import adapters.insightface.main as main_module
    return TestClient(main_module.app), main_module


@pytest.fixture
def insightface_app(insightface_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENNVR_ADAPTER_TOKEN", raising=False)
    client, _ = _boot_app(insightface_environment, monkeypatch)
    with client:
        yield client


@pytest.fixture
def insightface_app_with_auth(insightface_environment, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", "test-token")
    client, _ = _boot_app(insightface_environment, monkeypatch)
    with client:
        yield client, "test-token"


@pytest.fixture
def sample_jpeg(insightface_environment) -> bytes:
    return insightface_environment["sample_jpeg"]
