# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""The cookbook is executable documentation.

Every file in `cookbook/` is imported here, and the runnable ones are
run. An example that names something the SDK no longer exports, or calls
a method with the wrong signature, fails in CI rather than misleading a
model developer months later.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

COOKBOOK = Path(__file__).resolve().parent.parent / "cookbook"
FILES = sorted(COOKBOOK.glob("*.py"))


def load(path: Path):
    name = f"cookbook_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module", params=FILES, ids=lambda p: p.stem)
def example(request):
    return load(request.param)


def test_the_cookbook_is_not_empty():
    assert len(FILES) >= 9, "cookbook files went missing"


def test_every_example_imports(example):
    """Importing is the assertion: every name it references exists, and
    nothing loads a model or reaches the network at import time."""
    assert example.__doc__


def test_every_example_names_what_it_demonstrates(example):
    assert "Demonstrates:" in (example.__doc__ or "")


# ── The ones that run ───────────────────────────────────────────────


def test_the_facade_example_builds_a_real_adapter():
    module = load(COOKBOOK / "01_adapter_facade.py")
    adapter = module.adapter
    assert adapter.id == "fall-detection"
    assert adapter.tasks == ("object_detection",)
    # The weights file is absent here, so the fingerprint falls back to
    # the deterministic identity hash — and is still never null.
    assert adapter.fingerprint.startswith("sha256:")
    info = adapter.model_info()
    assert info.modalities_in == ["image"] and info.framework == "onnxruntime"


def test_the_service_example_is_a_working_adapter():
    module = load(COOKBOOK / "02_adapter_service.py")
    with TestClient(module.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        caps = client.get("/capabilities").json()
    assert caps["tasks_advertised"] == ["license_plate_recognition"]
    assert caps["model"]["fingerprint"]
    assert caps["scheduling"]["fair_queuing"] == "per_camera"


def test_the_result_conventions_validate():
    module = load(COOKBOOK / "03_result_conventions.py")
    detection = module.typed_detection()
    assert detection["detections"][0]["bbox"]["x"] == 0.11
    assert detection["frame_dimensions"] == {"w": 1920, "h": 1080}
    assert module.classify()["predictions"][0]["label"] == "daylight"

    import base64

    with TestClient(module.transcriber.app) as client:
        body = client.post("/infer", json={
            "audio_b64": base64.b64encode(b"pcm").decode()}).json()
    assert body["result"]["transcript"] == "the gate is open"
    assert body["result"]["segments"][0]["end_ms"] == 1400


def test_the_error_example_maps_every_category():
    module = load(COOKBOOK / "04_errors_and_backpressure.py")
    from opennvr_adapter_sdk.contract import ErrorCategory

    assert set(module.CATEGORIES) == set(ErrorCategory)

    import base64

    with TestClient(module.adapter.app) as client:
        empty = client.post("/infer", json={"frame_b64": ""})
        ok = client.post("/infer", json={
            "frame_b64": base64.b64encode(b"jpeg").decode()})
    assert empty.status_code == 400
    assert empty.json()["error"]["transient"] is False
    assert ok.status_code == 200


def test_the_streaming_example_advertises_and_documents_streaming():
    module = load(COOKBOOK / "05_streaming.py")
    with TestClient(module.app) as client:
        caps = client.get("/capabilities").json()
        asyncapi = client.get("/asyncapi.json").json()
    assert caps["endpoints"]["infer_stream"]["supported"] is True
    assert "handshake" in asyncapi["components"]["messages"]


def test_the_metrics_example_registers_its_own_series():
    module = load(COOKBOOK / "06_capabilities_and_metrics.py")
    import base64

    with TestClient(module.adapter.app) as client:
        client.post("/infer", json={
            "frame_b64": base64.b64encode(b"jpeg").decode()})
        exposition = client.get("/metrics").text
        caps = client.get("/capabilities").json()
    assert "adapter_preprocess_seconds" in exposition
    assert "adapter_frames_skipped_total" in exposition
    # The §3.4 baseline the SDK provides, still there.
    assert "adapter_model_info" in exposition
    assert caps["permissions"]["gpu"] is True
    assert caps["permissions"]["network_egress"] == ["huggingface.co"]


def test_the_conformance_example_passes_its_own_tests():
    """07 is a test file about testing an adapter; run it.

    Its conformance check posts multipart, because that is what KAI-C
    sends — so it needs FastAPI's form extra, which the SDK itself does
    not depend on."""
    module = load(COOKBOOK / "07_testing_and_conformance.py")
    has_multipart = importlib.util.find_spec("multipart") is not None
    ran = 0
    for name in dir(module):
        if not name.startswith("test_"):
            continue
        if name == "test_the_adapter_conforms_to_the_contract" and not has_multipart:
            continue
        getattr(module, name)()
        ran += 1
    assert ran >= 3


def test_the_weights_example_points_the_fingerprint_at_the_final_path():
    module = load(COOKBOOK / "08_weights_and_packaging.py")
    assert module.adapter.weights == module.WEIGHTS_PATH
    assert module.adapter._network_egress == ("huggingface.co",)


def test_the_spec_example_publishes_both_documents():
    module = load(COOKBOOK / "09_specs_and_publishing.py")
    openapi = module.the_http_surface()
    assert openapi["openapi"] == "3.1.0"
    assert "InferResponse" in openapi["components"]["schemas"]
    asyncapi = module.the_streaming_surface()
    assert asyncapi["asyncapi"] == "3.0.0"
    assert asyncapi["channels"] == {}      # this one does not stream
