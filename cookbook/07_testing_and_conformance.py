# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Proving it works — `ConformanceRunner` and the tests worth writing.

Demonstrates: `ConformanceRunner`, `ConformanceReport`, `CheckOutcome`,
`CheckResult`, FastAPI's `TestClient` against an adapter, and the
`opennvr-adapter validate` / `conform` commands.

Conformance is the contract, executable: it probes every mandatory
endpoint, validates the wire shapes against the SDK's own Pydantic
models, and reports PASS / WARN / FAIL / SKIP. A green run means KAI-C
will accept the adapter — which is the only assurance a model developer
can get without a deployment to try it in.
"""
import base64

from fastapi.testclient import TestClient

from opennvr_adapter_sdk import Adapter
from opennvr_adapter_sdk.conformance import CheckOutcome, ConformanceRunner

adapter = Adapter("demo-tested", tasks=["object_detection"],
                  framework="onnxruntime")


@adapter.load()
def load():
    return object()


@adapter.on_image()
def detect(call):
    if not call.image:
        raise ValueError("a frame is required")
    return [call.detection("person", 0.9, 0.1, 0.1, 0.2, 0.3)]


FRAME = b"\xff\xd8\xff\xe0 pretend jpeg"


# ── The conformance run, in-process ────────────────────────────────


def test_the_adapter_conforms_to_the_contract():
    """What `opennvr-adapter validate .` runs, as a test — so a change
    that breaks the contract fails in CI rather than at install time.

    The runner's client is duck-typed, so TestClient drives the real
    ASGI app with no port bound and no network."""
    base_url = "http://adapter.test"
    with TestClient(adapter.app, base_url=base_url) as client:
        report = ConformanceRunner(base_url, client=client).run_all()

    assert report.is_green, [
        (r.name, r.detail) for r in report.results
        if r.outcome == CheckOutcome.FAIL
    ]


# ── The tests worth writing about YOUR model ───────────────────────


def test_it_answers_in_the_contract_shape():
    with TestClient(adapter.app) as client:
        response = client.post("/infer", json={
            "frame_b64": base64.b64encode(FRAME).decode(),
            "task": "object_detection",
        })
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    detection = body["result"]["detections"][0]
    # Normalized coordinates. The single most common thing to get
    # wrong, and it is invisible until an operator sees a box in the
    # wrong place.
    assert 0.0 <= detection["bbox"]["x"] <= 1.0


def test_a_body_the_model_cannot_use_is_a_400_not_a_500():
    """The distinction decides whether KAI-C retries — so a bad frame
    classified as a 500 becomes a retry storm."""
    with TestClient(adapter.app) as client:
        response = client.post("/infer", json={"frame_b64": ""})
    assert response.status_code == 400
    assert response.json()["error"]["transient"] is False


def test_capabilities_advertise_a_task_and_a_fingerprint():
    """An adapter with no advertised task gets no work; one with a null
    fingerprint is silently exempt from drift detection."""
    with TestClient(adapter.app) as client:
        caps = client.get("/capabilities").json()
    assert caps["tasks_advertised"] == ["object_detection"]
    assert caps["model"]["fingerprint"]


def test_health_is_honest_about_a_failed_load():
    """A green dot on a dead adapter routes real work into a hole."""
    broken = Adapter("demo-broken", tasks=["object_detection"])
    broken.on_image()(lambda call: [])
    broken.load()(lambda: (_ for _ in ()).throw(RuntimeError("no weights")))

    service = broken.service
    service.load()
    assert service.is_ready() is False
    assert "no weights" in service.hardware_evaluation().reasoning


# Against a RUNNING adapter instead:
#
#     opennvr-adapter conform http://localhost:9001 --token $TOKEN
#     opennvr-adapter conform http://localhost:9001 --json > report.json
