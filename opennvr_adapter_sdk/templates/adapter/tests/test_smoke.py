# Copyright (c) 2026 __ADAPTER_NAME__ authors
# SPDX-License-Identifier: __LICENSE__

"""
Smoke tests for __ADAPTER_NAME__ — the parity bar for an adapter.

These drive the real ASGI app: the same routes, body parsing and
failure envelope KAI-C will meet. Keep them green as you replace the
starter handler with your model.
"""
from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from __ADAPTER_MODULE__ import app

# A valid 1x1 black JPEG — enough to get past a decoder.
FRAME = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBg"
    "YFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoK"
    "CgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAA"
    "EDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIE"
    "AwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJi"
    "coKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWW"
    "l5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09f"
    "b3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQA"
    "AQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKj"
    "U2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJma"
    "oqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9"
    "oADAMBAAIRAxEAPwD+f+iiigD/2Q=="
)


def test_health_goes_green_once_the_model_loads():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["adapter_name"] == "__ADAPTER_ID__"


def test_capabilities_advertise_the_task_and_a_fingerprint():
    """An adapter with no advertised task gets no work, and one with a
    null fingerprint is silently exempt from KAI-C's drift detection."""
    with TestClient(app) as client:
        caps = client.get("/capabilities").json()
    assert "__TASK__" in caps["tasks_advertised"]
    assert caps["model"]["fingerprint"]


def test_the_host_is_evaluated():
    with TestClient(app) as client:
        verdict = client.get("/hardware/evaluation").json()
    assert verdict["verdict"] in ("ok", "warn", "blocked")
    assert verdict["reasoning"]


def test_inference_answers_in_the_contract_shape():
    with TestClient(app) as client:
        response = client.post(
            "/infer",
            files={"frame": ("frame.jpg", FRAME, "image/jpeg")},
            data={"task": "__TASK__"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "__ADAPTER_ID__"
    # TODO: assert on YOUR model's answer once the handler is real.
    assert isinstance(body["result"], dict)


def test_a_body_this_model_cannot_use_is_a_400_not_a_500():
    """The distinction matters: KAI-C retries a 500 and does not retry a
    400, so misclassifying a bad frame produces a retry storm."""
    with TestClient(app) as client:
        response = client.post("/infer", json={"not_a_frame": True})
    assert response.status_code == 400
    assert response.json()["error"]["transient"] is False


def test_the_published_spec_documents_the_contract():
    with TestClient(app) as client:
        document = client.get("/openapi.json").json()
    assert document["info"]["x-opennvr-tasks"] == ["__TASK__"]
    assert "InferResponse" in document["components"]["schemas"]
