# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wiring for adapter domain metrics.

The per-adapter instrumentation registers its series inside ``load()``
and increments inside ``infer()``. That only reaches ``/metrics`` if
``AdapterApp`` attaches itself to the service BEFORE calling ``load()``
— otherwise registration lands on the service's private fallback
registry, ``inc_counter`` later raises against the app registry, the
adapter's own try/except swallows it, and every domain series silently
disappears from the scrape. This test locks that ordering in, plus the
model-identity metric the same lifespan populates.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from opennvr_adapter_sdk import (
    AdapterApp,
    AdapterService,
    BodyShape,
    HardwareEvaluationResponse,
    HardwareVerdict,
    InferResponse,
    ModelInfo,
)


class _DomainService(AdapterService):
    """Mirrors the real adapters: register in load(), increment in infer()."""

    def __init__(self) -> None:
        self._ready = False

    def load(self) -> None:
        self.metrics.register_counter(
            "adapter_detections_total", "Objects by class.",
            label_key="label", allowed_values=("person", "car"))
        self.metrics.register_histogram(
            "adapter_realtime_factor", "RTF.", buckets=(0.5, 1.0, 2.0))
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def fingerprint(self) -> str:
        return "sha256:deadbeef"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="demo-model", version="1.0", framework="numpy",
            modalities_in=["text"], modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        return HardwareEvaluationResponse(
            verdict=HardwareVerdict.OK, reasoning="ready",
            checked_at=datetime.now(timezone.utc), details={},
        )

    def infer(self, payload: dict) -> InferResponse:
        self.metrics.inc_counter("adapter_detections_total", 2, label_value="person")
        self.metrics.observe("adapter_realtime_factor", 0.4)
        return InferResponse(
            model_name="demo-model", model_version="1.0",
            inference_ms=1, result={"ok": True},
        )


def _client() -> TestClient:
    app = AdapterApp(
        service=_DomainService(),
        name="demo", version="9.9.9", vendor="opennvr", license="Apache-2.0",
        tasks_advertised=["echo"], body_shape=BodyShape.TEXT,
    ).fastapi_app
    return TestClient(app)


def test_domain_series_registered_in_load_reach_the_scrape():
    with _client() as c:
        assert c.post("/infer", json={"task": "echo"}).status_code == 200
        body = c.get("/metrics").text
    # The load()-registered series exist AND carry the infer() increments —
    # proving registration and increment hit the SAME registry.
    assert 'adapter_detections_total{label="person"} 2' in body
    assert 'adapter_realtime_factor_bucket{le="0.5"} 1' in body
    assert "adapter_realtime_factor_count 1" in body


def test_model_identity_and_task_label_present_on_scrape():
    with _client() as c:
        c.post("/infer", json={"task": "echo"})
        body = c.get("/metrics").text
    assert 'model="demo-model"' in body and 'fingerprint="sha256:deadbeef"' in body
    assert 'adapter="demo"' in body and 'adapter_version="9.9.9"' in body
    # Advertised task keeps its own series; the generic counter is labelled.
    assert 'adapter_infer_total{outcome="ok",task="echo"} 1' in body
