# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Inference must not hold the event loop.

``_handle_infer`` is a coroutine and used to call the model inline. A
model call is seconds of synchronous work, so for its whole duration the
loop could answer nothing — not /health, not /capabilities. On a busy
site that read as an adapter that was DOWN: KAI-C's polls timed out, it
marked the adapter unavailable, and registrations queued behind it stalled,
while the model was busy and perfectly healthy.

These pin the two halves of the fix: the loop stays free during a slow
inference, and the declared max_inflight still bounds concurrency now that
the loop no longer serialises calls by accident.
"""
from __future__ import annotations

import base64
import threading
import time

from fastapi.testclient import TestClient

from opennvr_adapter_sdk.facade import Adapter

FRAME = b"\xff\xd8\xff\xe0 not really a jpeg"
SLOW_S = 1.5


def slow_adapter(**kwargs) -> Adapter:
    adapter = Adapter("slow-model", version="1.0.0", vendor="ACME",
                      license="Apache-2.0", tasks=["object_detection"],
                      framework="onnxruntime", **kwargs)

    @adapter.load()
    def load():
        return object()

    @adapter.on_image()
    def detect(call):
        time.sleep(SLOW_S)          # a real model: synchronous, seconds
        return [call.detection("thing", 0.9, 0.1, 0.1, 0.2, 0.2)]

    return adapter


def _post_infer(client: TestClient, out: dict) -> None:
    started = time.monotonic()
    r = client.post("/infer", json={"frame_b64": base64.b64encode(FRAME).decode()})
    out["status"] = r.status_code
    out["elapsed"] = time.monotonic() - started


def test_health_answers_while_a_slow_inference_runs():
    # The context manager runs the lifespan, which is what loads the model;
    # a bare TestClient answers 503 not-ready to every inference.
    with TestClient(slow_adapter().app) as client:
        assert client.get("/health").status_code == 200

        result: dict = {}
        worker = threading.Thread(target=_post_infer, args=(client, result))
        worker.start()
        time.sleep(0.2)             # the inference is now inside the model

        started = time.monotonic()
        r = client.get("/health")
        health_latency = time.monotonic() - started
        worker.join()

    assert r.status_code == 200
    assert result["status"] == 200
    assert result["elapsed"] >= SLOW_S, "the inference really was slow"
    # Before the fix this was ~SLOW_S: /health waited for the model.
    assert health_latency < 0.5, f"/health blocked for {health_latency:.2f}s behind inference"


def test_declared_max_inflight_still_bounds_concurrency():
    """The loop no longer serialises calls by accident, so the contract's
    max_inflight has to — two calls into a max_inflight=1 model take two
    inference-times, not one."""
    with TestClient(slow_adapter(max_inflight=1).app) as client:
        results = [{}, {}]
        threads = [threading.Thread(target=_post_infer, args=(client, results[i])) for i in range(2)]
        started = time.monotonic()
        for t in threads: t.start()
        for t in threads: t.join()
        wall = time.monotonic() - started

    assert all(r["status"] == 200 for r in results)
    assert wall >= 2 * SLOW_S - 0.2, f"two calls overlapped ({wall:.2f}s) despite max_inflight=1"
