# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""The specs an adapter publishes about itself.

`AdapterApp` has always been FastAPI, so `/openapi.json` always existed
— and documented nothing: every route returned a bare `JSONResponse`, so
the document had six paths, zero schemas and zero components. A client
generator pointed at it produced `Any` everywhere, which is worse than
no document at all because it looks complete. These tests pin the
document to the contract types the adapter actually returns.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opennvr_adapter_sdk import (
    AdapterApp, AdapterService, HardwareEvaluationResponse, InferResponse, ModelInfo,
)
from opennvr_adapter_sdk.adapter_app import BodyShape
from opennvr_adapter_sdk.openapi import (
    CONTRACT_VERSION, adapter_asyncapi, error_responses, infer_request_body,
)

try:
    from openapi_spec_validator import validate as validate_openapi
except ImportError:  # pragma: no cover — optional dev dependency
    validate_openapi = None


class _Service(AdapterService):
    def load(self): pass

    def is_ready(self): return True

    def fingerprint(self): return "sha256:abc"

    def model_info(self):
        return ModelInfo(name="demo", version="1", framework="custom",
                         fingerprint="sha256:abc")

    def hardware_evaluation(self):
        return HardwareEvaluationResponse(verdict="ok")

    def infer(self, payload):
        return InferResponse(model_name="demo", model_version="1",
                             inference_ms=1, result={})

    async def handle_stream(self, websocket): pass


def build(**overrides) -> AdapterApp:
    kwargs = dict(service=_Service(), name="demo", version="1.0.0", vendor="ACME",
                  license="MIT", tasks_advertised=["object_detection"],
                  body_shape=BodyShape.IMAGE)
    kwargs.update(overrides)
    return AdapterApp(**kwargs)


def spec(**overrides) -> dict:
    return build(**overrides).fastapi_app.openapi()


# ── The document is no longer hollow ────────────────────────────────


def test_every_response_is_typed():
    document = spec()
    schemas = document["components"]["schemas"]
    # Before: zero. The contract types were Pydantic all along; they
    # were simply never wired to the routes.
    assert {"HealthResponse", "CapabilitiesResponse", "HardwareEvaluationResponse",
            "InferResponse", "FailureEnvelope", "ModelInfo"} <= set(schemas)

    def ref(path, verb="get", code="200"):
        content = document["paths"][path][verb]["responses"][code]["content"]
        return content["application/json"]["schema"]["$ref"]

    assert ref("/health").endswith("/HealthResponse")
    assert ref("/capabilities").endswith("/CapabilitiesResponse")
    assert ref("/hardware/evaluation").endswith("/HardwareEvaluationResponse")
    assert ref("/infer", "post").endswith("/InferResponse")


def test_errors_carry_the_failure_envelope():
    document = spec()
    responses = document["paths"]["/infer"]["post"]["responses"]
    assert {"400", "401", "403", "413", "415", "500", "503"} <= set(responses)
    for code in ("400", "500", "503"):
        schema = responses[code]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/FailureEnvelope")
        assert responses[code]["description"]


def test_metrics_is_declared_as_prometheus_text_not_json():
    """It returns an exposition format; saying `application/json` sends
    a generated client looking for a JSON body that never arrives."""
    responses = spec()["paths"]["/metrics"]["get"]["responses"]
    assert set(responses["200"]["content"]) == {"text/plain"}


@pytest.mark.parametrize("shape,expected", [
    (BodyShape.IMAGE, ("frame", "frame_b64")),
    (BodyShape.AUDIO, ("audio", "audio_b64")),
    (BodyShape.GENERIC, ("data", "data_b64")),
])
def test_the_infer_body_is_described_for_this_adapters_shape(shape, expected):
    body = spec(body_shape=shape)["paths"]["/infer"]["post"]["requestBody"]
    multipart, json_body = expected
    assert set(body["content"]) == {"multipart/form-data", "application/json"}
    assert multipart in body["content"]["multipart/form-data"]["schema"]["properties"]
    assert json_body in body["content"]["application/json"]["schema"]["properties"]
    assert body["content"]["multipart/form-data"]["schema"]["required"] == [multipart]


def test_a_text_adapter_documents_no_upload():
    body = spec(body_shape=BodyShape.TEXT)["paths"]["/infer"]["post"]["requestBody"]
    assert set(body["content"]) == {"application/json"}
    assert "takes no binary upload" in body["description"]


def test_the_body_size_limit_is_in_the_document():
    body = spec(max_body_bytes=1234)["paths"]["/infer"]["post"]["requestBody"]
    assert "1234 bytes" in body["description"]


# ── Auth, identity, navigation ──────────────────────────────────────


def test_bearer_auth_is_declared_where_it_is_enforced():
    document = spec()
    assert "bearerAuth" in document["components"]["securitySchemes"]
    protected = {"/capabilities", "/hardware/evaluation", "/infer"}
    for path, operations in document["paths"].items():
        for operation in operations.values():
            if path in protected:
                assert operation["security"] == [{"bearerAuth": []}], path
    # …and not where an operator must be able to scrape a failing adapter.
    for path in ("/health", "/metrics"):
        assert "security" not in document["paths"][path]["get"]


def test_the_info_block_identifies_the_adapter():
    info = spec()["info"]
    assert info["title"] == "demo adapter"
    assert info["x-opennvr-contract-version"] == CONTRACT_VERSION
    assert info["x-opennvr-tasks"] == ["object_detection"]
    assert info["license"] == {"name": "MIT"}
    assert info["contact"]["name"] == "ACME"
    assert "object_detection" in info["description"]


def test_routes_are_grouped_for_a_reader():
    document = spec()
    assert [tag["name"] for tag in document["tags"]] == [
        "contract", "inference", "observability"]
    assert document["paths"]["/infer"]["post"]["tags"] == ["inference"]


@pytest.mark.skipif(validate_openapi is None,
                    reason="openapi-spec-validator not installed")
@pytest.mark.parametrize("shape", list(BodyShape))
@pytest.mark.parametrize("stream", [True, False])
def test_the_document_is_valid_openapi_31(shape, stream):
    validate_openapi(spec(body_shape=shape, supports_stream=stream))


def test_error_responses_helper_narrows_to_what_a_route_can_do():
    assert set(error_responses(400, 503)) == {400, 503}
    assert set(error_responses()) >= {400, 401, 500, 503}
    assert error_responses(999) == {}


def test_infer_request_body_rejects_nothing_it_cannot_describe():
    for shape in BodyShape:
        body = infer_request_body(shape, max_bytes=10)["requestBody"]
        assert body["required"] is True and body["content"]


# ── AsyncAPI: the surface OpenAPI cannot reach ──────────────────────


def test_the_streaming_protocol_is_published_as_asyncapi():
    document = adapter_asyncapi("demo", "1.0.0", tasks=("object_detection",))
    assert document["asyncapi"] == "3.0.0"
    assert document["servers"]["adapter"]["protocol"] == "ws"
    assert document["servers"]["adapter"]["pathname"] == "/infer/stream"
    assert set(document["operations"]) == {"send", "receive"}

    messages = document["components"]["messages"]
    # Every message the §6 protocol defines, both directions.
    assert set(messages) == {
        "handshake", "frame", "frameRef", "resultAck", "close",
        "handshakeAck", "result", "pause", "resume", "stats"}
    # Each one resolves to a schema generated from the contract type.
    for message in messages.values():
        name = message["payload"]["$ref"].rsplit("/", 1)[-1]
        assert name in document["components"]["schemas"]
        assert document["components"]["schemas"][name]["type"] == "object"


def test_the_directions_are_right():
    document = adapter_asyncapi("demo", "1.0.0")
    sent = {m["$ref"].rsplit("/", 1)[-1] for m in document["operations"]["send"]["messages"]}
    received = {m["$ref"].rsplit("/", 1)[-1]
                for m in document["operations"]["receive"]["messages"]}
    assert "result" in sent and "pause" in sent
    assert "frame" in received and "handshake" in received
    assert not sent & received


def test_a_non_streaming_adapter_says_so_instead_of_lying():
    document = adapter_asyncapi("demo", "1.0.0", supports_stream=False)
    assert document["channels"] == {} and document["operations"] == {}
    assert "does not support streaming" in document["info"]["description"]


def test_the_adapter_serves_both_documents():
    client = TestClient(build(supports_stream=True).fastapi_app)
    with client:
        assert client.get("/openapi.json").json()["openapi"].startswith("3.")
        asyncapi = client.get("/asyncapi.json").json()
        assert asyncapi["asyncapi"] == "3.0.0"
        assert asyncapi["info"]["x-opennvr-tasks"] == ["object_detection"]


def test_a_non_streaming_adapter_documents_its_501():
    document = spec(supports_stream=False)
    responses = document["paths"]["/infer/stream"]["get"]["responses"]
    assert responses["501"]["content"]["application/json"]["schema"]["$ref"] \
        .endswith("/FailureEnvelope")
