# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
Machine-readable specs for an adapter's own surfaces.

``AdapterApp`` is FastAPI, so ``/openapi.json`` has always existed — but
every route returned a bare ``JSONResponse`` with no declared model, so
the document had six paths, zero schemas and zero components. A model
developer pointing a client generator at it got ``Any`` everywhere,
which is worse than no document at all: it looks complete.

The contract types in :mod:`~.contract` are already Pydantic. This
module wires them to the routes, so an adapter now publishes:

* **OpenAPI 3.1** at ``/openapi.json`` — every response typed, the
  ``/infer`` request body described for the adapter's own
  :class:`~.adapter_app.BodyShape`, the §7 failure envelope on every
  error status, bearer auth declared, and Swagger UI at ``/docs``.
* **AsyncAPI 3.0** at ``/asyncapi.json`` — the ``/infer/stream``
  WebSocket protocol, which OpenAPI cannot express at all. Every
  message in it (handshake, frame, result, pause, stats, close) is a
  contract type, so the document is generated, not written.

Both are emitted without running the adapter by ``opennvr-adapter
spec``, which is what makes them usable in CI and in a published API
reference.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opennvr_adapter_sdk.contract import (
    CloseMessage,
    FailureEnvelope,
    FrameMessage,
    FrameRefMessage,
    HandshakeAckMessage,
    HandshakeMessage,
    PauseMessage,
    ResultAckMessage,
    ResultMessage,
    ResumeMessage,
    StatsMessage,
)

if TYPE_CHECKING:  # pragma: no cover
    from opennvr_adapter_sdk.adapter_app import BodyShape

#: The AI Adapter Contract revision these documents describe.
CONTRACT_VERSION = "1"

_JSON = "application/json"

#: Every error status the SDK itself can return, and why. An adapter's
#: own ``ServiceError`` categories map onto these, so a generated client
#: has one error type to handle rather than an untyped blob.
ERROR_STATUSES: dict[int, str] = {
    400: "Malformed input — bad JSON, a missing required field, or "
         "undecodable base64 (§7 `transport_error`).",
    401: "Missing or invalid `Authorization: Bearer` token.",
    403: "Operator policy refused this call (§7 `permission_denied`).",
    413: "Body exceeds the adapter's configured size limit.",
    415: "Unsupported Content-Type for this adapter's body shape.",
    500: "The model failed on otherwise valid input (§7 `model_error`), "
         "or an upstream provider failed (`provider_error`).",
    503: "The model is still loading, or the adapter is shedding load "
         "(§7 `overloaded`) — see `retry_after_ms`.",
}


def error_responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """FastAPI ``responses=`` for the §7 failure envelope.

    With no arguments, every status the SDK can produce. Pass a subset
    for a route that cannot fail in some of those ways."""
    wanted = statuses or tuple(ERROR_STATUSES)
    return {
        code: {"model": FailureEnvelope, "description": ERROR_STATUSES[code]}
        for code in wanted if code in ERROR_STATUSES
    }


# ── The /infer request body ─────────────────────────────────────────


def infer_request_body(body_shape: "BodyShape", *, max_bytes: int) -> dict[str, Any]:
    """The ``openapi_extra`` describing ``POST /infer`` for one body shape.

    The handler parses the body itself (multipart or JSON, depending on
    what the adapter declared), so FastAPI cannot infer this from a
    signature — but the shape is fully determined by ``body_shape``, and
    a model developer needs it to know what to send."""
    from opennvr_adapter_sdk.adapter_app import (
        _BODY_SHAPE_B64_FIELD, _BODY_SHAPE_FILE_FIELD, BodyShape,
    )

    task = {
        "type": "string",
        "description": "Task name, one of the adapter's `tasks_advertised`. "
                       "Used for per-task metrics.",
    }
    camera = {
        "type": "string",
        "description": "Camera this frame came from, when the caller knows it.",
    }

    if body_shape == BodyShape.TEXT:
        return {
            "requestBody": {
                "required": True,
                "description": "This adapter takes no binary upload; the JSON "
                               "body is passed to `infer()` as-is.",
                "content": {_JSON: {"schema": {
                    "type": "object",
                    "properties": {"task": task, "camera_id": camera},
                    "additionalProperties": True,
                }}},
            }
        }

    file_field = _BODY_SHAPE_FILE_FIELD[body_shape]
    b64_field = _BODY_SHAPE_B64_FIELD[body_shape]
    noun = {"image": "frame", "audio": "audio clip"}.get(
        body_shape.value, "binary payload")

    return {
        "requestBody": {
            "required": True,
            "description": (
                f"The {noun} plus optional params, either way round:\n\n"
                f"* `multipart/form-data` with a binary `{file_field}` part "
                f"and an optional `params` part holding a JSON object — the "
                f"efficient route, and what KAI-C uses;\n"
                f"* `application/json` with `{b64_field}` holding the same "
                f"bytes base64-encoded — convenient from a shell.\n\n"
                f"Either way the limit is {max_bytes} bytes; over it is a 413."
            ),
            "content": {
                "multipart/form-data": {"schema": {
                    "type": "object",
                    "properties": {
                        file_field: {"type": "string", "format": "binary",
                                     "description": f"The {noun}."},
                        "params": {"type": "string",
                                   "description": "A JSON object of adapter "
                                                  "params, as a string."},
                        "task": task,
                    },
                    "required": [file_field],
                }},
                _JSON: {"schema": {
                    "type": "object",
                    "properties": {
                        b64_field: {"type": "string", "format": "byte",
                                    "description": f"The {noun}, base64-encoded."},
                        "task": task,
                        "camera_id": camera,
                    },
                    "required": [b64_field],
                    "additionalProperties": True,
                }},
            },
        }
    }


# ── Document-level polish ───────────────────────────────────────────


def describe(name: str, version: str, *, vendor: str = "",
             tasks: tuple[str, ...] = (), supports_stream: bool = False,
             license_name: str = "") -> dict[str, Any]:
    """The ``info`` block for an adapter's OpenAPI document."""
    lines = [
        f"**{name}** implements the OpenNVR AI Adapter Contract v"
        f"{CONTRACT_VERSION}. KAI-C polls `/health` and `/capabilities`, "
        f"sends inference to `/infer`, and scrapes `/metrics`.",
    ]
    if tasks:
        lines.append("Tasks advertised: " + ", ".join(f"`{t}`" for t in tasks) + ".")
    lines.append(
        "Streaming inference is available at `/infer/stream`; that surface is "
        "a WebSocket and so is described by the adapter's **AsyncAPI** "
        "document at `/asyncapi.json`."
        if supports_stream else
        "This adapter does not support streaming; `/infer/stream` answers 501."
    )
    lines.append(
        "This document is generated from the contract types the adapter "
        "already returns, so it cannot drift from the implementation."
    )
    info: dict[str, Any] = {
        "title": f"{name} adapter",
        "version": version,
        "description": "\n\n".join(lines),
        "x-opennvr-contract-version": CONTRACT_VERSION,
        "x-opennvr-tasks": list(tasks),
    }
    if license_name:
        info["license"] = {"name": license_name}
    if vendor:
        info["contact"] = {"name": vendor}
    return info


SECURITY_SCHEMES: dict[str, Any] = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "The adapter's token (`ADAPTER_AUTH_TOKEN`). KAI-C sends it on "
            "every call. `/health` and `/metrics` stay open so an operator "
            "can scrape a failing adapter."
        ),
    }
}

TAGS: list[dict[str, str]] = [
    {"name": "contract",
     "description": "The endpoints every adapter serves (§3)."},
    {"name": "inference",
     "description": "Running the model (§3.5, §6)."},
    {"name": "observability",
     "description": "What an operator scrapes (§3.4)."},
]


# ── AsyncAPI — the streaming surface ────────────────────────────────

_CLIENT_MESSAGES = {
    "handshake": HandshakeMessage,
    "frame": FrameMessage,
    "frameRef": FrameRefMessage,
    "resultAck": ResultAckMessage,
    "close": CloseMessage,
}

_ADAPTER_MESSAGES = {
    "handshakeAck": HandshakeAckMessage,
    "result": ResultMessage,
    "pause": PauseMessage,
    "resume": ResumeMessage,
    "stats": StatsMessage,
}


def adapter_asyncapi(
    name: str,
    version: str,
    *,
    supports_stream: bool = True,
    tasks: tuple[str, ...] = (),
    license_name: str = "",
) -> dict[str, Any]:
    """The **AsyncAPI 3.0** document for ``/infer/stream``.

    OpenAPI stops at the door of a WebSocket, and streaming is where a
    real-time adapter earns its keep: one session, one warm model, and
    backpressure that a request/response spec cannot express. Every
    message here is a contract type from :mod:`~.contract`, so this is
    generated rather than maintained."""
    messages = {
        key: _message(key, model)
        for key, model in {**_CLIENT_MESSAGES, **_ADAPTER_MESSAGES}.items()
    }
    schemas = {
        model.__name__: _schema(model)
        for model in {**_CLIENT_MESSAGES, **_ADAPTER_MESSAGES}.values()
    }

    doc: dict[str, Any] = {
        "asyncapi": "3.0.0",
        "info": {
            "title": f"{name} — streaming inference",
            "version": version,
            "description": (
                f"The `/infer/stream` WebSocket protocol for **{name}** (AI "
                f"Adapter Contract v{CONTRACT_VERSION} §6).\n\n"
                "A session opens with a `handshake` naming the camera, the "
                "task and the frame transport; the adapter answers "
                "`handshake_ack`. The client then sends `frame` (or "
                "`frame_ref` for shared memory) and receives `result`. The "
                "adapter may `pause` and `resume` the client as "
                "backpressure, and either side may `close`.\n\n"
                "One session means one warm model and one correlation id for "
                "the whole episode, which is what makes a sequence of frames "
                "traceable as one event rather than N unrelated inferences."
            ),
            **({"license": {"name": license_name}} if license_name else {}),
            "x-opennvr-contract-version": CONTRACT_VERSION,
            "x-opennvr-tasks": list(tasks),
        },
        "servers": {
            "adapter": {
                "host": "{host}:{port}",
                "pathname": "/infer/stream",
                "protocol": "ws",
                "description": "The adapter's own port on the internal network.",
                "variables": {
                    "host": {"default": name},
                    "port": {"default": "9000"},
                },
            }
        },
        "channels": {},
        "operations": {},
        "components": {"messages": messages, "schemas": schemas,
                       "securitySchemes": {"bearerAuth": {
                           "type": "http", "scheme": "bearer",
                           "description": "Sent as an Authorization header on "
                                          "the upgrade request."}}},
    }

    if not supports_stream:
        doc["info"]["description"] = (
            f"**{name}** does not support streaming inference — "
            f"`/infer/stream` answers 501 and callers should use "
            f"`POST /infer`. This document describes the protocol the "
            f"contract defines, for reference only."
        )
        return doc

    doc["channels"] = {
        "stream": {
            "address": "/infer/stream",
            "title": "Streaming inference session",
            "description": "One WebSocket connection = one camera's session.",
            "messages": {key: {"$ref": f"#/components/messages/{key}"}
                         for key in messages},
        }
    }
    doc["operations"] = {
        "send": {
            "action": "send",
            "channel": {"$ref": "#/channels/stream"},
            "title": "What the adapter sends",
            "summary": "Handshake acknowledgement, results, and backpressure.",
            "messages": [{"$ref": f"#/components/messages/{key}"}
                         for key in _ADAPTER_MESSAGES],
        },
        "receive": {
            "action": "receive",
            "channel": {"$ref": "#/channels/stream"},
            "title": "What the adapter receives",
            "summary": "The session handshake, frames, and acknowledgements.",
            "messages": [{"$ref": f"#/components/messages/{key}"}
                         for key in _CLIENT_MESSAGES],
        },
    }
    return doc


def _message(key: str, model: Any) -> dict[str, Any]:
    return {
        "name": model.__name__,
        "title": key,
        "summary": (model.__doc__ or "").strip().split("\n")[0],
        "contentType": _JSON,
        "payload": {"$ref": f"#/components/schemas/{model.__name__}"},
    }


def _schema(model: Any) -> dict[str, Any]:
    """A Pydantic model as a self-contained JSON Schema.

    ``$defs`` are inlined into the document's own schema map rather than
    left as local refs, so each message payload resolves on its own."""
    schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
    schema.pop("$defs", None)
    return schema


def contract_openapi_extras(*, name: str, version: str, vendor: str = "",
                            tasks: tuple[str, ...] = (),
                            supports_stream: bool = False,
                            license_name: str = "") -> dict[str, Any]:
    """The document-level additions FastAPI does not infer: the info
    block, the bearer scheme, and the tag descriptions."""
    return {
        "info": describe(name, version, vendor=vendor, tasks=tasks,
                         supports_stream=supports_stream,
                         license_name=license_name),
        "security_schemes": dict(SECURITY_SCHEMES),
        "tags": [dict(tag) for tag in TAGS],
    }


__all__ = [
    "CONTRACT_VERSION",
    "ERROR_STATUSES",
    "SECURITY_SCHEMES",
    "TAGS",
    "error_responses",
    "infer_request_body",
    "describe",
    "adapter_asyncapi",
    "contract_openapi_extras",
]
