# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
YOLOv8 object-detection adapter — contract-compliant FastAPI service.

Implements the six mandatory endpoints from the AI Adapter Contract v1,
including the full §6 WebSocket streaming protocol. See
``open-nvr/docs/AI_ADAPTER_CONTRACT.md`` for the spec and
``ai-adapter/app/interfaces/contract.py`` for the typed wire shapes.

Concurrency model (v1)
----------------------
The WebSocket loop is **serial** — frames are processed one at a time,
in receive order, and ``max_inflight=1`` is advertised in the
handshake_ack. This is the simplest correct implementation: every
result message echoes the frame's ``seq``, no reordering, no
backpressure logic. Concurrent inference within a single stream lands
in a follow-up alongside the shared-memory fast path. Across streams,
KAI-C's per-camera fair queuing (§9, declared via
``scheduling.fair_queuing="per_camera"`` in capabilities) handles the
multi-camera case.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.yolov8.main:app --host 0.0.0.0 --port 9002

Conformance check:
    python -m conformance http://localhost:9002 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, PlainTextResponse

from adapters.yolov8.auth import (
    AuthAndCorrelationMiddleware,
    websocket_auth_failure,
    _expected_token,
)
from adapters.yolov8.metrics import Metrics
from adapters.yolov8.service import YoloV8Service, ServiceError
from app.interfaces.contract import (
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    FrameMessage,
    FrameTransport,
    HandshakeMessage,
    HandshakeAckMessage,
    StreamCloseCode,
)

logger = logging.getLogger("yolov8-adapter")

# ── Service singletons ─────────────────────────────────────────────

_service: YoloV8Service | None = None
_metrics: Metrics = Metrics()


def get_service() -> YoloV8Service:
    if _service is None:  # pragma: no cover — guarded by lifespan
        raise RuntimeError("YoloV8Service not initialized; lifespan did not run")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = YoloV8Service()
    _service.load()
    _metrics.set_model_loaded(_service.is_ready())
    try:
        yield
    finally:
        _service = None
        _metrics.set_model_loaded(False)


app = FastAPI(
    title="OpenNVR YOLOv8 Object-Detection Adapter",
    version="1.0.0",
    description="Contract-compliant YOLOv8 adapter (AI Adapter Contract v1).",
    lifespan=lifespan,
)

app.add_middleware(AuthAndCorrelationMiddleware)


# ── Helpers ────────────────────────────────────────────────────────


def _transport_error(code: str, message: str, *, status_code: int = 400, **details) -> JSONResponse:
    envelope = FailureEnvelope(
        error=ErrorDetail(
            category=ErrorCategory.TRANSPORT_ERROR,
            code=code,
            message=message,
            transient=False,
            details=details,
        )
    )
    _metrics.record_infer("transport_error", 0.0)
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


# ── /health ────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Response:
    return JSONResponse(content=get_service().health().model_dump(mode="json"))


# ── /capabilities ──────────────────────────────────────────────────


@app.get("/capabilities")
def capabilities() -> Response:
    return JSONResponse(content=get_service().capabilities().model_dump(mode="json"))


# ── /hardware/evaluation ───────────────────────────────────────────


@app.get("/hardware/evaluation")
def hardware_evaluation() -> Response:
    return JSONResponse(content=get_service().hardware_evaluation().model_dump(mode="json"))


# ── /metrics ───────────────────────────────────────────────────────


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return PlainTextResponse(content=_metrics.render(), media_type="text/plain; version=0.0.4")


# ── /infer (HTTP) ──────────────────────────────────────────────────


async def _parse_infer_request(request: Request) -> tuple[bytes, dict[str, Any]]:
    """
    Parse /infer body. Returns (frame_bytes, params).

    Supports both content types declared in /capabilities:

    * ``multipart/form-data`` with ``frame`` file field carrying the
      image bytes and optional ``params`` JSON field (the §3.5 canonical
      pattern).
    * ``application/json`` with ``frame_b64`` (base64-encoded image)
      and the rest of the keys treated as params (convenience).

    Raises ``ValueError`` on any malformed input.
    """
    raw_ct = (request.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

    if raw_ct == "multipart/form-data":
        form = await request.form()
        frame_field = form.get("frame")
        if frame_field is None or not hasattr(frame_field, "read"):
            raise ValueError("Multipart body must include a 'frame' file field with image bytes.")
        frame_bytes = await frame_field.read()
        params_field = form.get("params")
        params: dict[str, Any] = {}
        if params_field is not None and isinstance(params_field, str):
            try:
                params = json.loads(params_field)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in 'params' field: {exc}") from exc
            if not isinstance(params, dict):
                raise ValueError("'params' must be a JSON object.")
        return frame_bytes, params

    if raw_ct == "application/json":
        try:
            body = await request.json()
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        b64 = body.get("frame_b64")
        if not isinstance(b64, str) or not b64:
            raise ValueError("JSON body must include 'frame_b64' (base64-encoded image).")
        try:
            frame_bytes = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise ValueError(f"'frame_b64' is not valid base64: {exc}") from exc
        params = {k: v for k, v in body.items() if k != "frame_b64"}
        return frame_bytes, params

    raise ValueError(
        f"Content-Type '{raw_ct or '(missing)'}' is not supported. "
        "Send 'multipart/form-data' or 'application/json'."
    )


@app.post("/infer")
async def infer(request: Request) -> Response:
    correlation_id = getattr(request.state, "correlation_id", "?")
    try:
        frame_bytes, params = await _parse_infer_request(request)
    except ValueError as exc:
        message = str(exc)
        if "Content-Type" in message:
            return _transport_error(
                "unsupported_content_type",
                message,
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                received=(request.headers.get("Content-Type") or "missing"),
            )
        return _transport_error("malformed_input", message)

    _metrics.inc_inflight()
    started = time.monotonic()
    try:
        result = get_service().infer_bytes(frame_bytes, params)
    except ServiceError as exc:
        latency = time.monotonic() - started
        outcome = _outcome_for_category(exc.category)
        _metrics.record_infer(outcome, latency)
        logger.info(
            "infer failed correlation_id=%s category=%s code=%s latency_ms=%d",
            correlation_id,
            exc.category.value,
            exc.code,
            int(latency * 1000),
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.envelope().model_dump(mode="json"),
        )
    finally:
        _metrics.dec_inflight()

    latency = time.monotonic() - started
    _metrics.record_infer("ok", latency)
    logger.info(
        "infer ok correlation_id=%s frame_bytes=%d latency_ms=%d",
        correlation_id,
        len(frame_bytes),
        int(latency * 1000),
    )
    return JSONResponse(content=result.model_dump(mode="json"))


_CATEGORY_TO_OUTCOME: dict[ErrorCategory, str] = {
    ErrorCategory.MODEL_ERROR: "model_error",
    ErrorCategory.TRANSPORT_ERROR: "transport_error",
    ErrorCategory.PROVIDER_ERROR: "provider_error",
    ErrorCategory.PERMISSION_DENIED: "refused",
    ErrorCategory.OVERLOADED: "refused",
    ErrorCategory.NOT_SUPPORTED: "refused",
}


def _outcome_for_category(category: ErrorCategory) -> str:
    return _CATEGORY_TO_OUTCOME.get(category, "model_error")


def _outcome_for_category_value(value: str) -> str:
    """Same mapping but keyed by the wire-string. Used in the WS loop
    where the service has already serialized the error envelope."""
    try:
        return _outcome_for_category(ErrorCategory(value))
    except ValueError:
        return "model_error"


# ── /infer/stream (WebSocket) ──────────────────────────────────────


@app.websocket("/infer/stream")
async def infer_stream(websocket: WebSocket) -> None:
    """
    §6 WebSocket streaming protocol.

    Flow:
      1. Accept the upgrade (or refuse if auth fails) — close 4001.
      2. Receive the first message; it must be a `handshake`. Reply
         with `handshake_ack` echoing the negotiated transport.
      3. Loop: receive either a control message (pause/resume/close/stats)
         or a `frame` JSON message immediately followed by a binary
         WS message carrying the frame bytes. Send back a `result`
         message after inference. Repeat until close.
      4. On any protocol violation or inference fatal, close with the
         appropriate §6.5 code.
    """
    # Auth check — BaseHTTPMiddleware does NOT run on WS upgrades, so we
    # do it here. dev mode (no env token) is allowed-open same as HTTP.
    auth_header = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    failure = websocket_auth_failure(_expected_token(), auth_header)
    if failure is not None:
        # Audit-log the rejection so brute-force probes leave a trail.
        # Token bytes are deliberately omitted from the log line.
        logger.info(
            "ws auth rejected code=%s remote=%s",
            failure,
            websocket.client.host if websocket.client else "?",
        )
        await websocket.close(
            code=StreamCloseCode.POLICY_REFUSED.value,
            reason=f"auth: {failure}",
        )
        return

    await websocket.accept()
    _metrics.inc_stream_connection()
    session_id = uuid.uuid4().hex

    try:
        # ── Handshake ──────────────────────────────────────────────
        try:
            first_raw = await websocket.receive_text()
            handshake = HandshakeMessage.model_validate(json.loads(first_raw))
        except (WebSocketDisconnect, json.JSONDecodeError):
            await websocket.close(code=StreamCloseCode.POLICY_REFUSED.value, reason="bad handshake")
            return
        except Exception as exc:
            logger.info("handshake validation failed: %s", exc)
            await websocket.close(code=StreamCloseCode.POLICY_REFUSED.value, reason="bad handshake")
            return

        # Reject shared-memory transport requests with a websocket
        # fallback — §6.1 allows the adapter to downgrade the offered
        # transport in the ack.
        negotiated_transport = FrameTransport.WEBSOCKET
        ack = HandshakeAckMessage(
            frame_transport=negotiated_transport,
            result_sink="websocket",  # NATS sink lands with B1
            max_inflight=1,            # serial inference for v1 — see module docstring
            session_id=session_id,
        )
        await websocket.send_text(json.dumps(ack.model_dump(mode="json")))

        # Service readiness check after the handshake — clients should
        # see a typed close rather than mysterious hangs if weights
        # never loaded.
        svc = get_service()
        if not svc.is_ready():
            await websocket.close(
                code=StreamCloseCode.MODEL_ERROR.value,
                reason="model not loaded",
            )
            return

        logger.info(
            "stream open session_id=%s client_id=%s camera_id=%s",
            session_id,
            handshake.client_id,
            handshake.camera_id or "-",
        )

        # ── Message loop ───────────────────────────────────────────
        paused = False
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("stream closed (client disconnect) session_id=%s", session_id)
                return

            if msg.get("type") == "websocket.disconnect":
                return

            text = msg.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.close(
                        code=StreamCloseCode.POLICY_REFUSED.value,
                        reason="non-JSON control message",
                    )
                    return
                msg_type = payload.get("type")

                if msg_type == "close":
                    return
                if msg_type == "pause":
                    paused = True
                    continue
                if msg_type == "resume":
                    paused = False
                    continue
                if msg_type == "stats":
                    await websocket.send_text(json.dumps({
                        "type": "stats",
                        "inflight": 0,
                        "queue_depth": 0,
                        "fps": 0.0,
                    }))
                    continue
                if msg_type == "frame":
                    try:
                        frame_meta = FrameMessage.model_validate(payload)
                    except Exception:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="bad frame metadata",
                        )
                        return
                    # Next message must be the binary frame payload.
                    try:
                        binary_msg = await websocket.receive()
                    except WebSocketDisconnect:
                        return
                    frame_bytes = binary_msg.get("bytes")
                    if not isinstance(frame_bytes, (bytes, bytearray)) or not frame_bytes:
                        await websocket.close(
                            code=StreamCloseCode.POLICY_REFUSED.value,
                            reason="frame must be followed by binary message",
                        )
                        return

                    if paused:
                        # Per §6.4 — once paused, drop frames until resume.
                        continue

                    _metrics.inc_inflight()
                    try:
                        result_dict = svc.infer_frame_for_stream(
                            bytes(frame_bytes),
                            seq=frame_meta.seq,
                            ts_ms=frame_meta.ts_ms,
                        )
                        await websocket.send_text(json.dumps(result_dict))
                        # Real latency from the service result, real
                        # outcome from the shape of `result.result`.
                        # Error-shaped results carry status=error +
                        # error.category — count them in the matching
                        # bucket rather than silently as "ok".
                        latency_seconds = result_dict.get("inference_ms", 0) / 1000.0
                        result_payload = result_dict.get("result") or {}
                        if isinstance(result_payload, dict) and result_payload.get("status") == "error":
                            category_value = (result_payload.get("error") or {}).get("category", "")
                            outcome = _outcome_for_category_value(category_value)
                        else:
                            outcome = "ok"
                        _metrics.record_infer(outcome, latency_seconds)
                    finally:
                        _metrics.dec_inflight()
                    continue

                # Unknown JSON message types are a protocol violation.
                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason=f"unknown message type: {msg_type}",
                )
                return

            # If the message had no text but has bytes, that's a stray
            # binary frame outside the metadata-then-bytes pattern.
            if msg.get("bytes") is not None:
                await websocket.close(
                    code=StreamCloseCode.POLICY_REFUSED.value,
                    reason="binary frame without preceding frame metadata",
                )
                return

    finally:
        _metrics.dec_stream_connection()
