# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Whisper ASR adapter — contract-compliant FastAPI service.

Implements the six mandatory endpoints from the AI Adapter Contract v1.
``/infer`` accepts both multipart (audio file field, §3.5 canonical
for binary data) and application/json (audio_b64 fallback).
``/infer/stream`` refuses with HTTP 501 — streaming ASR with partial-
result emission is its own design problem (overlap windows, VAD
gating) and lands in a follow-up commit.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.whisper.main:app --host 0.0.0.0 --port 9003

Conformance check:
    python -m conformance http://localhost:9003 --token $OPENNVR_ADAPTER_TOKEN
"""
from __future__ import annotations

import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse

from adapters.whisper.auth import AuthAndCorrelationMiddleware
from adapters.whisper.metrics import Metrics
from adapters.whisper.service import WhisperService, ServiceError
from app.interfaces.contract import (
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
)

logger = logging.getLogger("whisper-adapter")

# ── Service singletons ─────────────────────────────────────────────

_service: WhisperService | None = None
_metrics: Metrics = Metrics()


def get_service() -> WhisperService:
    if _service is None:  # pragma: no cover — guarded by lifespan
        raise RuntimeError("WhisperService not initialized; lifespan did not run")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = WhisperService()
    _service.load()
    _metrics.set_model_loaded(_service.is_ready())
    try:
        yield
    finally:
        _service = None
        _metrics.set_model_loaded(False)


app = FastAPI(
    title="OpenNVR Whisper ASR Adapter",
    version="1.0.0",
    description="Contract-compliant Whisper speech-to-text adapter (AI Adapter Contract v1).",
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


# ── Mandatory contract endpoints ───────────────────────────────────


@app.get("/health")
def health() -> Response:
    return JSONResponse(content=get_service().health().model_dump(mode="json"))


@app.get("/capabilities")
def capabilities() -> Response:
    return JSONResponse(content=get_service().capabilities().model_dump(mode="json"))


@app.get("/hardware/evaluation")
def hardware_evaluation() -> Response:
    return JSONResponse(content=get_service().hardware_evaluation().model_dump(mode="json"))


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return PlainTextResponse(content=_metrics.render(), media_type="text/plain; version=0.0.4")


# ── /infer ─────────────────────────────────────────────────────────


async def _parse_infer_request(request: Request) -> tuple[bytes, dict[str, Any]]:
    """Parse /infer body. Returns (audio_bytes, params).

    Same content-type story as YOLOv8: multipart with ``audio`` file
    field for the canonical §3.5 path, or application/json with
    ``audio_b64`` as a convenience fallback. Raises ``ValueError`` on
    any malformed input.
    """
    raw_ct = (request.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

    if raw_ct == "multipart/form-data":
        form = await request.form()
        audio_field = form.get("audio")
        if audio_field is None or not hasattr(audio_field, "read"):
            raise ValueError("Multipart body must include an 'audio' file field with audio bytes.")
        audio_bytes = await audio_field.read()
        params_field = form.get("params")
        params: dict[str, Any] = {}
        if params_field is not None and isinstance(params_field, str):
            try:
                params = json.loads(params_field)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in 'params' field: {exc}") from exc
            if not isinstance(params, dict):
                raise ValueError("'params' must be a JSON object.")
        return audio_bytes, params

    if raw_ct == "application/json":
        try:
            body = await request.json()
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        b64 = body.get("audio_b64")
        if not isinstance(b64, str) or not b64:
            raise ValueError("JSON body must include 'audio_b64' (base64-encoded audio).")
        try:
            audio_bytes = base64.b64decode(b64, validate=True)
        except Exception as exc:
            raise ValueError(f"'audio_b64' is not valid base64: {exc}") from exc
        params = {k: v for k, v in body.items() if k != "audio_b64"}
        return audio_bytes, params

    raise ValueError(
        f"Content-Type '{raw_ct or '(missing)'}' is not supported. "
        "Send 'multipart/form-data' or 'application/json'."
    )


@app.post("/infer")
async def infer(request: Request) -> Response:
    correlation_id = getattr(request.state, "correlation_id", "?")
    try:
        audio_bytes, params = await _parse_infer_request(request)
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
        result = get_service().infer_bytes(audio_bytes, params)
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
        "infer ok correlation_id=%s audio_bytes=%d latency_ms=%d",
        correlation_id,
        len(audio_bytes),
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


# ── /infer/stream — not supported ──────────────────────────────────


@app.get("/infer/stream")
@app.post("/infer/stream")
def infer_stream_http_probe() -> Response:
    """§3.6 — Whisper doesn't do realtime streaming ASR in v1.
    Streaming requires overlap-window decoding + partial-result
    emission + VAD gating; it's its own design problem and lands in
    a follow-up. Refuse with HTTP 501 + the canonical envelope so
    KAI-C knows up front."""
    envelope = FailureEnvelope(
        error=ErrorDetail(
            category=ErrorCategory.NOT_SUPPORTED,
            code="stream_not_supported",
            message="Whisper does not yet support streaming ASR; use POST /infer.",
            transient=False,
            details={},
        )
    )
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=envelope.model_dump(mode="json"),
    )
