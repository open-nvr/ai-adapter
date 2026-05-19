# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Piper TTS adapter — contract-compliant FastAPI service.

Implements the six mandatory endpoints from the AI Adapter Contract v1.
See ``open-nvr/docs/AI_ADAPTER_CONTRACT.md`` and
``ai-adapter/app/interfaces/contract.py`` for the spec and types.

Run locally:
    OPENNVR_ADAPTER_TOKEN=secret \\
    python -m uvicorn adapters.piper.main:app --host 0.0.0.0 --port 9001

Conformance check:
    python -m conformance http://localhost:9001
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, WebSocket, status
from fastapi.responses import JSONResponse, PlainTextResponse

from adapters.piper.auth import AuthAndCorrelationMiddleware
from adapters.piper.metrics import Metrics
from adapters.piper.service import PiperService, ServiceError
from app.interfaces.contract import (
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
)

logger = logging.getLogger("piper-adapter")

# ── Service singletons ─────────────────────────────────────────────

_service: PiperService | None = None
_metrics: Metrics = Metrics()


def get_service() -> PiperService:
    """FastAPI dependency — returns the loaded service singleton."""
    if _service is None:  # pragma: no cover — guarded by lifespan
        raise RuntimeError("PiperService not initialized; lifespan did not run")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Piper voice once on startup; clear on shutdown.

    ``service.load()`` catches its own exceptions and reports the
    failure via /health's status field — it does NOT raise. We only
    flip the ``adapter_model_loaded`` gauge to 1 when the load
    actually succeeded; otherwise it stays at 0 so dashboards don't
    show a false-positive "ready" signal for a broken adapter.
    """
    global _service
    _service = PiperService()
    _service.load()
    _metrics.set_model_loaded(_service.is_ready())
    try:
        yield
    finally:
        _service = None
        _metrics.set_model_loaded(False)


app = FastAPI(
    title="OpenNVR Piper TTS Adapter",
    version="1.0.0",
    description="Contract-compliant Piper TTS adapter (AI Adapter Contract v1).",
    lifespan=lifespan,
)

app.add_middleware(AuthAndCorrelationMiddleware)


# ── /health ────────────────────────────────────────────────────────


@app.get("/health")
def health() -> Response:
    svc = get_service()
    body = svc.health().model_dump(mode="json")
    return JSONResponse(content=body)


# ── /capabilities ──────────────────────────────────────────────────


@app.get("/capabilities")
def capabilities() -> Response:
    svc = get_service()
    body = svc.capabilities().model_dump(mode="json")
    return JSONResponse(content=body)


# ── /hardware/evaluation ───────────────────────────────────────────


@app.get("/hardware/evaluation")
def hardware_evaluation() -> Response:
    svc = get_service()
    body = svc.hardware_evaluation().model_dump(mode="json")
    return JSONResponse(content=body)


# ── /metrics ───────────────────────────────────────────────────────


@app.get("/metrics")
def metrics_endpoint() -> Response:
    return PlainTextResponse(content=_metrics.render(), media_type="text/plain; version=0.0.4")


# ── /voices (extra endpoint advertised in /capabilities.endpoints.extra) ──


@app.get("/voices")
def list_voices() -> Response:
    svc = get_service()
    return JSONResponse(content=svc.list_voices())


# ── /infer ─────────────────────────────────────────────────────────


def _transport_error(code: str, message: str, *, status_code: int = 400, **details) -> JSONResponse:
    """Build a FailureEnvelope JSONResponse for a request-validation error."""
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


async def _parse_infer_payload(request: Request) -> dict:
    """
    Parse the /infer request body into a dict, per §3.5.

    Adapters MUST accept ``multipart/form-data`` and MAY also accept
    ``application/json``. Piper supports both — for text-only input,
    a multipart request looks like a single ``params`` field
    containing a JSON object (matching the §3.5 example).

    Raises ``ValueError`` with a user-facing message on any malformed
    input; the caller translates that into a TRANSPORT_ERROR envelope.
    """
    raw_ct = (request.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

    if raw_ct == "application/json":
        try:
            payload = await request.json()
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    if raw_ct == "multipart/form-data":
        form = await request.form()
        # §3.5 example: a multipart request carries a ``params`` field
        # with a JSON blob containing the adapter-specific inputs.
        # For Piper, individual fields (text, voice, length_scale...)
        # are also accepted directly so curl examples stay readable.
        params_field = form.get("params")
        if params_field is not None:
            import json as _json
            try:
                parsed = _json.loads(params_field) if isinstance(params_field, str) else None
            except Exception as exc:
                raise ValueError(f"Invalid JSON in 'params' field: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("'params' must be a JSON object.")
            return parsed
        # Per-field fallback: collect known scalar fields from the form.
        payload: dict[str, object] = {}
        for field in ("text", "voice"):
            value = form.get(field)
            if isinstance(value, str):
                payload[field] = value
        for numeric_field in ("length_scale", "noise_scale", "noise_w"):
            value = form.get(numeric_field)
            if isinstance(value, str) and value:
                try:
                    payload[numeric_field] = float(value)
                except ValueError as exc:
                    raise ValueError(f"Field '{numeric_field}' must be numeric.") from exc
        if not payload:
            raise ValueError(
                "Multipart body must include either a 'params' JSON field "
                "or at least a 'text' form field."
            )
        return payload

    raise ValueError(
        f"Content-Type '{raw_ct or '(missing)'}' is not supported. "
        "Send 'multipart/form-data' or 'application/json'."
    )


@app.post("/infer")
async def infer(request: Request) -> Response:
    """
    Run TTS inference.

    Per §3.5 the contract requires multipart support; adapters MAY also
    accept application/json. Piper accepts both and treats them as
    equivalent. ``capabilities.endpoints.infer.input_content_types``
    declares the accepted set so KAI-C knows.
    """
    correlation_id = getattr(request.state, "correlation_id", "?")

    try:
        payload = await _parse_infer_payload(request)
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
        svc = get_service()
        result = svc.infer(payload)
    except ServiceError as exc:
        latency = time.monotonic() - started
        outcome = {
            ErrorCategory.MODEL_ERROR: "model_error",
            ErrorCategory.TRANSPORT_ERROR: "transport_error",
            ErrorCategory.PROVIDER_ERROR: "provider_error",
            ErrorCategory.PERMISSION_DENIED: "refused",
            ErrorCategory.OVERLOADED: "refused",
            ErrorCategory.NOT_SUPPORTED: "refused",
        }.get(exc.category, "model_error")
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
        "infer ok correlation_id=%s text_chars=%d latency_ms=%d",
        correlation_id,
        len(payload.get("text", "")),
        int(latency * 1000),
    )
    return JSONResponse(content=result.model_dump(mode="json"))


# ── /infer/stream — Piper does not stream ──────────────────────────


@app.websocket("/infer/stream")
async def infer_stream(websocket: WebSocket) -> None:
    """
    Piper is one-shot TTS — there is no streaming inference path.
    Per §3.6 we refuse the WebSocket upgrade with HTTP 501 *before*
    accepting the connection. The 4xxx close codes in §6.5 apply only
    to adapters that accepted the upgrade.
    """
    # Starlette's WebSocket.close() before .accept() rejects the upgrade.
    # Using a 4xxx code here is wrong per §6.5; the correct refusal is
    # the HTTP-side 501, which we emit via the dedicated route below.
    await websocket.close(code=1008, reason="streaming not supported")


# Some clients send a plain HTTP GET to /infer/stream first to probe.
# Per §3.6, that probe gets HTTP 501 with the canonical error code.
@app.get("/infer/stream")
@app.post("/infer/stream")
def infer_stream_http_probe() -> Response:
    envelope = FailureEnvelope(
        error=ErrorDetail(
            category=ErrorCategory.NOT_SUPPORTED,
            code="stream_not_supported",
            message="Piper does not support streaming inference; use POST /infer.",
            transient=False,
            details={},
        )
    )
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=envelope.model_dump(mode="json"),
    )
