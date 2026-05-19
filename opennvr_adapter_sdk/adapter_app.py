# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
``AdapterApp`` — wraps an ``AdapterService`` in a FastAPI app with
all six mandatory contract endpoints.

This is the load-bearing piece of the SDK. The adapter author writes
only ``AdapterService`` (4 abstract methods + optional streaming);
the SDK does:

* §3.x endpoint routes (/health, /capabilities, /hardware/evaluation,
  /metrics, /infer, /infer/stream)
* Auth + correlation_id middleware (§3.8)
* Multipart + JSON body parsing on /infer with configurable file-
  field name and max-bytes limit
* §7 ``FailureEnvelope`` translation from ``ServiceError`` exceptions
* Lifespan: service.load() at startup, gauge updates
* §3.4 Prometheus metrics with per-adapter latency buckets
* WebSocket auth check + delegation to service.handle_stream (or
  HTTP 501 refusal when supports_stream=False)

What the SDK does NOT do (intentionally):

* It does NOT call the adapter's underlying ML framework — that's
  ``AdapterService.infer()``'s job. The SDK only deals in wire
  shapes, not domain semantics.
* It does NOT enforce sandboxing — that's KAI-C's job per §8.
* It does NOT cache results — adapters that want to cache (e.g.,
  idempotency-key support) implement it themselves.
"""
from __future__ import annotations

import base64
import enum
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Sequence

from fastapi import FastAPI, Request, Response, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from app.interfaces.contract import (
    AdapterInfo,
    CapabilitiesResponse,
    Cost,
    EndpointsInfo,
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    FairQueuing,
    HealthResponse,
    InferEndpointInfo,
    Permissions,
    Scheduling,
    StreamCloseCode,
    StreamEndpointInfo,
)
from opennvr_adapter_sdk.auth import (
    AuthAndCorrelationMiddleware,
    expected_token,
    websocket_auth_failure,
)
from opennvr_adapter_sdk.metrics import DEFAULT_LATENCY_BUCKETS_SECONDS, Metrics
from opennvr_adapter_sdk.service import AdapterService, ServiceError

logger = logging.getLogger("opennvr-adapter-sdk")


class BodyShape(str, enum.Enum):
    """Hint to the SDK's /infer body parser about what to expect.

    * ``TEXT``    — JSON only; no binary upload. (e.g., Piper TTS)
    * ``IMAGE``   — multipart with a binary frame field + JSON params,
                     or JSON with ``frame_b64``. (e.g., YOLOv8)
    * ``AUDIO``   — multipart with a binary audio field + JSON params,
                     or JSON with ``audio_b64``. (e.g., Whisper)
    * ``GENERIC`` — multipart with a single binary file field + JSON
                     params, or JSON with the binary as ``data_b64``.
                     Use when you don't fit the named patterns.
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    GENERIC = "generic"


_BODY_SHAPE_FILE_FIELD: dict[BodyShape, str] = {
    BodyShape.IMAGE: "frame",
    BodyShape.AUDIO: "audio",
    BodyShape.GENERIC: "data",
}

_BODY_SHAPE_B64_FIELD: dict[BodyShape, str] = {
    BodyShape.IMAGE: "frame_b64",
    BodyShape.AUDIO: "audio_b64",
    BodyShape.GENERIC: "data_b64",
}


class AdapterApp:
    """Wraps an ``AdapterService`` in a contract-compliant FastAPI app.

    The ``fastapi_app`` attribute is what ``uvicorn`` loads:

    .. code-block:: python

        app = AdapterApp(
            service=MyService(),
            name="my-adapter",
            version="1.0.0",
            vendor="me",
            license="MIT",
            tasks_advertised=["my_task"],
        ).fastapi_app
    """

    def __init__(
        self,
        *,
        service: AdapterService | None = None,
        service_factory: Any = None,
        name: str,
        version: str,
        vendor: str,
        license: str,
        tasks_advertised: Sequence[str],
        body_shape: BodyShape = BodyShape.TEXT,
        max_body_bytes: int = 32 * 1024 * 1024,
        permissions: Permissions | None = None,
        scheduling: Scheduling | None = None,
        cost: Cost | None = None,
        model_card_url: str | None = None,
        supported_contract_versions: Sequence[str] = ("1",),
        extra_input_content_types: Sequence[str] = (),
        latency_buckets_seconds: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_SECONDS,
        cors_origins: Sequence[str] = ("*",),
        supports_stream: bool = False,
        stream_max_concurrent: int = 0,
        stream_supports_shared_memory: bool = False,
    ) -> None:
        # ``service`` is the eager case (production typical); the
        # factory is invoked at lifespan startup and useful when the
        # adapter needs late binding (env-var-driven config,
        # test-fixture monkeypatching of __init__, etc.). Exactly one
        # must be supplied.
        if (service is None) == (service_factory is None):
            raise ValueError("AdapterApp requires exactly one of service= or service_factory=.")
        self._service: AdapterService | None = service
        self._service_factory = service_factory
        self._supports_stream = supports_stream
        self._stream_max_concurrent = stream_max_concurrent
        self._stream_supports_shared_memory = stream_supports_shared_memory
        self._name = name
        self._version = version
        self._vendor = vendor
        self._license = license
        self._tasks_advertised = list(tasks_advertised)
        self._body_shape = body_shape
        self._max_body_bytes = max_body_bytes
        self._permissions = permissions or Permissions()
        self._scheduling = scheduling or Scheduling()
        self._cost = cost or Cost()
        self._model_card_url = model_card_url
        self._supported_contract_versions = list(supported_contract_versions)
        self._started_at_dt = datetime.now(timezone.utc)
        self._started_at_mono = time.monotonic()
        self._metrics = Metrics(latency_buckets_seconds=latency_buckets_seconds)

        self._input_content_types = self._compute_input_content_types(extra_input_content_types)

        self.fastapi_app: FastAPI = self._build_fastapi_app(cors_origins)

    # ── Public read-only accessors ─────────────────────────────────

    @property
    def service(self) -> AdapterService:
        assert self._service is not None, "service not built — lifespan has not run yet"
        return self._service

    def replace_service(self, service: AdapterService) -> None:
        """Test-fixture hook: swap the service after construction.
        Production code uses ``service=`` or ``service_factory=`` and
        does not call this."""
        self._service = service

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    # ── FastAPI construction ───────────────────────────────────────

    def _compute_input_content_types(self, extras: Sequence[str]) -> list[str]:
        # §3.5 — all adapters MUST accept multipart, even text-only
        # ones (operators may submit text via `params` form field for
        # clients that don't easily speak JSON). All adapters MAY
        # additionally accept JSON; the SDK always advertises both.
        types: list[str] = ["multipart/form-data", "application/json"]
        for t in extras:
            if t not in types:
                types.append(t)
        return types

    def _build_fastapi_app(self, cors_origins: Sequence[str]) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            # Build the service now if a factory was provided — this
            # is the late-binding path test fixtures rely on (they
            # monkey-patch __init__ between module load and TestClient
            # __enter__).
            if self._service is None and self._service_factory is not None:
                self._service = self._service_factory()
            assert self._service is not None
            self._service.load()
            self._metrics.set_model_loaded(self._service.is_ready())
            try:
                yield
            finally:
                self._metrics.set_model_loaded(False)

        app = FastAPI(
            title=f"{self._name} adapter",
            version=self._version,
            description=f"AI Adapter Contract v1 service: {self._name}",
            lifespan=lifespan,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_middleware(AuthAndCorrelationMiddleware)
        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI) -> None:
        @app.get("/health")
        def health() -> Response:
            return JSONResponse(content=self._build_health().model_dump(mode="json"))

        @app.get("/capabilities")
        def capabilities() -> Response:
            return JSONResponse(content=self._build_capabilities().model_dump(mode="json"))

        @app.get("/hardware/evaluation")
        def hardware_evaluation() -> Response:
            return JSONResponse(
                content=self._service.hardware_evaluation().model_dump(mode="json")
            )

        @app.get("/metrics")
        def metrics_endpoint() -> Response:
            return PlainTextResponse(
                content=self._metrics.render(),
                media_type="text/plain; version=0.0.4",
            )

        @app.post("/infer")
        async def infer(request: Request) -> Response:
            return await self._handle_infer(request)

        # WebSocket — only if the adapter declares streaming support.
        if self._supports_stream:
            @app.websocket("/infer/stream")
            async def infer_stream(websocket: WebSocket) -> None:
                await self._handle_stream(websocket)
        else:
            @app.get("/infer/stream")
            @app.post("/infer/stream")
            def infer_stream_probe() -> Response:
                envelope = FailureEnvelope(
                    error=ErrorDetail(
                        category=ErrorCategory.NOT_SUPPORTED,
                        code="stream_not_supported",
                        message=f"{self._name} does not support streaming inference; use POST /infer.",
                        transient=False,
                        details={},
                    )
                )
                return JSONResponse(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    content=envelope.model_dump(mode="json"),
                )

    # ── /health, /capabilities builders ────────────────────────────

    def _build_health(self) -> HealthResponse:
        info = self._service.model_info()
        if self._service.is_ready():
            from app.interfaces.contract import HealthStatus
            status_value = HealthStatus.OK
        else:
            from app.interfaces.contract import HealthStatus
            status_value = HealthStatus.LOADING
        return HealthResponse(
            status=status_value,
            adapter_name=self._name,
            adapter_version=self._version,
            model_name=info.name,
            model_version=info.version,
            started_at=self._started_at_dt,
            uptime_seconds=int(time.monotonic() - self._started_at_mono),
        )

    def _build_capabilities(self) -> CapabilitiesResponse:
        infer = InferEndpointInfo(
            supported=True,
            input_content_types=list(self._input_content_types),
        )
        if self._supports_stream:
            stream = StreamEndpointInfo(
                supported=True,
                max_concurrent_streams=self._stream_max_concurrent,
                supports_shared_memory=self._stream_supports_shared_memory,
                shared_memory_protocol_version=(
                    1 if self._stream_supports_shared_memory else None
                ),
            )
        else:
            stream = StreamEndpointInfo(supported=False, max_concurrent_streams=0)
        return CapabilitiesResponse(
            adapter=AdapterInfo(
                name=self._name,
                version=self._version,
                vendor=self._vendor,
                license=self._license,
                model_card_url=self._model_card_url,
                supported_contract_versions=list(self._supported_contract_versions),
            ),
            model=self._service.model_info(),
            endpoints=EndpointsInfo(infer=infer, infer_stream=stream),
            tasks_advertised=list(self._tasks_advertised),
            permissions=self._permissions,
            scheduling=self._scheduling,
            cost=self._cost,
        )

    # ── /infer handling ────────────────────────────────────────────

    async def _handle_infer(self, request: Request) -> Response:
        correlation_id = getattr(request.state, "correlation_id", "?")
        try:
            payload = await self._parse_infer_body(request)
        except ValueError as exc:
            message = str(exc)
            if "Content-Type" in message:
                return self._transport_error(
                    "unsupported_content_type",
                    message,
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    received=(request.headers.get("Content-Type") or "missing"),
                )
            return self._transport_error("malformed_input", message)

        self._metrics.inc_inflight()
        started = time.monotonic()
        try:
            result = self._service.infer(payload)
        except ServiceError as exc:
            latency = time.monotonic() - started
            self._metrics.record_infer(_outcome_for_category(exc.category), latency)
            logger.info(
                "infer failed adapter=%s correlation_id=%s category=%s code=%s latency_ms=%d",
                self._name, correlation_id, exc.category.value, exc.code,
                int(latency * 1000),
            )
            return JSONResponse(
                status_code=exc.http_status,
                content=exc.envelope().model_dump(mode="json"),
            )
        finally:
            self._metrics.dec_inflight()

        latency = time.monotonic() - started
        self._metrics.record_infer("ok", latency)
        logger.info(
            "infer ok adapter=%s correlation_id=%s latency_ms=%d",
            self._name, correlation_id, int(latency * 1000),
        )
        return JSONResponse(content=result.model_dump(mode="json"))

    async def _parse_infer_body(self, request: Request) -> dict[str, Any]:
        """Parse the request body into the dict that AdapterService.infer
        receives. Behavior depends on ``body_shape``:

        * TEXT → application/json only. Body must be a JSON object.
        * IMAGE/AUDIO/GENERIC → multipart with a binary file field OR
          application/json with a base64 ``<shape>_b64`` field.

        Returns a normalized dict where binary content (if any) is at
        ``payload["__file__"]`` as bytes, and the rest of the
        parameters are flat top-level keys.

        Raises ``ValueError`` on any malformed input.
        """
        raw_ct = (request.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

        if self._body_shape == BodyShape.TEXT:
            return await self._parse_text_body(request, raw_ct)

        # IMAGE / AUDIO / GENERIC
        file_field = _BODY_SHAPE_FILE_FIELD[self._body_shape]
        b64_field = _BODY_SHAPE_B64_FIELD[self._body_shape]

        if raw_ct == "multipart/form-data":
            form = await request.form()
            file_value = form.get(file_field)
            if file_value is None or not hasattr(file_value, "read"):
                raise ValueError(
                    f"Multipart body must include a {file_field!r} file field with binary content."
                )
            content = await file_value.read()
            if len(content) > self._max_body_bytes:
                raise ValueError(
                    f"Body exceeds {self._max_body_bytes}-byte limit ({len(content)} received)."
                )
            params: dict[str, Any] = {}
            params_field = form.get("params")
            if params_field is not None and isinstance(params_field, str):
                try:
                    params = json.loads(params_field)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in 'params' field: {exc}") from exc
                if not isinstance(params, dict):
                    raise ValueError("'params' must be a JSON object.")
            params["__file__"] = content
            return params

        if raw_ct == "application/json":
            try:
                body = await request.json()
            except Exception as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            b64 = body.get(b64_field)
            if not isinstance(b64, str) or not b64:
                raise ValueError(
                    f"JSON body must include {b64_field!r} (base64-encoded binary)."
                )
            try:
                content = base64.b64decode(b64, validate=True)
            except Exception as exc:
                raise ValueError(f"{b64_field!r} is not valid base64: {exc}") from exc
            if len(content) > self._max_body_bytes:
                raise ValueError(
                    f"Body exceeds {self._max_body_bytes}-byte limit ({len(content)} received)."
                )
            params = {k: v for k, v in body.items() if k != b64_field}
            params["__file__"] = content
            return params

        raise ValueError(
            f"Content-Type '{raw_ct or '(missing)'}' is not supported. "
            f"Send 'multipart/form-data' or 'application/json'."
        )

    async def _parse_text_body(self, request: Request, raw_ct: str) -> dict[str, Any]:
        """Parse a TEXT-body-shape request. Multipart and JSON both
        accepted (§3.5 mandates multipart for every adapter).

        * JSON body → dict directly.
        * Multipart → if a ``params`` text field is present, parse as
          JSON; otherwise treat all string form fields as top-level
          keys. ``length_scale``/``noise_scale``/etc. coerce to float
          (the convention text-only adapters expect).
        """
        if raw_ct == "application/json":
            try:
                body = await request.json()
            except Exception as exc:
                raise ValueError(f"Invalid JSON body: {exc}") from exc
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            return body

        if raw_ct == "multipart/form-data":
            form = await request.form()
            params_field = form.get("params")
            if params_field is not None and isinstance(params_field, str):
                try:
                    parsed = json.loads(params_field)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in 'params' field: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("'params' must be a JSON object.")
                return parsed
            # Per-field fallback — collect all text fields verbatim
            # (no auto-coercion). Adapter ``.infer()`` is responsible
            # for ``float()``/``int()`` conversions on fields it
            # expects to be numeric.
            params: dict[str, Any] = {}
            for key, value in form.multi_items():
                if isinstance(value, str):
                    params[key] = value
            if not params:
                raise ValueError(
                    "Multipart body must include either a 'params' JSON field "
                    "or at least one text form field."
                )
            return params

        raise ValueError(
            f"Content-Type '{raw_ct or '(missing)'}' is not supported. "
            f"Send 'multipart/form-data' or 'application/json'."
        )

    def _transport_error(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        **details,
    ) -> JSONResponse:
        envelope = FailureEnvelope(
            error=ErrorDetail(
                category=ErrorCategory.TRANSPORT_ERROR,
                code=code,
                message=message,
                transient=False,
                details=details,
            )
        )
        self._metrics.record_infer("transport_error", 0.0)
        return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))

    # ── /infer/stream handling ─────────────────────────────────────

    async def _handle_stream(self, websocket: WebSocket) -> None:
        auth_header = (
            websocket.headers.get("authorization")
            or websocket.headers.get("Authorization")
        )
        failure = websocket_auth_failure(expected_token(), auth_header)
        if failure is not None:
            logger.info(
                "ws auth rejected adapter=%s code=%s remote=%s",
                self._name, failure,
                websocket.client.host if websocket.client else "?",
            )
            await websocket.close(
                code=StreamCloseCode.POLICY_REFUSED.value,
                reason=f"auth: {failure}",
            )
            return
        await self._service.handle_stream(websocket)


# ── Helpers (module-private) ───────────────────────────────────────


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
