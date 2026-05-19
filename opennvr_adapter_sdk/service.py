# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
AdapterService ABC + ServiceError envelope.

This is the entire surface an adapter author has to implement. The
SDK plumbs it into the six mandatory contract endpoints; the author
writes only model-specific logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.interfaces.contract import (
    ErrorCategory,
    ErrorDetail,
    FailureEnvelope,
    HardwareEvaluationResponse,
    InferResponse,
    ModelInfo,
)


class AdapterService(ABC):
    """The interface every contract-compliant adapter implements.

    The methods are deliberately few:

    * ``load()`` — eagerly load the model. Called once at lifespan
      startup. ``is_ready()`` should return True afterwards.
    * ``is_ready()`` — used by /health to report ``loading`` vs ``ok``.
    * ``model_info()`` — describes the loaded model. Called on every
      /capabilities request so live-fingerprint drift detection per
      §11.3 works automatically.
    * ``hardware_evaluation()`` — returns the §3.3 verdict + details.
    * ``infer(payload)`` — runs one inference. Raises ``ServiceError``
      on failure; the SDK translates to a typed §7 envelope.

    Streaming is optional. Default is ``supports_stream = False`` and
    the SDK refuses the WebSocket upgrade with HTTP 501. Override
    ``handle_stream`` to implement the §6 protocol.
    """

    # ── Required ───────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> None:
        """Eagerly load the model. Idempotent — safe to call twice.

        Implementations should:
          1. Read weights from disk / download / connect to upstream.
          2. Compute and cache an initial fingerprint.
          3. Update internal state so ``is_ready()`` returns True.

        Exceptions are caught by the implementation and reflected via
        ``is_ready() == False`` plus an error message in
        ``hardware_evaluation()``; we never let load() exceptions
        kill the FastAPI app's startup.
        """

    @abstractmethod
    def is_ready(self) -> bool:
        """True iff the model is loaded and inference is possible."""

    @abstractmethod
    def fingerprint(self) -> str | None:
        """Live model fingerprint for §11.3 drift detection.

        Called on every /capabilities request — KAI-C polls every 60s
        and uses the returned value to detect tamper (file swap,
        weights rotation). Return None if the adapter can't compute
        one (cloud-fronted adapters, unsupported model formats);
        KAI-C will surface "model identity not verifiable" in the UI.
        """

    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Construct the §4 ``ModelInfo`` block for /capabilities.

        Implementations should call ``self.fingerprint()`` for the
        fingerprint field — DO NOT cache it; live recomputation is
        the whole point of §11.3 drift detection.
        """

    @abstractmethod
    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        """Construct the §3.3 ``HardwareEvaluationResponse``.

        The adapter decides verdict semantics — local hardware probe,
        cloud-endpoint ping, model-load status. The contract only
        standardizes the response shape.
        """

    @abstractmethod
    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run one inference. Return §3.5 ``InferResponse``.

        ``payload`` is a dict produced by the SDK's body parser:

        * For multipart requests, file-typed form fields are exposed
          as ``payload["__files__"][field_name] = bytes`` and the
          ``params`` JSON field is merged into the top-level dict.
        * For application/json requests, the request body is the
          payload directly.

        Raise ``ServiceError`` on every failure path so the SDK can
        translate to a typed §7 envelope with the correct HTTP
        status.
        """

    # ── Optional streaming ─────────────────────────────────────────
    #
    # The streaming-support flags (``supports_stream``,
    # ``stream_max_concurrent``, ``stream_supports_shared_memory``)
    # are passed to ``AdapterApp(...)`` directly, not declared here —
    # they're adapter-level metadata, not service-implementation
    # state, and they're needed at AdapterApp construction time
    # (before the service exists in the lazy-factory case) to wire
    # the right /infer/stream routes.

    async def handle_stream(self, websocket: Any) -> None:  # pragma: no cover
        """Override to implement the §6 WS protocol. The SDK calls
        this from the /infer/stream route AFTER the auth and lifespan
        readiness checks pass. Only called when the adapter declares
        ``supports_stream=True`` on its ``AdapterApp``."""
        raise NotImplementedError(
            "AdapterService.handle_stream() must be overridden when "
            "supports_stream=True is set on AdapterApp."
        )


# ── ServiceError ───────────────────────────────────────────────────


class ServiceError(Exception):
    """Carries enough information to construct a §7 ``FailureEnvelope``
    without re-parsing exception strings in the FastAPI route.

    Use prefix-namespaced codes (``<adapter>.<code>``) for adapter-
    specific failure modes that aren't in §7.1's canonical set.
    """

    def __init__(
        self,
        category: ErrorCategory,
        *,
        code: str,
        message: str,
        transient: bool,
        http_status: int,
        retry_after_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.code = code
        self.message = message
        self.transient = transient
        self.http_status = http_status
        self.retry_after_ms = retry_after_ms

    def envelope(self) -> FailureEnvelope:
        return FailureEnvelope(
            error=ErrorDetail(
                category=self.category,
                code=self.code,
                message=self.message,
                transient=self.transient,
                retry_after_ms=self.retry_after_ms,
                details={},
            )
        )
