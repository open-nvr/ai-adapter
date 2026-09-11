# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""`AdapterService` + `AdapterApp` — the classes the facade compiles to.

Demonstrates: `AdapterService`, `AdapterApp`, `BodyShape`,
`BODY_BYTES_KEY`, `ModelInfo`, `HardwareEvaluationResponse`,
`HealthStatus`, `InferResponse`, `Permissions`, `Scheduling`, `Cost`.

Write this directly when the facade stops fitting: a model with several
loading phases, an adapter that must answer /capabilities differently
per host, one that owns its own health semantics. The process, the
endpoints, the metrics and the published specs are identical — the
facade produces exactly this.
"""
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import (
    BODY_BYTES_KEY, AdapterApp, AdapterService, BodyShape, Cost, FairQueuing,
    HardwareEvaluationResponse, HardwareVerdict, HealthStatus, InferResponse,
    ModelInfo, Permissions, Scheduling, ServiceError,
)
from opennvr_adapter_sdk.contract import ErrorCategory


class PlateReader(AdapterService):
    """The four required methods, plus `infer`."""

    def __init__(self) -> None:
        self._state = HealthStatus.LOADING
        self._error: str | None = None
        self._model: Any = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        """Called once before /health goes green."""
        try:
            self._model = object()          # your real load
            self._state = HealthStatus.OK
        except Exception as exc:            # noqa: BLE001
            self._state = HealthStatus.ERROR
            self._error = str(exc)

    def is_ready(self) -> bool:
        return self._state == HealthStatus.OK

    # ── Identity ───────────────────────────────────────────────────

    def fingerprint(self) -> str | None:
        """A content hash of the weights.

        Returning None is honest for a cloud-fronting adapter, but KAI-C
        SKIPS drift detection for a null fingerprint — so a null one is
        silently exempt from the tamper check that protects the
        operator. Prefer a deterministic value."""
        return "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="plate-reader",
            version="2.1.0",
            framework="onnxruntime",
            size_mb=18.4,
            modalities_in=["image"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        """Rendered verbatim on the operator's hardware dashboard, so
        make the reasoning actionable."""
        ready = self._state == HealthStatus.OK
        return HardwareEvaluationResponse(
            verdict=HardwareVerdict.OK if ready else HardwareVerdict.BLOCKED,
            reasoning="Model loaded." if ready else f"Load failed: {self._error}",
            checked_at=datetime.now(timezone.utc),
            details={"providers": ["CPUExecutionProvider"]},
        )

    # ── Inference ──────────────────────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """For an IMAGE/AUDIO/GENERIC adapter the binary content is at
        ``payload[BODY_BYTES_KEY]``; for TEXT the JSON body is merged
        into ``payload`` directly."""
        if self._state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_loading",
                message=self._error or "Still loading.",
                transient=True, http_status=503, retry_after_ms=2000)

        frame = payload.get(BODY_BYTES_KEY)
        if not frame:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message="A frame is required.", transient=False, http_status=400)

        return InferResponse(model_name="plate-reader", model_version="2.1.0",
                             inference_ms=12, result={"plate_text": "MH12AB1234"})


# ── Wiring it up ───────────────────────────────────────────────────
#
# Everything the contract needs that is not the model: the six
# endpoints, auth, correlation ids, metrics, body parsing, the failure
# envelope, OpenAPI + AsyncAPI, and the lifespan.

_adapter_app = AdapterApp(
    # `service_factory=` instead of `service=` defers construction to
    # the lifespan, so an import never loads weights.
    service_factory=PlateReader,
    name="plate-reader",
    version="2.1.0",
    vendor="ACME Vision",
    license="Apache-2.0",
    model_card_url="https://example.com/plate-reader",
    tasks_advertised=["license_plate_recognition"],
    body_shape=BodyShape.IMAGE,
    max_body_bytes=8 * 1024 * 1024,
    permissions=Permissions(
        # KAI-C refuses to register an adapter asking for more than the
        # operator granted, so over-declaring blocks deployment.
        gpu=False, network_egress=[], host_filesystem=[],
        shared_memory_paths=[], host_metadata=False,
    ),
    scheduling=Scheduling(
        max_inflight=1, preferred_batch_size=1,
        # Round-robin frames across cameras under contention — almost
        # always what you want.
        fair_queuing=FairQueuing.PER_CAMERA,
    ),
    cost=Cost(currency="USD"),
    supports_stream=False,
)

app = _adapter_app.fastapi_app
