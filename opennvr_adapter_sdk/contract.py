# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
AI Adapter Contract — v1 Pydantic skeletons.

This module codifies the wire shapes defined in
``open-nvr/docs/AI_ADAPTER_CONTRACT.md``. Every adapter that wants to
participate in the OpenNVR / KAI-C ecosystem MUST conform to these
shapes for the six mandatory endpoints (``/health``,
``/capabilities``, ``/hardware/evaluation``, ``/metrics``, ``/infer``,
``/infer/stream``) and the WebSocket streaming protocol.

Nothing in this module is wired up yet. It is the *contract* —
reviewable in isolation, importable by adapter authors as a typed
target, and the source of truth that ``BaseAdapter`` will be reworked
against in phase A2.

Design notes:

* The **envelope** shapes (``HealthResponse``, ``CapabilitiesResponse``,
  ``HardwareEvaluationResponse``, ``InferResponse``,
  ``FailureEnvelope``, and every WS message) use
  ``extra="forbid"`` — they are the wire contract and must not silently
  accept unknown fields.

* The **result conventions** in §5 of the design doc
  (``DetectionResult``, ``ClassificationResult``, ``AsrResult``,
  ``LlmChatResult``) are *guidance*, not strict types. Adapters return
  them as the ``result`` field inside ``InferResponse`` but are free to
  return any JSON-serializable shape. We keep them ``extra="allow"`` so
  adapters can add fields without breaking validation, and the envelope
  itself types ``result`` as ``dict[str, Any]``.

* Enums use ``str`` mixins so they serialize as JSON strings.

* All envelope models set ``extra="forbid"`` but DO NOT set
  ``strict=True``. Pydantic 2's strict mode rejects string-form enum
  inputs and ISO datetime strings when called via ``model_validate``
  (the dict path). ``model_validate_json`` (the actual wire path)
  relaxes strict mode automatically — but tests, mocks, and in-process
  code that construct payloads as Python dicts would fail. Coercion
  from strings is the friendly default for wire formats; ``extra``
  forbidden gives us the safety we actually need.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Contract version ─────────────────────────────────────────────────

# Bumped only on breaking changes to wire shapes. Adapters advertise
# which versions they speak via ``CapabilitiesResponse.adapter.supported_contract_versions``.
CONTRACT_VERSION: str = "1"


# ── Enums ────────────────────────────────────────────────────────────


class HealthStatus(str, Enum):
    """Liveness status reported by ``/health``."""

    OK = "ok"
    DEGRADED = "degraded"  # working but slow / partial
    LOADING = "loading"  # model loading, not ready yet
    ERROR = "error"  # alive but broken


class HardwareVerdict(str, Enum):
    """Verdict for ``/hardware/evaluation``."""

    OK = "ok"
    WARN = "warn"
    BLOCKED = "blocked"


class ErrorCategory(str, Enum):
    """Failure envelope category. Consumers use this to decide retry policy."""

    MODEL_ERROR = "model_error"  # inference failed: OOM, bad weights, NaN
    PROVIDER_ERROR = "provider_error"  # upstream cloud failure: 429, 503
    TRANSPORT_ERROR = "transport_error"  # network or framing
    PERMISSION_DENIED = "permission_denied"  # sandbox or policy refused
    NOT_SUPPORTED = "not_supported"  # endpoint not implemented
    OVERLOADED = "overloaded"  # backpressure


class FairQueuing(str, Enum):
    """How KAI-C should schedule traffic across cameras into this adapter."""

    NONE = "none"  # FIFO, useful for cloud adapters
    PER_CAMERA = "per_camera"  # token bucket per camera_id


class StreamMessageType(str, Enum):
    """WebSocket frame ``type`` discriminator."""

    HANDSHAKE = "handshake"
    HANDSHAKE_ACK = "handshake_ack"
    FRAME = "frame"
    FRAME_REF = "frame_ref"
    RESULT = "result"
    RESULT_ACK = "result_ack"
    PAUSE = "pause"
    RESUME = "resume"
    STATS = "stats"
    CLOSE = "close"


class FrameTransport(str, Enum):
    """How the client will deliver frames in the streaming protocol."""

    WEBSOCKET = "websocket"  # inline binary frames
    SHARED_MEMORY = "shared_memory"  # frame_ref pointing to /dev/shm


class StreamCloseCode(int, Enum):
    """WS close codes for adapter-initiated termination beyond 1000."""

    POLICY_REFUSED = 4001  # sovereignty / permissions / etc.
    MODEL_ERROR = 4002  # OOM, weights missing, runtime crash
    PROVIDER_ERROR = 4003  # cloud endpoint failure (proxy adapters)
    OVERLOADED = 4004  # back off and retry


# ── /health ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response shape for ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: HealthStatus
    adapter_name: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    started_at: datetime
    uptime_seconds: int = Field(ge=0)


# ── /capabilities ────────────────────────────────────────────────────


class AdapterInfo(BaseModel):
    """Adapter-identifying metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    license: str = Field(min_length=1)
    model_card_url: str | None = None
    supported_contract_versions: list[str] = Field(min_length=1)

    @field_validator("supported_contract_versions")
    @classmethod
    def _versions_nonempty(cls, value: list[str]) -> list[str]:
        if any(not v.strip() for v in value):
            raise ValueError("supported_contract_versions entries must be non-empty")
        return value


class ModelInfo(BaseModel):
    """Information about the underlying model the adapter wraps."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    framework: str = Field(min_length=1)  # "ultralytics", "transformers", "ollama", etc.
    size_mb: float | None = Field(default=None, ge=0.0)
    modalities_in: list[str] = Field(default_factory=list)  # "image", "audio", "text"
    modalities_out: list[str] = Field(default_factory=list)  # "bbox_classes", "text", etc.
    # Opaque adapter-chosen string, typically a content hash of the
    # weights (e.g. "sha256:..."). KAI-C records it at registration
    # and on every /capabilities poll; a mismatch is a tamper signal
    # and triggers an audit event (§11.2). Optional — adapters that
    # can't compute a meaningful fingerprint (cloud-fronting, hosted)
    # omit the field; KAI-C surfaces "model identity not verifiable."
    fingerprint: str | None = Field(default=None, min_length=1)


class InferEndpointInfo(BaseModel):
    """Capability description for ``POST /infer``."""

    model_config = ConfigDict(extra="forbid")

    supported: bool
    input_content_types: list[str] = Field(default_factory=list)
    input_schema_ref: str | None = None
    output_schema_ref: str | None = None


class StreamEndpointInfo(BaseModel):
    """Capability description for ``POST /infer/stream`` (WebSocket)."""

    model_config = ConfigDict(extra="forbid")

    supported: bool
    max_concurrent_streams: int = Field(default=0, ge=0)
    supports_shared_memory: bool = False
    shared_memory_protocol_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _shm_protocol_implies_support(self) -> "StreamEndpointInfo":
        if self.supports_shared_memory and self.shared_memory_protocol_version is None:
            raise ValueError(
                "shared_memory_protocol_version is required when supports_shared_memory is true"
            )
        return self


class ExtraEndpoint(BaseModel):
    """Adapter-specific endpoint outside the mandatory six."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"]
    purpose: str = Field(min_length=1)


class EndpointsInfo(BaseModel):
    """Aggregated endpoints section of /capabilities."""

    model_config = ConfigDict(extra="forbid")

    infer: InferEndpointInfo
    infer_stream: StreamEndpointInfo
    extra: list[ExtraEndpoint] = Field(default_factory=list)


class Permissions(BaseModel):
    """
    Sandboxing declaration. KAI-C reads this on adapter registration
    and applies the declared scope as container constraints. See §8 of
    the contract.
    """

    model_config = ConfigDict(extra="forbid")

    gpu: bool = False
    network_egress: list[str] = Field(default_factory=list)
    host_filesystem: list[str] = Field(default_factory=list)
    shared_memory_paths: list[str] = Field(default_factory=list)
    host_metadata: bool = False  # block IMDS / /proc/host* by default


class Scheduling(BaseModel):
    """How KAI-C should schedule work into this adapter."""

    model_config = ConfigDict(extra="forbid")

    # Defaults to 1 so a first-time adapter author can omit this whole
    # block and ship — safe, sequential, advertise more once measured.
    max_inflight: int = Field(default=1, ge=1)
    preferred_batch_size: int = Field(default=1, ge=1)
    fair_queuing: FairQueuing = FairQueuing.NONE


class Cost(BaseModel):
    """
    Cost declaration. Lets OpenNVR show a running estimate of spend
    and refuse to schedule once a budget is exhausted. For free/local
    adapters everything is zero.
    """

    model_config = ConfigDict(extra="forbid")

    currency: str = Field(default="USD", min_length=3, max_length=3)
    estimated_per_call: float = Field(default=0.0, ge=0.0)
    estimated_per_hour: float = Field(default=0.0, ge=0.0)
    rate_limit_per_minute: int | None = Field(default=None, ge=0)
    is_metered: bool = False


class CapabilitiesResponse(BaseModel):
    """Top-level response shape for ``GET /capabilities``."""

    model_config = ConfigDict(extra="forbid")

    adapter: AdapterInfo
    model: ModelInfo
    endpoints: EndpointsInfo
    # ``tasks_advertised`` is intentionally a free-text vocabulary. See
    # design doc §15.1 — we may canonicalize it later if the community
    # converges on names.
    tasks_advertised: list[str] = Field(default_factory=list)
    permissions: Permissions = Field(default_factory=Permissions)
    scheduling: Scheduling
    cost: Cost = Field(default_factory=Cost)


# ── /hardware/evaluation ─────────────────────────────────────────────


class HardwareEvaluationResponse(BaseModel):
    """
    Verdict + reasoning for "can this adapter serve from where it is
    deployed". Adapter decides how to compute the verdict — local
    hardware probe, cloud ping, service-mesh health, model load
    status. Only the response shape is standardized.
    """

    model_config = ConfigDict(extra="forbid")

    verdict: HardwareVerdict
    reasoning: str = Field(min_length=1)
    checked_at: datetime
    # ``details`` is free-form per adapter — a local adapter puts
    # GPU/VRAM info, a cloud adapter puts endpoint_reachable /
    # measured_latency_ms / rate_limit_headroom_pct.
    details: dict[str, Any] = Field(default_factory=dict)


# ── Failure envelope ─────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """The ``error`` block inside the failure envelope."""

    model_config = ConfigDict(extra="forbid")

    category: ErrorCategory
    code: str = Field(min_length=1)  # adapter-defined stable identifier
    message: str = Field(min_length=1)
    transient: bool
    retry_after_ms: int | None = Field(default=None, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _retry_after_matches_transient(self) -> "ErrorDetail":
        if self.retry_after_ms is not None and not self.transient:
            raise ValueError(
                "retry_after_ms is only meaningful for transient errors"
            )
        return self


class FailureEnvelope(BaseModel):
    """Standard error response used by every endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = "error"
    error: ErrorDetail


# ── /infer ───────────────────────────────────────────────────────────


class InferResponse(BaseModel):
    """Successful response shape for ``POST /infer``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    inference_ms: int = Field(ge=0)
    # ``result`` is free-form per §5 — adapter MAY use one of the
    # conventions below, MAY return its own shape. Wire-level type is
    # just JSON-serializable dict.
    result: dict[str, Any]


# ── Inference output conventions (§5) — guidance only ────────────────


class NormalizedBBox(BaseModel):
    """Resolution-independent bounding box, all values in [0, 1]."""

    model_config = ConfigDict(extra="allow")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _bbox_fits_in_frame(self) -> "NormalizedBBox":
        if self.x + self.w > 1.0 + 1e-6:
            raise ValueError("bbox x + w must be <= 1.0")
        if self.y + self.h > 1.0 + 1e-6:
            raise ValueError("bbox y + h must be <= 1.0")
        return self


class FrameDimensions(BaseModel):
    """Source frame dimensions in pixels."""

    model_config = ConfigDict(extra="allow")

    w: int = Field(gt=0)
    h: int = Field(gt=0)


class DetectionItem(BaseModel):
    """A single detection in the §5.1 convention."""

    model_config = ConfigDict(extra="allow")

    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: NormalizedBBox
    track_id: str | int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    """§5.1 — Detection (bounding boxes) convention."""

    model_config = ConfigDict(extra="allow")

    detections: list[DetectionItem] = Field(default_factory=list)
    frame_dimensions: FrameDimensions | None = None


class ClassificationItem(BaseModel):
    """A single label + confidence pair."""

    model_config = ConfigDict(extra="allow")

    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    """§5.2 — Classification convention."""

    model_config = ConfigDict(extra="allow")

    predictions: list[ClassificationItem] = Field(default_factory=list)


class AsrSegment(BaseModel):
    """A single transcribed segment with start/end timestamps."""

    model_config = ConfigDict(extra="allow")

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _end_after_start(self) -> "AsrSegment":
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class AsrResult(BaseModel):
    """§5.3 — ASR convention."""

    model_config = ConfigDict(extra="allow")

    transcript: str
    language: str = Field(min_length=1)
    segments: list[AsrSegment] = Field(default_factory=list)


class LlmUsage(BaseModel):
    """Token usage block for LLM responses."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LlmChatResult(BaseModel):
    """§5.4 — LLM chat convention."""

    model_config = ConfigDict(extra="allow")

    completion: str
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "unknown"] = "stop"
    usage: LlmUsage | None = None


# ── /infer/stream — WebSocket protocol (§6) ──────────────────────────


class HandshakeMessage(BaseModel):
    """
    First message client sends after opening the WebSocket. Negotiates
    transport, result sink, and expected input rate.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.HANDSHAKE] = StreamMessageType.HANDSHAKE
    client_id: str = Field(min_length=1)
    # Optional at the model level but REQUIRED at runtime when the
    # target adapter advertises ``scheduling.fair_queuing="per_camera"``.
    # The model can't enforce this — KAI-C does at the server side
    # by inspecting the target adapter's /capabilities and refusing
    # the handshake with a CloseMessage(close_code=POLICY_REFUSED).
    camera_id: str | None = None
    frame_transport: FrameTransport = FrameTransport.WEBSOCKET
    shared_memory_root: str | None = None
    # ``result_sink`` is "websocket" or a NATS subject like
    # "nats:detections.cam-7.object_detection". The bus itself ships in
    # B1; until then adapters MAY refuse non-websocket sinks.
    result_sink: str = "websocket"
    expected_input_rate_hz: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _shared_memory_requires_root(self) -> "HandshakeMessage":
        if self.frame_transport == FrameTransport.SHARED_MEMORY and not self.shared_memory_root:
            raise ValueError(
                "shared_memory_root is required when frame_transport is shared_memory"
            )
        return self


class HandshakeAckMessage(BaseModel):
    """Adapter's reply to the handshake, confirming negotiated terms."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.HANDSHAKE_ACK] = StreamMessageType.HANDSHAKE_ACK
    frame_transport: FrameTransport  # may differ from offer if adapter can't honor it
    result_sink: str = "websocket"
    max_inflight: int = Field(ge=1)
    session_id: str = Field(min_length=1)


class FrameMessage(BaseModel):
    """
    Inline-frame metadata. The next WS message after this one is the
    raw binary frame payload (handled outside JSON parsing).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.FRAME] = StreamMessageType.FRAME
    seq: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    content_type: str = Field(min_length=1)  # e.g. "image/jpeg", "audio/wav"


class FrameRefMessage(BaseModel):
    """
    Shared-memory frame reference. Client wrote the frame to
    ``shm_path``; adapter reads it. Client is responsible for the
    lifecycle of the shm file (unlink or ring-buffer wrap).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.FRAME_REF] = StreamMessageType.FRAME_REF
    seq: int = Field(ge=0)
    ts_ms: int = Field(ge=0)
    shm_path: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)


class ResultMessage(BaseModel):
    """Adapter → client: a completed inference result."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.RESULT] = StreamMessageType.RESULT
    seq: int = Field(ge=0)  # echoes the frame seq
    ts_ms: int = Field(ge=0)
    inference_ms: int = Field(ge=0)
    result: dict[str, Any]  # free-form per §5


class ResultAckMessage(BaseModel):
    """
    Heartbeat from adapter when ``result_sink`` is NATS rather than
    websocket. Lets the client know inference is still happening
    without echoing every payload back over the socket.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.RESULT_ACK] = StreamMessageType.RESULT_ACK
    last_seq: int = Field(ge=0)
    published_count: int = Field(ge=0)
    ts_ms: int = Field(ge=0)


class PauseMessage(BaseModel):
    """Either side: stop sending until resume."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.PAUSE] = StreamMessageType.PAUSE


class ResumeMessage(BaseModel):
    """Either side: resume flow after a pause."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.RESUME] = StreamMessageType.RESUME


class StatsMessage(BaseModel):
    """
    Either request stats (no fields beyond ``type``) or report them.
    Fields are populated in responses, omitted in requests.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.STATS] = StreamMessageType.STATS
    inflight: int | None = Field(default=None, ge=0)
    queue_depth: int | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0.0)


class CloseMessage(BaseModel):
    """Either side: graceful shutdown with a human-readable reason."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[StreamMessageType.CLOSE] = StreamMessageType.CLOSE
    reason: str = Field(min_length=1)
    close_code: StreamCloseCode | None = None  # adapter sets this; client typically does not


# ── Union types for convenient typing in routers/clients ────────────


# Client → Adapter messages
ClientToAdapterMessage = (
    HandshakeMessage
    | FrameMessage
    | FrameRefMessage
    | PauseMessage
    | ResumeMessage
    | StatsMessage
    | CloseMessage
)

# Adapter → Client messages
AdapterToClientMessage = (
    HandshakeAckMessage
    | ResultMessage
    | ResultAckMessage
    | PauseMessage
    | ResumeMessage
    | StatsMessage
    | CloseMessage
)


__all__ = [
    # version
    "CONTRACT_VERSION",
    # enums
    "HealthStatus",
    "HardwareVerdict",
    "ErrorCategory",
    "FairQueuing",
    "StreamMessageType",
    "FrameTransport",
    "StreamCloseCode",
    # health
    "HealthResponse",
    # capabilities
    "AdapterInfo",
    "ModelInfo",
    "InferEndpointInfo",
    "StreamEndpointInfo",
    "ExtraEndpoint",
    "EndpointsInfo",
    "Permissions",
    "Scheduling",
    "Cost",
    "CapabilitiesResponse",
    # hardware
    "HardwareEvaluationResponse",
    # failure envelope
    "ErrorDetail",
    "FailureEnvelope",
    # infer
    "InferResponse",
    # §5 conventions
    "NormalizedBBox",
    "FrameDimensions",
    "DetectionItem",
    "DetectionResult",
    "ClassificationItem",
    "ClassificationResult",
    "AsrSegment",
    "AsrResult",
    "LlmUsage",
    "LlmChatResult",
    # streaming
    "HandshakeMessage",
    "HandshakeAckMessage",
    "FrameMessage",
    "FrameRefMessage",
    "ResultMessage",
    "ResultAckMessage",
    "PauseMessage",
    "ResumeMessage",
    "StatsMessage",
    "CloseMessage",
    # unions
    "ClientToAdapterMessage",
    "AdapterToClientMessage",
]
