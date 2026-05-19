# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
PiperService — the contract-semantics layer around ``PiperAdapter``.

Translates between the AI Adapter Contract v1 wire shapes and the
legacy ``PiperAdapter`` infer interface. Stateful (holds the loaded
adapter, the start time, the fingerprint cache).

This layer does NOT do any TTS work itself — it delegates to
``PiperAdapter.infer`` and reshapes the result.

Coexistence with the legacy monolith
------------------------------------
The legacy ``app/main.py`` (which bundles all 8 adapters into one
FastAPI service) is *intentionally untouched* by A2.1. Operators who
were running the monolith keep running it; the new per-adapter
services are an opt-in surface for KAI-C to register against. This
incremental migration path is documented in §13 of the design doc
(``open-nvr/docs/AI_ADAPTER_CONTRACT.md``). A2.2 through A2.4 migrate
the remaining adapters one at a time; once each has its own
contract-compliant service, the monolith can be retired.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from app.adapters.audio.piper_adapter import PiperAdapter
from app.config import MODEL_WEIGHTS_DIR
from app.interfaces.contract import (
    AdapterInfo,
    CapabilitiesResponse,
    Cost,
    EndpointsInfo,
    ErrorCategory,
    ErrorDetail,
    ExtraEndpoint,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthResponse,
    HealthStatus,
    InferEndpointInfo,
    InferResponse,
    ModelInfo,
    Permissions,
    Scheduling,
    StreamEndpointInfo,
)

logger = logging.getLogger(__name__)

# Service-level constants — surfaced in /capabilities and /health so
# operators see what they're running.
ADAPTER_NAME: str = "piper-tts"
ADAPTER_VERSION: str = "1.0.0"
ADAPTER_VENDOR: str = "open-nvr"
ADAPTER_LICENSE: str = "AGPL-3.0"
MODEL_FRAMEWORK: str = "piper-tts"
TASKS_ADVERTISED: tuple[str, ...] = ("speech_synthesis",)

# Default text length cap mirrors the underlying PiperAdapter guard. We
# advertise this via /capabilities so callers can avoid pathological
# inputs without first hitting a 400.
MAX_TEXT_CHARS: int = 10_000


class PiperService:
    """Stateful façade around ``PiperAdapter`` implementing contract semantics."""

    def __init__(self, default_voice: str = "en_US-libritts-high", voice_dir: str | None = None) -> None:
        self._default_voice = default_voice
        self._voice_dir = voice_dir or os.path.join(MODEL_WEIGHTS_DIR, "piper")
        self._started_at_dt: datetime = datetime.now(timezone.utc)
        self._started_at_mono: float = time.monotonic()

        self._adapter: PiperAdapter = PiperAdapter(
            config={
                "enabled": True,
                "voice": default_voice,
                "voice_dir": self._voice_dir,
            }
        )

        # Track load state separately so /health can report loading vs.
        # error vs. ok without poking the underlying adapter.
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint: str | None = None
        self._load_lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the default voice. Idempotent."""
        with self._load_lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                self._adapter.ensure_model_loaded()
                self._fingerprint = self._compute_fingerprint(self._default_voice)
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "PiperService ready: voice=%s fingerprint=%s",
                    self._default_voice,
                    self._fingerprint,
                )
            except Exception as exc:  # pragma: no cover — covered in tests
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("PiperService failed to load voice %s", self._default_voice)

    def _compute_fingerprint(self, voice_name: str) -> str:
        """sha256 of the voice ONNX file. Treated as opaque by KAI-C."""
        onnx_path = os.path.join(self._voice_dir, f"{voice_name}.onnx")
        if not os.path.exists(onnx_path):
            # Should never happen — load() would have raised — but be
            # defensive so /capabilities never crashes after a successful
            # load that later loses the file.
            return "sha256:unavailable"
        digest = hashlib.sha256()
        with open(onnx_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    # ── Contract endpoints ─────────────────────────────────────────

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=self._load_state,
            adapter_name=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            model_name=self._default_voice,
            model_version=self._adapter_model_version(),
            started_at=self._started_at_dt,
            uptime_seconds=int(time.monotonic() - self._started_at_mono),
        )

    def capabilities(self) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            adapter=AdapterInfo(
                name=ADAPTER_NAME,
                version=ADAPTER_VERSION,
                vendor=ADAPTER_VENDOR,
                license=ADAPTER_LICENSE,
                model_card_url="https://github.com/rhasspy/piper",
                supported_contract_versions=["1"],
            ),
            model=ModelInfo(
                name=self._default_voice,
                version=self._adapter_model_version(),
                framework=MODEL_FRAMEWORK,
                modalities_in=["text"],
                modalities_out=["audio"],
                fingerprint=self._fingerprint,
            ),
            endpoints=EndpointsInfo(
                infer=InferEndpointInfo(
                    supported=True,
                    # §3.5 mandates multipart support; we list it first
                    # to signal the preferred path. JSON is also accepted
                    # for clients that can't easily build multipart.
                    input_content_types=["multipart/form-data", "application/json"],
                ),
                # §3.6 — TTS is one-shot, not real-time. Refuse the WS
                # upgrade with HTTP 501. supported=false here so KAI-C
                # knows up front and never tries.
                infer_stream=StreamEndpointInfo(
                    supported=False,
                    max_concurrent_streams=0,
                ),
                extra=[
                    ExtraEndpoint(
                        path="/voices",
                        method="GET",
                        purpose="List installed voices and their fingerprints",
                    ),
                ],
            ),
            tasks_advertised=list(TASKS_ADVERTISED),
            permissions=Permissions(
                gpu=False,
                # No outbound calls — Piper is fully local.
                network_egress=[],
                # Read voices from the configured voice dir; write WAVs
                # to BASE_AUDIO_DIR/tts/. KAI-C will bind-mount these.
                host_filesystem=[self._voice_dir],
                shared_memory_paths=[],
                host_metadata=False,
            ),
            scheduling=Scheduling(
                max_inflight=4,
                preferred_batch_size=1,
            ),
            cost=Cost(
                currency="USD",
                estimated_per_call=0.0,
                estimated_per_hour=0.0,
                is_metered=False,
            ),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        # Piper runs on CPU. The only thing we really need to confirm
        # is that the voice file is loadable, which load() already did.
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = "Piper runs on CPU; default voice loaded."
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Voice still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Voice failed to load: {self._load_error}"

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "default_voice": self._default_voice,
                "voice_dir": self._voice_dir,
            },
        )

    def list_voices(self) -> dict[str, Any]:
        """Extra endpoint advertised in /capabilities."""
        voices = []
        if os.path.isdir(self._voice_dir):
            for entry in sorted(os.listdir(self._voice_dir)):
                if entry.endswith(".onnx"):
                    voices.append(
                        {
                            "name": entry[: -len(".onnx")],
                            "size_bytes": os.path.getsize(
                                os.path.join(self._voice_dir, entry)
                            ),
                        }
                    )
        return {"voices": voices, "default": self._default_voice}

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """
        Run TTS inference.

        Accepts the JSON body from POST /infer (already validated as a
        non-empty dict by the FastAPI route). Translates contract input
        → PiperAdapter input → contract output.

        Raises ``ServiceError`` (a typed wrapper around FailureEnvelope)
        on every failure path so the route can return the correct
        category + code without re-parsing exception strings.
        """
        if self._load_state != HealthStatus.OK:
            # ``weights_missing`` is canonical (§7.1). ``piper.model_loading``
            # is adapter-specific — §7.1 says adapter codes outside the
            # canonical set MUST be prefix-namespaced. Transient=true on
            # LOADING tells consumers to retry; ERROR is operator-actionable.
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if self._load_state == HealthStatus.ERROR else "piper.model_loading",
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        # We accept either flat shape ({"text": ...}) or the
        # PiperAdapter-native shape ({"task": ..., "text": ...}). Both
        # are valid input to the underlying adapter.
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message="Field 'text' is required and must be a non-empty string.",
                transient=False,
                http_status=400,
            )
        if len(text) > MAX_TEXT_CHARS:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"'text' exceeds {MAX_TEXT_CHARS}-char limit.",
                transient=False,
                http_status=400,
            )

        adapter_input: dict[str, Any] = {"task": "speech_synthesis", "text": text}
        for key in ("voice", "length_scale", "noise_scale", "noise_w"):
            if key in payload:
                adapter_input[key] = payload[key]

        try:
            raw = self._adapter.infer(adapter_input)
        except (FileNotFoundError, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="weights_missing" if isinstance(exc, FileNotFoundError) else "malformed_input",
                message=str(exc),
                transient=False,
                http_status=400 if isinstance(exc, ValueError) else 500,
            ) from exc
        except Exception as exc:
            # Any other exception is a model_error per §7 — we do not
            # leak internals to the caller.
            logger.exception("PiperAdapter.infer raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ) from exc

        if isinstance(raw, dict):
            result = raw
        else:
            # Pydantic BaseModel — convert to dict
            result = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)

        return InferResponse(
            model_name=self._default_voice,
            model_version=self._adapter_model_version(),
            inference_ms=int(result.get("latency_ms", 0)),
            result=result,
        )

    def is_ready(self) -> bool:
        """True iff the default voice loaded successfully. Used by the
        lifespan hook to set the ``adapter_model_loaded`` gauge."""
        return self._load_state == HealthStatus.OK

    # ── Helpers ────────────────────────────────────────────────────

    def _adapter_model_version(self) -> str:
        """
        Best-effort model_version. Piper itself doesn't expose one per
        voice — the voice name is the only stable identifier — so we
        return the framework + voice name so callers see something
        meaningful in audit logs.
        """
        return f"{MODEL_FRAMEWORK}/{self._default_voice}"


# ── Typed error envelope ───────────────────────────────────────────


class ServiceError(Exception):
    """
    Carries enough information to construct a FailureEnvelope without
    re-parsing exception strings in the FastAPI route.
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
