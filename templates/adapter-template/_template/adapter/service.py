# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
__SERVICE_CLASS__ — your model wrapper goes here.

# TODO: replace this docstring with a one-paragraph design note.
# What does this adapter do? What model does it wrap? What
# assumptions about input shape, output shape, and per-camera state
# do callers need to know about?

The contract endpoints (/health, /capabilities, etc.) are wired by
the SDK's AdapterApp — you only fill in the four AdapterService
methods below plus ``infer()``.
"""
from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import AdapterService, ServiceError
from opennvr_adapter_sdk.contract import (
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)


# TODO: name your model framework — "onnxruntime", "torch", "tflite",
# "openvino", "custom", whatever. Surfaced via /capabilities.
MODEL_FRAMEWORK: str = "TODO"


class __SERVICE_CLASS__(AdapterService):
    """AdapterService implementation for the __SLUG__ adapter."""

    def __init__(self) -> None:
        # State markers — keep these; the SDK reads them via the four
        # AdapterService methods below to drive /health correctly.
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        # TODO: add any fields your model needs (weights handle,
        # tracker state, sessions, etc.).

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        """Eagerly load the model. Called once at startup before
        /health goes green.

        Best practice: import heavy ML libraries INSIDE this method
        (not at module top) so /capabilities discovery stays fast and
        a broken upstream dependency doesn't crash module import.
        """
        if self._load_state == HealthStatus.OK:
            return
        try:
            # TODO: load your model here.
            #   import your_ml_library
            #   self._model = your_ml_library.load(...)
            self._load_state = HealthStatus.OK
            self._load_error = None
            logger.info("__SERVICE_CLASS__ ready")
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = str(exc)
            logger.exception("__SERVICE_CLASS__ failed to load")

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        """sha256 of the model weights, or a synthetic version-derived
        fingerprint for adapters that don't have a weights file.

        Returning ``None`` is honest if your adapter has nothing
        meaningful to fingerprint, but KAI-C's drift detection skips
        ``None``-fingerprint adapters — prefer a deterministic value
        when you can.
        """
        # TODO: compute a real fingerprint. Example for an ONNX file:
        #   import hashlib
        #   digest = hashlib.sha256()
        #   with open(self._weights_path, "rb") as fh:
        #       for chunk in iter(lambda: fh.read(65536), b""):
        #           digest.update(chunk)
        #   return f"sha256:{digest.hexdigest()}"
        return None

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            # TODO: model identity. ``name`` should match what
            # /capabilities advertises so KAI-C's drift detection
            # joins the two cleanly.
            name="__SLUG__",
            version="1.0.0",
            framework=MODEL_FRAMEWORK,
            size_mb=None,  # TODO: report the weights file size if you have one
            # TODO: declare your input/output modalities.
            # Examples: ["image"] / ["bbox_classes"], ["audio"] / ["text"],
            # ["text"] / ["text"].
            modalities_in=["TODO"],
            modalities_out=["TODO"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        # TODO: report whether the host can actually run this adapter
        # well. GPU adapters check torch.cuda.is_available(); ONNX
        # adapters check onnxruntime.get_available_providers(); CPU-
        # only adapters can just return OK once load() succeeded.
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = "Model loaded successfully."
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Load failed: {self._load_error}"

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                # TODO: include adapter-specific diagnostics here. The
                # operator's /hardware/evaluation dashboard surfaces
                # this dict verbatim, so make it useful.
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
        )

    # ── §3.5 inference entry point ─────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run one inference call.

        For ``BodyShape.IMAGE`` / ``AUDIO`` / ``GENERIC`` adapters,
        the binary content lives at ``payload[BODY_BYTES_KEY]`` (import
        BODY_BYTES_KEY from the SDK). For ``BodyShape.TEXT`` adapters,
        the JSON body is merged directly into ``payload``.

        Errors must raise ``ServiceError`` with the right category;
        the SDK translates that into the standard failure envelope.
        Categories:
          * TRANSPORT_ERROR — malformed input, oversized body, etc. (4xx)
          * MODEL_ERROR — model can't process valid input (5xx)
          * PROVIDER_ERROR — upstream dep down (5xx)
          * PERMISSION_DENIED — operator policy refused (403)
          * OVERLOADED — backpressure (503)
        """
        if self._load_state != HealthStatus.OK:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code=(
                    "model_failed"
                    if self._load_state == HealthStatus.ERROR
                    else "model_loading"
                ),
                message=self._load_error or "Model still loading.",
                transient=(self._load_state == HealthStatus.LOADING),
                http_status=503,
                retry_after_ms=2000 if self._load_state == HealthStatus.LOADING else None,
            )

        start = time.monotonic()

        # TODO: parse payload, validate inputs, run your model, shape
        # the response. Below is a placeholder that returns the input
        # echoed back — replace it.
        try:
            # Example validation pattern:
            #   value = payload.get("required_field")
            #   if not isinstance(value, str) or not value:
            #       raise ServiceError(
            #           ErrorCategory.TRANSPORT_ERROR,
            #           code="malformed_input",
            #           message="required_field must be a non-empty string.",
            #           transient=False,
            #           http_status=400,
            #       )
            #
            # Placeholder: just report how many bytes/fields landed.
            # Echoing ``payload`` directly would crash on IMAGE/AUDIO
            # adapters because ``payload[BODY_BYTES_KEY]`` is raw
            # bytes and pydantic can't JSON-serialise those.
            result = {"payload_keys": sorted(payload.keys())}
        except ServiceError:
            raise
        except Exception as exc:
            logger.exception("__SERVICE_CLASS__ inference raised unexpectedly")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_runtime_crash",
                message="Inference failed.",
                transient=False,
                http_status=500,
            ) from exc

        inference_ms = int((time.monotonic() - start) * 1000)
        return InferResponse(
            model_name="__SLUG__",
            model_version="1.0.0",
            inference_ms=inference_ms,
            result=result,
        )
