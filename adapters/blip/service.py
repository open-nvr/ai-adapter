# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
BLIP scene-caption adapter — contract-compliant ``AdapterService``.

SDK-based replacement for the legacy in-tree adapter at
``app/adapters/llm/blip_adapter.py``. The legacy code only accepted
``opennvr://frames/<camera>/<file>`` URIs that required a shared
volume between the adapter and the upstream. This service accepts
image bytes directly via the contract's ``BodyShape.IMAGE`` body
parser, so any HTTP caller can invoke it without a filesystem
mount.

Single task: ``scene_caption`` — return a one-sentence natural-
language description of the image. Used by the ``camera-agent``
example's ``describe_camera`` tool.

CPU-friendly on a 4-core homelab: BLIP-base is ~990 MB on download,
runs at roughly 2-4s/image on CPU (much faster with CUDA passthrough).
First inference also triggers the HuggingFace download — operators
should pre-bake the model into the image or mount a populated
HF_HOME cache to avoid first-request latency spikes.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import (
    AdapterService,
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    ServiceError,
)

logger = logging.getLogger(__name__)


# Default HuggingFace model id. ``blip-image-captioning-base`` is the
# canonical scene-caption model — ~990 MB, well-supported. Operators
# can swap to ``-large`` (~2 GB, more accurate) via the
# ``OPENNVR_BLIP_MODEL`` env var.
DEFAULT_MODEL_ID: str = "Salesforce/blip-image-captioning-base"

# Max body bytes per call. A 4K JPEG is well under 4 MB; the 8 MB cap
# matches the YOLOv8 / InsightFace adapters. Above this the SDK
# returns 413 before the service is ever invoked.
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024

# Generation cap — keeps responses short and prevents pathological
# memory growth on degenerate prompts.
DEFAULT_MAX_NEW_TOKENS: int = 50

_SUPPORTED_TASKS: tuple[str, ...] = ("scene_caption",)


class BlipService(AdapterService):
    """SDK-based scene-captioning service backed by HuggingFace BLIP."""

    def __init__(
        self,
        model_id: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model_id: str = model_id or os.getenv(
            "OPENNVR_BLIP_MODEL", DEFAULT_MODEL_ID
        )
        self._max_new_tokens = int(max_new_tokens)
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at: datetime = datetime.now(timezone.utc)

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                # Lazy imports — keeps adapter discovery fast and
                # avoids forcing transformers/torch into discovery-
                # only code paths.
                import torch
                from transformers import (
                    BlipForConditionalGeneration,
                    BlipProcessor,
                )

                logger.info("BlipService: loading %s", self._model_id)
                self._processor = BlipProcessor.from_pretrained(self._model_id)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    self._model_id
                )

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)
                self._model.eval()

                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info(
                    "BlipService ready: model=%s device=%s fingerprint=%s",
                    self._model_id, self._device, self._fingerprint_cache,
                )
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception(
                    "BlipService failed to load model %s", self._model_id,
                )

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        # Compute on every call so KAI-C's /capabilities poll picks
        # up a model swap. Fall back to the cached value if the
        # HuggingFace cache directory isn't readable.
        try:
            return self._compute_fingerprint()
        except OSError:
            return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_id,
            version=f"blip/{self._model_id}",
            framework="transformers",
            modalities_in=["image"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (
                f"BLIP loaded on device {self._device!r}. CPU inference "
                f"works; GPU is faster if available."
            )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Model failed to load: {self._load_error}"

        uptime_s = int(
            (datetime.now(timezone.utc) - self._started_at).total_seconds()
        )
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "device": self._device or "not_loaded",
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "model_id": self._model_id,
                "max_new_tokens": self._max_new_tokens,
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """Run scene captioning on a single image.

        ``payload['__file__']`` holds the raw image bytes (multipart
        upload or base64-decoded JSON). ``task`` defaults to
        ``scene_caption``; ``max_new_tokens`` may override the
        per-instance default.
        """
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="model_not_loaded",
                message=self._load_error or "BLIP model not yet loaded",
                transient=transient,
                http_status=503,
                retry_after_ms=2000 if transient else None,
            )

        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="missing_image",
                message=(
                    "blip expects an image in the request body "
                    "(multipart 'frame' field or JSON 'frame_b64'). "
                    "No image bytes were provided."
                ),
                transient=False,
                http_status=400,
            )

        task = (payload.get("task") or "scene_caption").strip()
        if task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED,
                code="unsupported_task",
                message=(
                    f"task {task!r} is not supported; "
                    f"choose one of {list(_SUPPORTED_TASKS)}"
                ),
                transient=False,
                http_status=400,
            )

        max_new_tokens = self._resolve_max_new_tokens(payload)

        try:
            image = _decode_image(image_bytes)
        except _ImageDecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_image",
                message=str(exc),
                transient=False,
                http_status=400,
            ) from exc

        started = time.monotonic()
        try:
            caption = self._run_blip(image, max_new_tokens=max_new_tokens)
        except Exception as exc:
            logger.exception("BLIP inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="inference_failed",
                message=f"BLIP raised during inference: {exc}",
                transient=True,
                http_status=500,
                retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        return InferResponse(
            model_name=self._model_id,
            model_version=f"blip/{self._model_id}",
            inference_ms=elapsed_ms,
            result={
                "task": "scene_caption",
                "caption": caption,
                "model_id": self._model_id,
                "device": self._device,
            },
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _resolve_max_new_tokens(self, payload: dict[str, Any]) -> int:
        raw = payload.get("max_new_tokens")
        if raw is None:
            return self._max_new_tokens
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="invalid_max_new_tokens",
                message=f"max_new_tokens must be an integer; got {raw!r}",
                transient=False,
                http_status=400,
            )
        if not 1 <= value <= 256:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="max_new_tokens_out_of_range",
                message=f"max_new_tokens must be in [1, 256]; got {value}",
                transient=False,
                http_status=400,
            )
        return value

    def _run_blip(self, image: Any, *, max_new_tokens: int) -> str:
        """Run the BLIP processor + generation. Pulled into a helper
        so tests can monkeypatch a stub without spinning up real
        transformers."""
        import torch

        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )
        caption = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip()
        return caption

    def _compute_fingerprint(self) -> str:
        """Sha256 over the model identifier — BLIP weights live inside
        HuggingFace's per-user cache, walking that to hash the full
        weight files is slow + brittle. The id-derived fingerprint is
        good enough for KAI-C drift detection (operators rotate by
        changing OPENNVR_BLIP_MODEL).

        A future enhancement: hash the underlying ``.safetensors``
        files if the HF cache layout is stable enough to discover.
        """
        digest = hashlib.sha256(self._model_id.encode("utf-8")).hexdigest()
        return f"sha256-id:{digest}"


# ── Inference helpers ──────────────────────────────────────────────


class _ImageDecodeError(Exception):
    """Raised when the input bytes don't decode as an image."""


def _decode_image(image_bytes: bytes) -> Any:
    """Decode JPEG / PNG / WebP / BMP bytes into a Pillow ``Image``
    in RGB mode (BLIP expects RGB)."""
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Force-load so a corrupted-but-recognised-header doesn't
        # surface a lazy IOError downstream inside transformers.
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise _ImageDecodeError(
            "could not decode image bytes (expected JPEG / PNG / "
            "WebP / BMP)"
        ) from exc
    return image.convert("RGB")
