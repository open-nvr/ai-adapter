# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Moondream2 vision-language adapter — contract-compliant ``AdapterService``.

Unlike the BLIP captioner (which only *describes* a scene), this is a small
**vision-language model** that does real **visual question answering**: it
answers "what is the person wearing?", "what is he doing?", "is the gate open?"
grounded in the frame — what conversations *about* video need (open-nvr
camera-agent test-report S-6).

**Runtime:** the official ``moondream`` package with a **quantized model file**
(int8 ``.mf.gz``), NOT transformers + trust_remote_code. The package is built
for fast VQA on CPU/edge, so it sidesteps the transformers remote-code fragility
and the documented CPU image-encoding slowness — the right fit for limited
hardware. Model files: 0.5B int8 (~593 MiB, the edge default) or 2B int8
(~1.7 GiB, more capable). Apache-2.0.

Two tasks:
  * ``visual_qa``      — answer ``question`` about the image (key ``answer``).
  * ``scene_caption``  — one-line caption (key ``caption``) — the SAME key the
                         camera-agent's describe_camera consumes, so it's a
                         drop-in for the BLIP ``caption_adapter`` slot.

The model file is baked into the image (see Dockerfile) so it runs offline under
AI_SOVEREIGNTY=local_only with no runtime egress.
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

# Path to the quantized model file (.mf / .mf.gz), baked into the image.
DEFAULT_MODEL_PATH: str = os.getenv(
    "OPENNVR_MOONDREAM_MODEL_PATH", "/models/moondream-0_5b-int8.mf.gz"
)
MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
_SUPPORTED_TASKS: tuple[str, ...] = ("visual_qa", "scene_caption")


class MoondreamService(AdapterService):
    """SDK-based VQA + captioning service backed by the moondream package."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._model: Any | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

    # ── AdapterService impl ────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                import moondream as md  # the official, CPU-optimised runtime

                self._ensure_model()
                logger.info("MoondreamService: loading %s", self._model_path)
                self._model = md.vl(model=self._model_path)
                self._fingerprint_cache = self._compute_fingerprint()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("MoondreamService ready: %s", self._model_path)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("MoondreamService failed to load %s", self._model_path)

    def _ensure_model(self) -> None:
        """Provision the model file if it isn't present. Three ways to supply it,
        in order of sovereignty:
          1. BAKED into the image at build (offline / local_only — best).
          2. MOUNTED into the model volume (offline).
          3. DOWNLOADED once at startup from OPENNVR_MOONDREAM_MODEL_URL — makes
             the code-only CI image pull-and-run, but it's a one-time network
             fetch (operator opt-in; NOT for a strict local_only posture).
        Cached at the model path, so subsequent starts skip it."""
        if os.path.exists(self._model_path):
            return
        url = (os.getenv("OPENNVR_MOONDREAM_MODEL_URL")
               or os.getenv("MOONDREAM_MODEL_URL") or "").strip()
        if not url:
            return  # no source — md.vl() will fail with a clear "file not found"
        import urllib.request
        d = os.path.dirname(self._model_path)
        if d:
            os.makedirs(d, exist_ok=True)
        logger.info("Moondream: model missing; downloading once from %s -> %s",
                    url, self._model_path)
        tmp = self._model_path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, self._model_path)
        logger.info("Moondream: model downloaded.")

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._model_path),
            version=f"moondream/{os.path.basename(self._model_path)}",
            framework="moondream",
            modalities_in=["image", "text"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict, reasoning = HardwareVerdict.OK, (
                "Moondream (quantized) loaded. Optimised for CPU; "
                "0.5B int8 is the fast edge default, 2B int8 more capable.")
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Load failed: {self._load_error}"
        uptime_s = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "runtime": "moondream-quantized",
                "model_path": self._model_path,
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "adapter_uptime_seconds": uptime_s,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._load_error or "Moondream model not yet loaded",
                transient=transient, http_status=503,
                retry_after_ms=2000 if transient else None,
            )

        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_image",
                message=("moondream expects an image in the request body "
                         "(multipart 'frame' field or JSON 'frame_b64')."),
                transient=False, http_status=400,
            )

        task = (payload.get("task") or "").strip()
        question = (payload.get("question") or payload.get("prompt") or "").strip()
        if task and task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED, code="unsupported_task",
                message=f"task {task!r} not in {list(_SUPPORTED_TASKS)}",
                transient=False, http_status=400,
            )
        is_vqa = bool(question) and task != "scene_caption"

        try:
            image = _decode_image(image_bytes)
        except _ImageDecodeError as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="invalid_image",
                message=str(exc), transient=False, http_status=400,
            ) from exc

        started = time.monotonic()
        try:
            # moondream 0.0.6 (onnxruntime, no torch): encode the image once,
            # then caption/query the ENCODED image. Be defensive so a newer
            # build that accepts the raw image still works.
            enc = image
            if hasattr(self._model, "encode_image"):
                enc = self._model.encode_image(image)
            if is_vqa:
                text = str(self._model.query(enc, question)["answer"]).strip()
            else:
                try:
                    text = str(self._model.caption(enc, length="short")["caption"]).strip()
                except TypeError:
                    # 0.0.6 caption() takes no length kwarg.
                    text = str(self._model.caption(enc)["caption"]).strip()
        except Exception as exc:
            logger.exception("Moondream inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"Moondream raised during inference: {exc}",
                transient=True, http_status=500, retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        result = {
            "task": "visual_qa" if is_vqa else "scene_caption",
            "model_path": self._model_path,
        }
        if is_vqa:
            result["answer"] = text
            result["question"] = question
        else:
            result["caption"] = text
        return InferResponse(
            model_name=os.path.basename(self._model_path),
            model_version=f"moondream/{os.path.basename(self._model_path)}",
            inference_ms=elapsed_ms, result=result,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256(self._model_path.encode("utf-8")).hexdigest()
        return f"sha256-id:{digest}"


# ── Inference helpers ──────────────────────────────────────────────


class _ImageDecodeError(Exception):
    """Raised when the input bytes don't decode as an image."""


def _decode_image(image_bytes: bytes) -> Any:
    from PIL import Image, UnidentifiedImageError
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise _ImageDecodeError(
            "could not decode image bytes (expected JPEG / PNG / WebP / BMP)"
        ) from exc
    return image.convert("RGB")
