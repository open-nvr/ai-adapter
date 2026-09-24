# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ClipEmbeddingService — the ``embed`` task, over OpenCLIP.

Turns a frame into a vector, and a QUERY IN WORDS into a vector in the
same space. That second half is the whole point: OpenNVR's search fuses
a keyword ranking with a similarity ranking, and a similarity ranking
is only useful if the operator's sentence and the camera's frame land
somewhere comparable. A vision-only embedder would let you find frames
similar to another frame, which is a different and much less useful
feature.

WHY THIS ADAPTER IS ``BodyShape.TEXT``
--------------------------------------

Every other vision adapter here is ``BodyShape.IMAGE``, and this one
looks like it should be. It cannot be.

``BodyShape.IMAGE`` makes the binary MANDATORY — the SDK's JSON parser
rejects a body without ``frame_b64`` before ``infer()`` is ever called.
That is right for a detector, where a request with no frame is
meaningless. Here a request with no frame is the *query* half of the
feature, and it would be refused at the door.

So the body shape is TEXT (a JSON object, no mandatory binary) and this
service decodes ``frame_b64`` itself when it is present. The cost is
that the SDK's ``max_body_bytes`` guard does not apply to a base64
image on this path, so :data:`MAX_IMAGE_BYTES` is enforced here
instead — before the decode, on the encoded length, so an oversized
body is refused without being materialised.

Both call shapes OpenNVR already sends work unchanged:

    {"task": "embed", "text": "a red truck at the loading bay"}
    {"task": "embed", "frame_b64": "...", "camera_id": "cam3"}

MODEL, AND WHY THIS ONE
-----------------------

``ViT-B-32`` with LAION's ``laion2b_s34b_b79k`` weights, 512
dimensions, MIT-licensed — checked, because a non-commercial weights
licence would quietly poison OpenNVR's commercial licensing, and
several popular embedding checkpoints carry one.

It is the small end of the CLIP family on purpose. OpenNVR's floor is a
mini-PC, and this runs there: the image tower is ~88M parameters and
embeds a frame in tens of milliseconds on a couple of x86 cores. A
larger tower would rank slightly better and would move the floor,
which is the wrong trade for this project. ``CLIP_MODEL`` and
``CLIP_PRETRAINED`` change it for anyone who disagrees, and the
dimension is read from the model rather than assumed.

VECTORS COME BACK NORMALISED
----------------------------

Unit length, always. Cosine similarity is then a dot product, which is
what the consumer wants; more importantly it means two vectors from
this adapter are comparable without the consumer knowing anything about
the model that produced them. The ``dim`` and ``model`` in the response
are there so a store can refuse to compare vectors that are not.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import logging
import os
import platform
import threading
import time
from datetime import datetime, timezone
from typing import Any

from opennvr_adapter_sdk import AdapterService, BODY_BYTES_KEY, ServiceError
from opennvr_adapter_sdk.contract import (
    ErrorCategory,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

#: OpenCLIP architecture and pretrained tag. Both overridable.
MODEL_ENV = "CLIP_MODEL"
PRETRAINED_ENV = "CLIP_PRETRAINED"
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

#: Where weights are cached. A populated cache means no network at all.
CACHE_DIR_ENV = "CLIP_CACHE_DIR"
DEFAULT_CACHE_DIR = "/models/clip"

#: Torch intra-op threads. Left unset, torch grabs every core it can
#: see, which on a shared NVR box means the embedder competes with the
#: detector that is the reason the box exists. Two is a deliberate,
#: modest default.
THREADS_ENV = "CLIP_THREADS"
DEFAULT_THREADS = 2

#: Largest accepted image body, measured on the BASE64 text before it is
#: decoded. See the module docstring: on a TEXT body shape the SDK's own
#: limit does not cover this, so it is enforced here.
MAX_IMAGE_BYTES = 12 * 1024 * 1024

#: Longest accepted query string. CLIP's tokenizer truncates at 77
#: tokens anyway; this only stops a megabyte of prose being tokenised
#: to discover that.
MAX_TEXT_CHARS = 4096


class ClipEmbeddingService(AdapterService):
    """Image and text embeddings in one shared space."""

    def __init__(self) -> None:
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._dim: int | None = None
        self._fingerprint: str | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._gpu_in_use = False
        self._device = "cpu"
        # One model instance, and torch modules are not safe to call
        # concurrently from several threads with a shared buffer. The
        # declared max_inflight=1 already serialises at the SDK, but a
        # scheduler change upstream should not become a data race down
        # here.
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────

    def load(self) -> None:
        arch = os.getenv(MODEL_ENV, "").strip() or DEFAULT_MODEL
        pretrained = os.getenv(PRETRAINED_ENV, "").strip() or DEFAULT_PRETRAINED
        cache_dir = os.getenv(CACHE_DIR_ENV, "").strip() or DEFAULT_CACHE_DIR

        try:
            import torch
            import open_clip
        except Exception as exc:                       # pragma: no cover
            self._load_state = HealthStatus.ERROR
            self._load_error = f"open-clip-torch / torch unavailable: {exc}"
            logger.error("clip: %s", self._load_error)
            return

        threads = _int_env(THREADS_ENV, DEFAULT_THREADS)
        if threads > 0:
            try:
                torch.set_num_threads(threads)
            except Exception:                          # pragma: no cover
                pass

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gpu_in_use = self._device == "cuda"

        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            # A read-only or absent cache dir is not fatal — open_clip
            # falls back to its own default location.
            cache_dir = ""

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained,
                cache_dir=cache_dir or None,
            )
            model.eval().to(self._device)
            tokenizer = open_clip.get_tokenizer(arch)
        except Exception as exc:
            self._load_state = HealthStatus.ERROR
            self._load_error = (
                f"could not load {arch}/{pretrained}: {exc}. With no weights "
                f"cached in {cache_dir or 'the default cache'} this needs one "
                f"download; an air-gapped box must be given the cache."
            )
            logger.error("clip: %s", self._load_error)
            return

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._arch = arch
        self._pretrained = pretrained
        self._dim = _embed_dim(model)
        self._fingerprint = _weights_fingerprint(model)
        self._load_state = HealthStatus.OK
        logger.info("clip: %s/%s ready on %s, dim=%s",
                    arch, pretrained, self._device, self._dim)

    def is_ready(self) -> bool:
        return self._load_state == HealthStatus.OK

    def health_status(self) -> HealthStatus | None:
        return self._load_state

    def fingerprint(self) -> str | None:
        return self._fingerprint

    def model_info(self) -> ModelInfo:
        arch = getattr(self, "_arch", DEFAULT_MODEL)
        pretrained = getattr(self, "_pretrained", DEFAULT_PRETRAINED)
        return ModelInfo(
            name=f"open_clip/{arch}",
            version=pretrained,
            framework="open_clip",
            # BOTH directions declared, which is the machine-readable
            # form of this adapter's one unusual property: it answers a
            # request that carries no image.
            modalities_in=["image", "text"],
            modalities_out=["embedding"],
            fingerprint=self._fingerprint,
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        cpu_count = os.cpu_count() or 0
        if self._load_state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            if self._gpu_in_use:
                reasoning = "CUDA available and in use; weights loaded."
            else:
                reasoning = (
                    f"Running on CPU with {cpu_count} cores. ViT-B-32 is the "
                    f"small end of the CLIP family and embeds a frame in tens "
                    f"of milliseconds here; a GPU makes it faster, not possible."
                )
        elif self._load_state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = "Model still loading."
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Weights failed to load: {self._load_error}"

        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False,
                "gpu_in_use": self._gpu_in_use,
                "device": self._device,
                "cpu_count": cpu_count,
                "torch_threads": _int_env(THREADS_ENV, DEFAULT_THREADS),
                "embedding_dim": self._dim,
                "architecture": getattr(self, "_arch", DEFAULT_MODEL),
                "pretrained": getattr(self, "_pretrained", DEFAULT_PRETRAINED),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
        )

    # ── inference ────────────────────────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        """One embedding, of whichever modality the request carries.

        Exactly one of ``text`` or an image is expected. Both, or
        neither, is a malformed request and says so rather than picking
        — silently preferring one would make a caller's bug look like a
        bad ranking, weeks later, with nothing to trace it to.
        """
        self._require_ready()

        text = payload.get("text")
        text = text.strip() if isinstance(text, str) else None
        image_bytes = self._image_bytes(payload)

        if text and image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="ambiguous_input",
                message=("Send either 'text' or an image, not both — this "
                         "adapter returns one vector and cannot say which "
                         "modality it came from if given two."),
                transient=False,
                http_status=400,
            )
        if not text and not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                # Names the multipart limitation explicitly. This
                # adapter is TEXT-body-shaped so that a query with no
                # image is accepted at all (see the module docstring),
                # and the SDK's TEXT parser collects only string form
                # fields — a multipart FILE part never reaches here. A
                # caller who uploads one would otherwise get "send an
                # image" while looking at the image they just sent.
                message=("Send JSON with 'text' (a query) or 'frame_b64' "
                         "(a base64 frame) — this adapter embeds either. "
                         "Note that a multipart file upload is not the "
                         "image path here: this adapter accepts a body "
                         "with no binary at all, so frames travel as "
                         "'frame_b64' in JSON."),
                transient=False,
                http_status=400,
            )

        started = time.perf_counter()
        if text:
            if len(text) > MAX_TEXT_CHARS:
                raise ServiceError(
                    ErrorCategory.TRANSPORT_ERROR,
                    code="text_too_long",
                    message=f"'text' exceeds {MAX_TEXT_CHARS} characters.",
                    transient=False,
                    http_status=413,
                )
            vector = self._encode_text(text)
            modality = "text"
        else:
            vector = self._encode_image(image_bytes)
            modality = "image"

        return InferResponse(
            model_name=f"open_clip/{getattr(self, '_arch', DEFAULT_MODEL)}",
            model_version=getattr(self, "_pretrained", DEFAULT_PRETRAINED),
            inference_ms=int((time.perf_counter() - started) * 1000),
            result={
            "embedding": vector,
            # Declared, not implied. A consumer comparing vectors from
            # two models produces a confident ordering of noise, and
            # these two fields are the only way to notice.
            "dim": len(vector),
            "model": f"open_clip/{getattr(self, '_arch', DEFAULT_MODEL)}/"
                     f"{getattr(self, '_pretrained', DEFAULT_PRETRAINED)}",
            "modality": modality,
            # Stated so a consumer need not re-normalise defensively.
            "normalized": True,
            },
        )

    # ── internals ────────────────────────────────────────────────────

    def _require_ready(self) -> None:
        if self._load_state == HealthStatus.OK:
            return
        raise ServiceError(
            ErrorCategory.MODEL_ERROR,
            code=("weights_missing" if self._load_state == HealthStatus.ERROR
                  else "clip.model_loading"),
            message=self._load_error or "Model still loading.",
            transient=(self._load_state == HealthStatus.LOADING),
            http_status=503,
        )

    def _image_bytes(self, payload: dict[str, Any]) -> bytes | None:
        """Image bytes, or None.

        ``frame_b64`` is the path every caller actually uses, and the
        only one this body shape offers: the SDK's TEXT parser collects
        string form fields and never populates ``BODY_BYTES_KEY``, so a
        multipart file part does not arrive here.

        The ``BODY_BYTES_KEY`` branch is kept anyway — it costs four
        lines, it makes this service correct if it is ever mounted
        behind a different body shape, and it is the shape every other
        adapter in this repo reads. It is NOT currently reachable on
        ``BodyShape.TEXT``; ``infer()``'s refusal message is what tells
        a multipart caller that.
        """
        raw = payload.get(BODY_BYTES_KEY)
        if isinstance(raw, (bytes, bytearray)) and raw:
            if len(raw) > MAX_IMAGE_BYTES:
                raise self._too_large(len(raw))
            return bytes(raw)

        b64 = payload.get("frame_b64")
        if not isinstance(b64, str) or not b64:
            return None
        # Checked on the ENCODED length, before decoding: refusing an
        # oversized body should not require building it in memory first.
        if len(b64) > MAX_IMAGE_BYTES:
            raise self._too_large(len(b64))
        try:
            return base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"'frame_b64' is not valid base64: {exc}",
                transient=False,
                http_status=400,
            ) from exc

    @staticmethod
    def _too_large(size: int) -> ServiceError:
        return ServiceError(
            ErrorCategory.TRANSPORT_ERROR,
            code="body_too_large",
            message=f"Image exceeds {MAX_IMAGE_BYTES} bytes ({size} received).",
            transient=False,
            http_status=413,
        )

    def _encode_text(self, text: str) -> list[float]:
        import torch

        with self._lock, torch.no_grad():
            tokens = self._tokenizer([text])
            if self._device != "cpu":
                tokens = tokens.to(self._device)
            vector = self._model.encode_text(tokens)
        return _unit(vector)

    def _encode_image(self, blob: bytes) -> list[float]:
        import torch
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(io.BytesIO(blob))
            image.load()
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR,
                code="malformed_input",
                message=f"Could not decode the image: {exc}",
                transient=False,
                http_status=400,
            ) from exc

        with self._lock, torch.no_grad():
            batch = self._preprocess(image).unsqueeze(0)
            if self._device != "cpu":
                batch = batch.to(self._device)
            vector = self._model.encode_image(batch)
        return _unit(vector)


# ── helpers ──────────────────────────────────────────────────────────


def _unit(tensor: Any) -> list[float]:
    """First row of ``tensor``, unit length, as plain floats.

    A zero vector is returned unchanged rather than dividing by zero: a
    model emitting zeros is broken, and taking the request down with it
    turns a bad vector into a failed enrichment for no benefit. It will
    simply never be similar to anything.
    """
    row = tensor[0]
    norm = float(row.norm().item())
    if norm > 0:
        row = row / norm
    return [float(v) for v in row.tolist()]


def _embed_dim(model: Any) -> int | None:
    """The model's output width, asked of the model rather than assumed.

    ViT-B-32 is 512 today. Hard-coding that would make ``CLIP_MODEL``
    a trap: point it at ViT-L-14, get 768-wide vectors, and a store
    told they were 512 wide.
    """
    for attr in ("text_projection", "visual"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape):
            return int(shape[-1])
        width = getattr(obj, "output_dim", None)
        if isinstance(width, int):
            return width
    return None


def _weights_fingerprint(model: Any) -> str | None:
    """sha256 over the model's parameters.

    Not the file: open_clip may have loaded from a cache path, a
    safetensors shard or an already-open handle, and the CONTRACT wants
    a stable identifier for the weights actually in memory — which is
    what drift detection compares. Hashing the tensors answers that
    whatever the file layout was.
    """
    try:
        import torch

        digest = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(model.state_dict().items()):
                digest.update(name.encode())
                digest.update(
                    param.detach().to("cpu").contiguous().numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"
    except Exception as exc:                           # pragma: no cover
        logger.warning("clip: could not fingerprint weights (%s)", exc)
        return None


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("clip: %s=%r is not an integer; using %d",
                       name, raw, default)
        return default
