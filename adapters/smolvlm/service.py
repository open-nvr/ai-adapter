# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
SmolVLM2 VLM service — SmolVLM2-2.2B-Instruct (GGUF + mmproj) served by a
persistent multimodal ``llama-server`` child. Torch-free, CPU-first.

On-demand vision only (the agent calls it per visual question — never a
continuous stream). ``load()`` spawns ``llama-server -m <gguf> --mmproj <proj>``
on loopback inside the container; ``infer()`` resizes the frame (Pillow),
builds an in-memory JPEG data-URL, and proxies to the child's OpenAI-compatible
``/v1/chat/completions`` with an image content part.

Input:  image bytes at payload["__file__"] (JPEG/PNG) + {"question"|"prompt"}
Output: InferResponse result = {"text": "...", "caption": "..."}
"""
from __future__ import annotations

import atexit
import base64
import hashlib
import io
import logging
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Any

import httpx

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
from opennvr_adapter_sdk.model_fetch import ensure_model_file

logger = logging.getLogger(__name__)

MODEL_PATH: str = os.getenv("OPENNVR_VLM_MODEL_PATH", "/app/models/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf")
MMPROJ_PATH: str = os.getenv("OPENNVR_VLM_MMPROJ_PATH", "/app/models/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf")
# Both weights download to their paths on first boot when absent. Set the URLs
# to empty strings to forbid any fetch (offline / sovereignty-strict installs).
_SMOLVLM_REPO = "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main"
MODEL_URL: str = os.getenv(
    "OPENNVR_VLM_MODEL_URL", f"{_SMOLVLM_REPO}/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
)
MMPROJ_URL: str = os.getenv(
    "OPENNVR_VLM_MMPROJ_URL", f"{_SMOLVLM_REPO}/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"
)
SERVER_BIN: str = os.getenv("LLAMACPP_SERVER_BIN", "llama-server")
CTX_SIZE: int = int(os.getenv("SMOLVLM_CTX_SIZE", "4096"))
THREADS: int = int(os.getenv("SMOLVLM_THREADS", str(os.cpu_count() or 4)))
GPU_LAYERS: int = int(os.getenv("SMOLVLM_GPU_LAYERS", "0"))
INTERNAL_HOST: str = "127.0.0.1"
INTERNAL_PORT: int = int(os.getenv("SMOLVLM_INTERNAL_PORT", "8081"))
STARTUP_TIMEOUT_S: float = float(os.getenv("SMOLVLM_STARTUP_TIMEOUT_S", "180"))
MAX_TOKENS: int = int(os.getenv("SMOLVLM_MAX_TOKENS", "120"))
MAX_LONG_EDGE: int = int(os.getenv("SMOLVLM_MAX_LONG_EDGE", "768"))
JPEG_QUALITY: int = int(os.getenv("SMOLVLM_JPEG_QUALITY", "82"))
MAX_IMAGE_BYTES: int = int(os.getenv("SMOLVLM_MAX_IMAGE_BYTES", str(16 * 1024 * 1024)))

VISUAL_SYSTEM = (
    "You are analysing a single live security-camera frame. Answer the user's "
    "exact question using only visible evidence. Do not infer identities, "
    "intentions, or events that are not visually supported. If the frame is "
    "unclear, dark, or obstructed, say so. Answer in one or two short sentences."
)
DEFAULT_QUESTION = "What is happening in this frame right now?"


class SmolVlmService(AdapterService):
    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or MODEL_PATH
        self._mmproj_path = MMPROJ_PATH
        self._base_url = f"http://{INTERNAL_HOST}:{INTERNAL_PORT}"
        self._proc: subprocess.Popen | None = None
        self._http: httpx.Client | None = None
        self._load_state: HealthStatus = HealthStatus.LOADING
        self._load_error: str | None = None
        self._fingerprint_cache: str | None = None
        self._lock = threading.Lock()
        self._started_at = datetime.now(timezone.utc)

    def load(self) -> None:
        with self._lock:
            if self._load_state == HealthStatus.OK:
                return
            try:
                ensure_model_file(
                    self._model_path, MODEL_URL, label="SmolVLM GGUF", logger=logger
                )
                ensure_model_file(
                    self._mmproj_path, MMPROJ_URL, label="SmolVLM projector",
                    logger=logger,
                )
                self._fingerprint_cache = _cheap_fingerprint(self._model_path)
                self._spawn_server()
                self._http = httpx.Client(base_url=self._base_url, timeout=120.0)
                self._wait_healthy()
                self._load_state = HealthStatus.OK
                self._load_error = None
                logger.info("SmolVlmService ready at %s", self._base_url)
            except Exception as exc:
                self._load_state = HealthStatus.ERROR
                self._load_error = str(exc)
                logger.exception("SmolVlmService failed to load")

    def _spawn_server(self) -> None:
        args = [
            SERVER_BIN, "-m", self._model_path, "--mmproj", self._mmproj_path,
            "-c", str(CTX_SIZE), "-t", str(THREADS),
            "--host", INTERNAL_HOST, "--port", str(INTERNAL_PORT),
        ]
        if GPU_LAYERS > 0:
            args += ["-ngl", str(GPU_LAYERS)]
        logger.info("spawning multimodal llama-server: %s …", " ".join(args[:3]))
        self._proc = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        atexit.register(self._terminate)

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        probe = httpx.Client(base_url=self._base_url, timeout=2.0)
        try:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited during startup (code {self._proc.returncode})"
                    )
                try:
                    if probe.get("/health").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            raise TimeoutError(f"multimodal llama-server not healthy within {STARTUP_TIMEOUT_S}s")
        finally:
            probe.close()

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()

    def is_ready(self) -> bool:
        return (
            self._load_state == HealthStatus.OK
            and self._proc is not None
            and self._proc.poll() is None
        )

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._model_path),
            version="q4_k_m",
            framework="llama.cpp",
            modalities_in=["image", "text"],
            modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        if self._load_state == HealthStatus.OK:
            verdict, reasoning = HardwareVerdict.OK, "SmolVLM2 GGUF served by llama.cpp multimodal."
        elif self._load_state == HealthStatus.LOADING:
            verdict, reasoning = HardwareVerdict.WARN, "Model still loading."
        else:
            verdict, reasoning = HardwareVerdict.BLOCKED, f"Load failed: {self._load_error}"
        uptime = int((datetime.now(timezone.utc) - self._started_at).total_seconds())
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning,
            checked_at=datetime.now(timezone.utc),
            details={
                "gpu_required": False, "gpu_layers": GPU_LAYERS,
                "runtime": "llama.cpp-multimodal", "model_path": self._model_path,
                "mmproj_path": self._mmproj_path, "threads": THREADS,
                "cpu_count": os.cpu_count() or 0, "platform": platform.platform(),
                "adapter_uptime_seconds": uptime,
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._load_state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._load_error or "VLM not yet loaded",
                transient=transient, http_status=503,
                retry_after_ms=2000 if transient else None,
            )
        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_image",
                message="no image supplied (multipart 'frame' or JSON 'frame_b64')",
                transient=False, http_status=400,
            )
        question = str(payload.get("question") or payload.get("prompt") or DEFAULT_QUESTION)
        try:
            data_url = _to_resized_jpeg_data_url(image_bytes)
        except Exception as exc:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message=f"could not decode image: {exc}",
                transient=False, http_status=400,
            ) from exc

        body = {
            "messages": [
                {"role": "system", "content": VISUAL_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ]},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": float(payload.get("temperature", 0.2)),
            "stream": False,
        }
        started = time.monotonic()
        try:
            resp = self._http.post("/v1/chat/completions", json=body)  # type: ignore[union-attr]
        except httpx.HTTPError as exc:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_runtime_crash",
                message=f"llama-server unreachable: {exc}",
                transient=True, http_status=502, retry_after_ms=1000,
            ) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if resp.status_code != 200:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"llama-server HTTP {resp.status_code}: {resp.text[:200]}",
                transient=True, http_status=502,
            )
        text = ((resp.json().get("choices") or [{}])[0].get("message", {}).get("content") or "").strip()
        return InferResponse(
            model_name=os.path.basename(self._model_path),
            model_version="q4_k_m",
            inference_ms=elapsed_ms,
            result={"text": text, "caption": text},
        )


def _to_resized_jpeg_data_url(image_bytes: bytes) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / float(long_edge)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _cheap_fingerprint(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode())
    chunk = 4 * 1024 * 1024
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(0, size - chunk))
            h.update(f.read(chunk))
    return f"sha256:{h.hexdigest()[:16]}"
