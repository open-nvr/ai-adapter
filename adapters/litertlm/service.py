# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemma on the Qualcomm Hexagon NPU through LiteRT-LM — a contract-compliant
``AdapterService`` for captions and visual question answering.

Why this exists. On a CPU-only edge box the vision-language captioner was the
one thing that could not keep up: qwen3-vl 2B through Ollama took 60–90 s per
photo at four cores on a Ventuno Q (IQ-8275), every call outran core's deadline,
and nothing was ever saved. Measured on that board on 2026-11-05, Gemma 4 E2B
compiled for the QCS8275 Hexagon v75 captions the same photo in **2.5 s warm,
4.3 s cold, with the image encoder on the NPU too**, about 1.2 GiB resident,
and the detector sharing the NPU slowed from 38 to 44 ms per inference.

**Runtime.** Google's LiteRT-LM (``litert-lm-api``) with the Linux arm64 NPU
build of its native library (``native-v0.18.0-e``), the LiteRT Qualcomm
dispatch library, and six Qualcomm AI Runtime (QAIRT 2.50) libraries fetched
ONCE at first start from Qualcomm's download server (pinned SHA-256; Qualcomm's
libraries are under Qualcomm's terms and are not redistributed in the image).
The model is the ahead-of-time compiled NPU bundle
``gemma-4-E2B-it_Qualcomm_QCS8275.litertlm`` (2.7 GB, Apache-2.0), downloaded
once into the model volume when ``OPENNVR_LITERTLM_MODEL_URL`` is set.

**Tasks.** ``scene_caption`` (key ``caption``) and ``visual_qa`` (key
``answer``): the same keys the moondream and ollamavlm adapters use, so this
is a drop-in for the ``caption-adapter`` slot. Images are downscaled to
``OPENNVR_LITERTLM_MAX_IMAGE_PX`` on the longest side first: a caption does not
need 1080p and the vision tokens scale with pixels.

**Hardware.** NPU-only: the compiled model has no CPU graph. Needs
``/dev/fastrpc-cdsp`` and the ``/dev/dma_heap`` nodes, the ``fastrpc`` and
``dmaheap`` groups and the host's ``libcdsprpc.so`` (docker-compose.accel-
qualcomm.yml in open-nvr mounts all of it). Without the device node the
adapter reports ``blocked`` and stays up, answering ``/health`` honestly.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import platform
import tempfile
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
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

MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
_SUPPORTED_TASKS: tuple[str, ...] = ("visual_qa", "scene_caption")

DEFAULT_MODEL_PATH = os.getenv(
    "OPENNVR_LITERTLM_MODEL_PATH", "/models/gemma-4-E2B-it_Qualcomm_QCS8275.litertlm")
#: Empty by default: a 2.7 GB download is the operator's call (open-nvr's
#: compose sets it to the Hugging Face URL on Qualcomm boxes).
DEFAULT_MODEL_URL = os.getenv("OPENNVR_LITERTLM_MODEL_URL", "")
DEFAULT_MODEL_SHA256 = os.getenv(
    "OPENNVR_LITERTLM_MODEL_SHA256",
    "794339191f0e1dc8353b7f84fb205af5e62870ed5647a315eddbf7a9a9c90f06")
#: Where the Qualcomm libraries and the LiteRT dispatch library live (the
#: dispatch library is baked at build; the Qualcomm ones are fetched here).
NPU_LIB_DIR = Path(os.getenv("OPENNVR_LITERTLM_NPU_LIB_DIR", "/opt/npu/lib"))
#: Where the image bakes LiteRT's Qualcomm dispatch library (copied into
#: ``NPU_LIB_DIR`` at load: LiteRT looks for it and the QNN libraries in one place).
DISPATCH_DIR = Path(os.getenv("OPENNVR_LITERTLM_DISPATCH_DIR", "/opt/npu/dispatch"))
#: The NPU build of the LiteRT-LM native library (baked at build).
RUNTIME_DIR = Path(os.getenv("OPENNVR_LITERTLM_RUNTIME_DIR", "/opt/npu/litertlm"))
MAX_IMAGE_PX = int(os.getenv("OPENNVR_LITERTLM_MAX_IMAGE_PX", "768"))
MAX_OUTPUT_TOKENS = int(os.getenv("OPENNVR_LITERTLM_MAX_OUTPUT_TOKENS", "96"))
CAPTION_PROMPT = os.getenv(
    "OPENNVR_LITERTLM_CAPTION_PROMPT",
    "Describe this security-camera photo in one sentence: who or what is in it, "
    "what they wear and what they are doing. Plain words, no preamble.")
FASTRPC_NODE = "/dev/fastrpc-cdsp"

#: Qualcomm AI Runtime pieces the dispatch library needs, read out of the
#: Community SDK zip by HTTP range requests (the zip is 2.6 GB; these six
#: are ~110 MB). Pinned to the QAIRT version the model was compiled with.
QAIRT_VERSION = "2.50.0.260828"
QAIRT_URL = ("https://softwarecenter.qualcomm.com/api/download/software/sdks/"
             f"Qualcomm_AI_Runtime_Community/All/{QAIRT_VERSION}/v{QAIRT_VERSION}.zip")
QAIRT_SIZE = 2601473189
QAIRT_ROOT = f"qairt/{QAIRT_VERSION}/lib/"
QAIRT_FILES: dict[str, tuple[str, str]] = {  # name -> (folder in the zip, sha256)
    "libQnnTFLiteDelegate.so": ("aarch64-oe-linux-gcc11.2/", "88d5ec6b722d21fc165a4519aa772822ab1b4e3a8548c3ed51046e2821aa9047"),
    "libQnnHtpPrepare.so": ("aarch64-oe-linux-gcc11.2/", "7c9a8a59ce090bb0766f5ff660809ef58e0a5e65d1c6ba46c7fa182f8d032682"),
    "libQnnHtp.so": ("aarch64-oe-linux-gcc11.2/", "823f764d434869919c7e01825ba76da7db828225525b0d27a4c21f4b2841e25b"),
    "libQnnSystem.so": ("aarch64-oe-linux-gcc11.2/", "65c1faa1c2175128fd3c3c8a8edb276e93520666afd9696977e947e6d7a3d1d2"),
    "libQnnHtpV75Stub.so": ("aarch64-oe-linux-gcc11.2/", "c9ca7044e946248ce8bbac85c54f720e9d732793fada9160ef82d037629e7046"),
    "libQnnHtpV75Skel.so": ("hexagon-v75/unsigned/", "2f6cfe5aae553d8ec88b6741bf8598a557375eacfc58056c2667fc43fda64f6a"),
}


class _RemoteZip(io.RawIOBase):
    """A big remote zip read in pieces (HTTP range requests), so ``zipfile``
    can pull out single members without downloading 2.6 GB."""

    def __init__(self, url: str, size: int, opener=urllib.request.urlopen) -> None:
        self.url, self.size, self.position, self._open = url, size, 0, opener

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer) -> int:
        if not len(buffer) or self.position >= self.size:
            return 0
        end = min(self.position + len(buffer), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with self._open(req, timeout=120) as resp:
            if getattr(resp, "status", 206) != 206:
                raise OSError("the download server did not honour a range request")
            data = resp.read()
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch_qairt_libraries(lib_dir: Path, *, opener=urllib.request.urlopen,
                          url: str = QAIRT_URL, size: int = QAIRT_SIZE) -> list[str]:
    """Make sure the six Qualcomm libraries are in ``lib_dir`` with the pinned
    checksums; fetch the missing ones. Returns the names fetched."""
    lib_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n, (_, sha) in QAIRT_FILES.items()
               if not (lib_dir / n).is_file() or _sha256_file(lib_dir / n) != sha]
    if not missing:
        return []
    logger.info("litertlm: fetching %d Qualcomm AI Runtime libraries (%s)", len(missing), QAIRT_VERSION)
    archive = zipfile.ZipFile(io.BufferedReader(_RemoteZip(url, size, opener), 1 << 20))
    for name in missing:
        folder, sha = QAIRT_FILES[name]
        data = archive.read(QAIRT_ROOT + folder + name)
        if hashlib.sha256(data).hexdigest() != sha:
            raise OSError(f"checksum mismatch for {name}")
        tmp = lib_dir / (name + ".part")
        tmp.write_bytes(data)
        tmp.replace(lib_dir / name)
    return missing


def stage_dispatch_library(dispatch_dir: Path, lib_dir: Path) -> None:
    """Put LiteRT's Qualcomm dispatch library (baked into the image) next to
    the fetched Qualcomm libraries; LiteRT reads them from one directory."""
    src = dispatch_dir / "libLiteRtDispatch_Qualcomm.so"
    dst = lib_dir / src.name
    if src.is_file() and (not dst.is_file() or dst.stat().st_size != src.stat().st_size):
        lib_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".so.part")
        tmp.write_bytes(src.read_bytes())
        tmp.replace(dst)


def download_model(url: str, dest: Path, sha256: str | None,
                   *, opener=urllib.request.urlopen) -> None:
    """Download the model once, to a temporary name, and verify it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("litertlm: downloading the model from %s (2.7 GB)", url)
    with opener(url, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if sha256 and _sha256_file(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise OSError("model checksum mismatch; download refused")
    tmp.replace(dest)


def shrink_image(image_bytes: bytes, max_px: int) -> tuple[bytes, tuple[int, int]]:
    """JPEG bytes with the longest side at most ``max_px`` (never upscaled),
    and the size handed to the model."""
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("could not decode image bytes (expected JPEG / PNG / WebP / BMP)") from exc
    img = img.convert("RGB")
    w, h = img.size
    scale = max_px / float(max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88)
    return out.getvalue(), img.size


_TOOL_CLASS_CACHE: dict[int, type] = {}


def agent_tool_class(base: type) -> type:
    """A ``litert_lm.Tool`` subclass that presents an OpenAI function envelope
    as given and never executes: automatic tool calling is off, the caller
    runs its own tools and sends the result back as a message."""
    cls = _TOOL_CLASS_CACHE.get(id(base))
    if cls is None:
        class AgentTool(base):  # type: ignore[misc,valid-type]
            def __init__(self, spec: dict[str, Any]) -> None:
                self.spec = spec if "function" in spec else {"type": "function", "function": spec}

            def get_tool_description(self) -> dict[str, Any]:
                return self.spec

            def execute(self, param: Any) -> Any:
                return {"error": "tool execution is the caller's"}
        cls = _TOOL_CLASS_CACHE[id(base)] = AgentTool
    return cls


class LitertLmService(AdapterService):
    """Gemma 4 E2B on the Hexagon NPU through LiteRT-LM.

    The engine is loaded once in a background thread (``load`` returns at
    once so ``/health`` answers from the first second: ``loading`` until the
    model is in, ``ok`` after, ``error`` with the reason if it never comes).
    One conversation per request, one request at a time.
    """

    def __init__(self, *, model_path: str | None = None, model_url: str | None = None,
                 model_sha256: str | None = None, lib_dir: Path | None = None,
                 runtime_dir: Path | None = None, max_image_px: int | None = None,
                 fetch_libraries: bool = True, engine_factory=None) -> None:
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._model_url = (DEFAULT_MODEL_URL if model_url is None else model_url).strip()
        self._model_sha256 = (DEFAULT_MODEL_SHA256 if model_sha256 is None else model_sha256) or None
        self._lib_dir = lib_dir or NPU_LIB_DIR
        self._runtime_dir = runtime_dir or RUNTIME_DIR
        self._max_image_px = max_image_px or MAX_IMAGE_PX
        self._fetch_libraries = fetch_libraries
        self._engine_factory = engine_factory or self._make_engine
        self._engine: Any | None = None
        self._lm: Any | None = None
        self._state: HealthStatus = HealthStatus.LOADING
        self._error: str | None = None
        self._lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self._started_at = datetime.now(timezone.utc)
        self._fingerprint_cache: str | None = None
        self._captions = 0
        self._last_ms: int | None = None

    # ── lifecycle ─────────────────────────────────────────────────

    def load(self) -> None:
        if self._loader is not None:
            return
        self.metrics.register_counter("adapter_generated_chars_total",
                                      "Characters of caption/answer text generated.")
        self._loader = threading.Thread(target=self._load_blocking, name="litertlm-load", daemon=True)
        self._loader.start()

    def load_now(self) -> None:
        """Load on the calling thread (tests, tools)."""
        self._load_blocking()

    def _load_blocking(self) -> None:
        try:
            if not os.path.exists(FASTRPC_NODE):
                raise RuntimeError(
                    f"{FASTRPC_NODE} is not present: this adapter runs the model on the "
                    "Qualcomm Hexagon NPU only (open-nvr's docker-compose.accel-qualcomm.yml "
                    "passes the node, the dma_heap nodes and the fastrpc group).")
            if self._fetch_libraries:
                fetched = fetch_qairt_libraries(self._lib_dir)
                if fetched:
                    logger.info("litertlm: fetched %s", ", ".join(fetched))
                stage_dispatch_library(DISPATCH_DIR, self._lib_dir)
            if not self._model_path.is_file():
                if not self._model_url:
                    raise RuntimeError(
                        f"model file {self._model_path} is missing and OPENNVR_LITERTLM_MODEL_URL "
                        "is empty: mount the model or set the URL for a one-time download.")
                download_model(self._model_url, self._model_path, self._model_sha256)
            self._engine = self._engine_factory()
            self._fingerprint_cache = self._compute_fingerprint()
            with self._lock:
                self._state, self._error = HealthStatus.OK, None
            logger.info("litertlm: ready: %s on the NPU", self._model_path.name)
        except Exception as exc:  # noqa: BLE001 - reported through /health
            with self._lock:
                self._state, self._error = HealthStatus.ERROR, str(exc)
            logger.exception("litertlm: load failed")

    def _make_engine(self) -> Any:
        import litert_lm as lm  # the Python package; the native library is swapped below

        self._lm = lm
        self._use_bundled_runtime(lm)
        npu = lm.Backend.NPU(litert_dispatch_lib_dir=str(self._lib_dir))
        return lm.Engine(str(self._model_path), backend=npu, vision_backend=npu)

    def _use_bundled_runtime(self, lm: Any) -> None:
        """Make ``litert-lm-api`` load the NPU build of LiteRT-LM instead of the
        library in the wheel, which leaves the HTP in its default power mode on
        Linux (about three times slower). Relies on the package's internals,
        hence the exact version pin; a failure here is logged, not fatal."""
        bundled = self._runtime_dir / "libLiteRtLm.so"
        if not bundled.is_file():
            logger.warning("litertlm: %s missing; using the wheel's runtime (slower)", bundled)
            return
        try:
            import ctypes
            import importlib

            ffi = importlib.import_module("litert_lm._ffi")
            real = ctypes.CDLL

            def cdll(path, *a, **kw):
                return real(str(bundled) if str(path).endswith("liblitert-lm.so") else path, *a, **kw)

            ctypes.CDLL = cdll
            try:
                ffi._get_lib()
            finally:
                ctypes.CDLL = real
            if not str(getattr(ffi._LIB, "_name", "")).endswith("libLiteRtLm.so"):
                logger.warning("litertlm: runtime swap did not take; using the wheel's runtime")
        except Exception:  # noqa: BLE001
            logger.warning("litertlm: runtime swap failed; using the wheel's runtime", exc_info=True)

    # ── contract ──────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._state == HealthStatus.OK

    def health_status(self) -> HealthStatus | None:
        return self._state

    def fingerprint(self) -> str | None:
        return self._fingerprint_cache

    def model_info(self) -> ModelInfo:
        size_mb = None
        try:
            size_mb = round(self._model_path.stat().st_size / (1024 * 1024), 1)
        except OSError:
            pass
        return ModelInfo(
            name=self._model_path.name, version=f"litert-lm/{self._model_path.name}",
            framework="litert-lm", size_mb=size_mb,
            modalities_in=["image", "text"], modalities_out=["text"],
            fingerprint=self.fingerprint(),
        )

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        node = os.path.exists(FASTRPC_NODE)
        if self._state == HealthStatus.OK:
            verdict = HardwareVerdict.OK
            reasoning = (f"Gemma on the Hexagon NPU through LiteRT-LM; last caption "
                         f"{self._last_ms} ms" if self._last_ms is not None else
                         "Gemma loaded on the Hexagon NPU through LiteRT-LM.")
        elif self._state == HealthStatus.LOADING:
            verdict = HardwareVerdict.WARN
            reasoning = ("Loading: fetching the Qualcomm libraries or the 2.7 GB model, "
                         "then compiling the NPU session." if node else
                         f"{FASTRPC_NODE} is missing: this adapter needs the Qualcomm NPU.")
        else:
            verdict = HardwareVerdict.BLOCKED
            reasoning = f"Load failed: {self._error}"
        return HardwareEvaluationResponse(
            verdict=verdict, reasoning=reasoning, checked_at=datetime.now(timezone.utc),
            details={
                "accelerator": "qualcomm-hexagon-npu", "npu_node_present": node,
                "model_path": str(self._model_path), "qairt_version": QAIRT_VERSION,
                "max_image_px": self._max_image_px, "captions_served": self._captions,
                "last_inference_ms": self._last_ms,
                "platform": platform.platform(), "python_version": platform.python_version(),
                "adapter_uptime_seconds": int((datetime.now(timezone.utc) - self._started_at).total_seconds()),
            },
        )

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if not self.is_ready():
            transient = self._state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="model_not_loaded",
                message=self._error or "Gemma is still loading on the NPU",
                transient=transient, http_status=503, retry_after_ms=5000 if transient else None)
        image_bytes = payload.get("__file__")
        if not image_bytes:
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="missing_image",
                message="litertlm expects an image in the request body (multipart 'frame' or JSON 'frame_b64').",
                transient=False, http_status=400)
        task = (payload.get("task") or "").strip()
        question = (payload.get("question") or payload.get("prompt") or "").strip()
        if task and task not in _SUPPORTED_TASKS:
            raise ServiceError(
                ErrorCategory.NOT_SUPPORTED, code="unsupported_task",
                message=f"task {task!r} not in {list(_SUPPORTED_TASKS)}", transient=False, http_status=400)
        is_vqa = bool(question) and task != "scene_caption"
        prompt = question if is_vqa else CAPTION_PROMPT
        try:
            jpeg, size = shrink_image(image_bytes, self._max_image_px)
        except ValueError as exc:
            raise ServiceError(ErrorCategory.TRANSPORT_ERROR, code="invalid_image",
                               message=str(exc), transient=False, http_status=400) from exc
        started = time.monotonic()
        try:
            text = self._ask(jpeg, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("litertlm: inference failed")
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"LiteRT-LM raised during inference: {exc}",
                transient=True, http_status=500, retry_after_ms=1000) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        self._last_ms, self._captions = elapsed_ms, self._captions + 1
        result: dict[str, Any] = {
            "task": "visual_qa" if is_vqa else "scene_caption",
            "model_path": str(self._model_path), "image_px": list(size),
            "accelerator": "qualcomm-hexagon-npu",
        }
        if is_vqa:
            result["answer"], result["question"] = text, question
        else:
            result["caption"] = text
        try:
            self.metrics.inc_counter("adapter_generated_chars_total", len(text))
        except Exception:  # pragma: no cover
            pass
        return InferResponse(
            model_name=self._model_path.name, model_version=f"litert-lm/{self._model_path.name}",
            inference_ms=elapsed_ms, result=result)

    class NotReady(RuntimeError):
        """The engine is not loaded (still loading, or the load failed)."""

    class PromptTooLong(RuntimeError):
        """The prompt does not fit the bundle's compiled context window."""

    def chat(self, *, system: str | None, history: list[dict[str, Any]],
             user: str | list[dict[str, Any]],
             tools: list[dict[str, Any]], max_tokens: int,
             temperature: float | None = None) -> tuple[str, list[dict[str, Any]], bool]:
        """One chat turn on the same engine as the captions: returns (text,
        tool calls as [{name, arguments, id}], True). ``user`` is the text
        to send, or the tool results (the runtime's ``tool_response``
        turns) when the caller's loop is handing a result back. Serialised
        with the captions by the service lock. A prompt the window cannot
        hold raises PromptTooLong; the runtime says only that the send
        failed, and with a prompt past the compiled context that is what
        it means."""
        if not self.is_ready():
            raise self.NotReady(self._error or "the model is still loading on the NPU")
        lm = self._lm
        kwargs: dict[str, Any] = {"max_output_tokens": int(max_tokens), "automatic_tool_calling": False}
        if system:
            kwargs["system_message"] = system
        if history:
            kwargs["messages"] = history
        if tools and lm is not None:
            kwargs["tools"] = [agent_tool_class(lm.Tool)(t) for t in tools]
        if lm is not None and hasattr(lm, "ThinkingConfig"):
            kwargs["thinking_config"] = lm.ThinkingConfig(enable_thinking=False, thinking_token_budget=0)
        if temperature is not None and lm is not None and hasattr(lm, "SamplerConfig"):
            try:
                kwargs["sampler_config"] = lm.SamplerConfig(temperature=float(temperature))
            except Exception:  # noqa: BLE001 — an odd temperature is not worth refusing the turn
                pass
        with self._lock:
            try:
                with self._engine.create_conversation(**kwargs) as conversation:
                    response = conversation.send_message(user)
            except RuntimeError as exc:
                if "send_message failed" in str(exc) or "overflow" in str(exc).lower():
                    raise self.PromptTooLong(
                        "the prompt does not fit the model's context window; send fewer tool "
                        "schemas or less history") from exc
                raise
        text = ""
        try:
            items = response.get("content", []) if hasattr(response, "get") else []
        except Exception:  # noqa: BLE001
            items = []
        for p in items or []:
            if isinstance(p, dict) and p.get("type") == "text":
                text += p.get("text", "")
        calls: list[dict[str, Any]] = []
        for c in getattr(response, "tool_calls", None) or []:
            calls.append({"name": getattr(c, "name", None), "arguments": getattr(c, "arguments", None) or {},
                          "id": getattr(c, "id", None)})
        return " ".join(text.split()).strip(), calls, True

    def _ask(self, jpeg: bytes, prompt: str) -> str:
        """One bounded conversation turn with the image; serialised."""
        with self._lock, tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(jpeg)
            tmp.flush()
            kwargs: dict[str, Any] = {"max_output_tokens": MAX_OUTPUT_TOKENS}
            if self._lm is not None and hasattr(self._lm, "ThinkingConfig"):
                kwargs["thinking_config"] = self._lm.ThinkingConfig(enable_thinking=False, thinking_token_budget=0)
            with self._engine.create_conversation(**kwargs) as conversation:
                response = conversation.send_message({
                    "role": "user",
                    "content": [{"type": "image", "path": tmp.name}, {"type": "text", "text": prompt}],
                })
        text = "".join(p.get("text", "") for p in response.get("content", []) if p.get("type") == "text")
        text = " ".join(text.split()).strip()
        if not text:
            raise RuntimeError("the model returned no text")
        return text

    def _compute_fingerprint(self) -> str:
        try:
            st = self._model_path.stat()
            seed = f"{self._model_path.name}:{st.st_size}:{QAIRT_VERSION}"
        except OSError:
            seed = str(self._model_path)
        return "sha256-id:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()
