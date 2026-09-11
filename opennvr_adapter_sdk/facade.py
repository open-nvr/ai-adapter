# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0

"""
The ``Adapter`` facade — the front door of the adapter SDK.

Everything here compiles down to :class:`~.service.AdapterService` and
:class:`~.adapter_app.AdapterApp`; nothing is a new runtime. A facade
adapter and a hand-written one produce the same service, the same
contract endpoints, the same metrics and the same specs. The facade
exists because publishing a model should not require first learning the
contract's health state machine, its error taxonomy, its fingerprint
convention and its hardware-evaluation payload.

A whole adapter::

    from opennvr_adapter_sdk import Adapter

    adapter = Adapter(
        "fall-detection",
        version="1.0.0",
        vendor="ACME",
        license="Apache-2.0",
        tasks=["object_detection"],
        framework="onnxruntime",
        weights="models/fall.onnx",
    )

    @adapter.load()
    def load():
        import onnxruntime as ort
        return ort.InferenceSession(adapter.weights)

    @adapter.on_image()
    def detect(call):
        boxes = call.model.run(None, {"images": preprocess(call.image)})
        return [call.detection("fallen", score, x, y, w, h)
                for score, (x, y, w, h) in boxes]

    app = adapter.app          # uvicorn my_model:app

What the facade derives, so a model developer does not write it:

* **the fingerprint** — sha256 of the weights file when ``weights=`` is
  given, else a deterministic value from name and version. KAI-C's
  drift detection skips a null fingerprint, so the default should not
  be null;
* **health** — LOADING until ``@adapter.load()`` returns, OK after,
  ERROR with the exception message if it raises. ``/health`` and
  ``/infer`` answer correctly at every point without a state machine;
* **hardware evaluation** — a verdict from the load result plus the
  accelerator the adapter declares, with useful diagnostics. Override
  with ``@adapter.check_hardware()`` when the model has real hardware
  requirements to test;
* **model info** — name, version, framework, weights size, and the
  modalities implied by which handler was registered;
* **the error taxonomy** — an ordinary ``ValueError`` from a handler
  becomes a 400 ``transport_error``, anything else a 500
  ``model_error``, and :class:`~.service.ServiceError` still passes
  through untouched when a handler wants to be precise;
* **the body shape** — ``@adapter.on_image`` is
  ``BodyShape.IMAGE``, and so on, which is also what shapes the
  ``/infer`` request body in the published OpenAPI document.

When an adapter outgrows this, implement :class:`~.service.AdapterService`
directly: the facade is additive and the base classes are unchanged.
"""
from __future__ import annotations

import hashlib
import logging
import os
import platform
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from opennvr_adapter_sdk.adapter_app import BODY_BYTES_KEY, AdapterApp, BodyShape
from opennvr_adapter_sdk.contract import (
    Cost,
    ErrorCategory,
    FairQueuing,
    HardwareEvaluationResponse,
    HardwareVerdict,
    HealthStatus,
    InferResponse,
    ModelInfo,
    Permissions,
    Scheduling,
)
from opennvr_adapter_sdk.service import AdapterService, ServiceError

logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Which body shape each handler decorator implies, and the modalities
#: it puts in ``ModelInfo`` — the two things a developer should not have
#: to state twice.
_KIND_TO_SHAPE: dict[str, BodyShape] = {
    "image": BodyShape.IMAGE,
    "audio": BodyShape.AUDIO,
    "text": BodyShape.TEXT,
    "data": BodyShape.GENERIC,
}


class Overloaded(Exception):
    """Raise from a handler to shed load: a 503 with ``retry_after_ms``.

    The honest way to apply backpressure. KAI-C backs off and retries
    rather than treating the call as a model failure."""

    def __init__(self, message: str = "Adapter is at capacity.",
                 *, retry_after_ms: int = 1000) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms


class InferCall:
    """One inference request — what a handler is called with.

    Flat on purpose: the binary payload is an attribute, the caller's
    params are a dict, and the loaded model is right there. Anything the
    facade does not model stays available as :attr:`payload`, which is
    exactly what :meth:`AdapterService.infer` would have received.
    """

    __slots__ = ("payload", "model", "_adapter")

    def __init__(self, payload: dict[str, Any], model: Any,
                 adapter: "Adapter") -> None:
        self.payload = payload
        #: Whatever ``@adapter.load()`` returned — the session, the
        #: pipeline, the weights handle.
        self.model = model
        self._adapter = adapter

    # ── What arrived ───────────────────────────────────────────────

    @property
    def body(self) -> bytes:
        """The raw binary payload, for any non-text adapter."""
        value = self.payload.get(BODY_BYTES_KEY)
        return value if isinstance(value, (bytes, bytearray)) else b""

    #: The frame, for an ``@adapter.on_image`` handler.
    image = body
    #: The clip, for an ``@adapter.on_audio`` handler.
    audio = body
    #: The bytes, for an ``@adapter.on_data`` handler.
    data = body

    @property
    def params(self) -> dict[str, Any]:
        """Everything the caller sent besides the binary body — the
        adapter's own knobs, plus ``task`` and ``camera_id``."""
        return {k: v for k, v in self.payload.items() if k != BODY_BYTES_KEY}

    @property
    def text(self) -> str:
        """The prompt or utterance, for a text adapter. Looks at the
        conventional keys before giving up."""
        for key in ("text", "prompt", "input"):
            value = self.payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    @property
    def task(self) -> str:
        """Which of the adapter's advertised tasks this call is for."""
        return str(self.payload.get("task") or "")

    @property
    def camera_id(self) -> str:
        """The camera, when the caller knows it."""
        return str(self.payload.get("camera_id") or "")

    def param(self, name: str, default: Any = None) -> Any:
        """One caller param, with a default."""
        return self.payload.get(name, default)

    # ── Shaping the answer ─────────────────────────────────────────

    @staticmethod
    def detection(label: str, confidence: float, x: float, y: float,
                  w: float, h: float, *, track_id: str | int | None = None,
                  **attributes: Any) -> dict[str, Any]:
        """One §5.1 detection, in the shape every consumer expects.

        Coordinates are NORMALIZED (0–1 of the frame) — the single most
        common thing to get wrong, and the reason a detection that looks
        right on the adapter shows up in the wrong place on the
        operator's screen. Divide pixel coordinates by the frame size.
        """
        item: dict[str, Any] = {
            "label": str(label),
            "confidence": _clamp(confidence),
            "bbox": {"x": _clamp(x), "y": _clamp(y),
                     "w": _clamp(w), "h": _clamp(h)},
        }
        if track_id is not None:
            item["track_id"] = track_id
        if attributes:
            item["attributes"] = attributes
        return item

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (f"<InferCall task={self.task!r} camera={self.camera_id!r} "
                f"bytes={len(self.body)} params={sorted(self.params)}>")


def _clamp(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class _Handler:
    fn: Callable[[InferCall], Any]
    kind: str
    tasks: tuple[str, ...]


class Adapter:
    """A whole AI adapter: identity, model, inference, lifecycle.

    Construct one at module scope, decorate the model loader and the
    inference handler, and expose :attr:`app` to your server. The six
    contract endpoints, auth, correlation ids, Prometheus metrics, body
    parsing, the failure envelope, the OpenAPI and AsyncAPI documents
    and the lifespan are inherited from :class:`~.adapter_app.AdapterApp`.
    """

    def __init__(
        self,
        adapter_id: str,
        *,
        version: str = "1.0.0",
        vendor: str = "",
        license: str = "",
        tasks: Sequence[str] = (),
        framework: str = "custom",
        weights: str | os.PathLike[str] | None = None,
        model_version: str | None = None,
        model_card_url: str | None = None,
        modalities_in: Sequence[str] | None = None,
        modalities_out: Sequence[str] | None = None,
        gpu: bool = False,
        network_egress: Sequence[str] = (),
        max_inflight: int = 1,
        max_body_bytes: int = 8 * 1024 * 1024,
        cost: Cost | None = None,
    ) -> None:
        if not _ID_RE.match((adapter_id or "").strip()):
            raise ValueError(
                f"Adapter({adapter_id!r}): the id must be kebab-case — "
                f"lowercase letters and digits with single hyphens "
                f"(e.g. 'fall-detection'). It becomes the adapter's name in "
                f"/capabilities, the image name and the KAI-C registration."
            )
        self.id = adapter_id.strip()
        self.version = version
        self.vendor = vendor
        self.license = license
        self.tasks = tuple(tasks)
        self.framework = framework
        #: Path to the weights, when there is a file. Used for the
        #: fingerprint and the reported size, and handy in ``load()``.
        self.weights: str | None = str(weights) if weights else None
        self.model_version = model_version or version
        self.model_card_url = model_card_url
        self._modalities_in = tuple(modalities_in or ())
        self._modalities_out = tuple(modalities_out or ())
        self.gpu = gpu
        self._network_egress = tuple(network_egress)
        self._max_inflight = max_inflight
        self._max_body_bytes = max_body_bytes
        self._cost = cost

        self._load_fn: Callable[[], Any] | None = None
        self._handler: _Handler | None = None
        self._hardware_fn: Callable[[Any], Any] | None = None
        self._shutdown_fn: Callable[[Any], Any] | None = None
        self._stream_fn: Callable[[Any], Any] | None = None
        self._service: "_FacadeService | None" = None
        self._app: AdapterApp | None = None

    # ── Declaration ────────────────────────────────────────────────

    def load(self) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
        """Register the model loader, called once at startup.

        Whatever it returns becomes ``call.model``. Import heavy ML
        libraries inside it, not at module top: a broken dependency then
        shows up as a red ``/health`` with the real error message rather
        than as a container that will not import."""

        def decorate(fn: Callable[[], Any]):
            self._load_fn = fn
            return fn

        return decorate

    def on_image(self, *tasks: str) -> Callable[..., Any]:
        """The inference handler for an adapter that takes a frame.

        ``call.image`` is the JPEG/PNG bytes, ``call.params`` the
        caller's knobs, ``call.model`` what ``load()`` returned. Return
        a list of :meth:`InferCall.detection` items, a result dict, or an
        :class:`~.contract.InferResponse` for full control."""
        return self._handler_decorator("image", tasks)

    def on_audio(self, *tasks: str) -> Callable[..., Any]:
        """The inference handler for an adapter that takes an audio clip
        (``call.audio``)."""
        return self._handler_decorator("audio", tasks)

    def on_text(self, *tasks: str) -> Callable[..., Any]:
        """The inference handler for a text-in adapter (``call.text``);
        no binary upload is parsed."""
        return self._handler_decorator("text", tasks)

    def on_data(self, *tasks: str) -> Callable[..., Any]:
        """The inference handler for any other binary payload
        (``call.data``)."""
        return self._handler_decorator("data", tasks)

    def _handler_decorator(self, kind: str, tasks: tuple[str, ...]):
        def decorate(fn: Callable[[InferCall], Any]):
            if self._handler is not None:
                raise RuntimeError(
                    f"Adapter({self.id!r}) already has an inference handler "
                    f"({self._handler.fn.__name__}). One adapter answers with "
                    f"one body shape; branch on call.task inside the handler "
                    f"to serve several tasks."
                )
            self._handler = _Handler(fn=fn, kind=kind, tasks=tasks or self.tasks)
            return fn

        return decorate

    def check_hardware(self) -> Callable[..., Any]:
        """Override the derived hardware verdict.

        Called with the loaded model; return a
        :class:`~.contract.HardwareEvaluationResponse`, a
        :class:`~.contract.HardwareVerdict`, a bool, or a
        ``(verdict, reasoning)`` pair. Use it when the model has a real
        requirement to test — a CUDA device, an NPU, enough RAM."""

        def decorate(fn: Callable[[Any], Any]):
            self._hardware_fn = fn
            return fn

        return decorate

    def on_shutdown(self) -> Callable[..., Any]:
        """Run on the way out, with the loaded model — release a device,
        close a session."""

        def decorate(fn: Callable[[Any], Any]):
            self._shutdown_fn = fn
            return fn

        return decorate

    def on_stream(self) -> Callable[..., Any]:
        """Implement the §6 WebSocket protocol yourself.

        Declaring it advertises streaming in ``/capabilities`` and
        publishes the protocol in the adapter's AsyncAPI document. The
        handler is called with the raw WebSocket."""

        def decorate(fn: Callable[[Any], Any]):
            self._stream_fn = fn
            return fn

        return decorate

    # ── Compilation ────────────────────────────────────────────────

    @property
    def fingerprint(self) -> str:
        """sha256 of the weights file, or a deterministic value derived
        from the adapter's identity when there is no file.

        Never ``None``: KAI-C's drift detection skips a null fingerprint,
        so an adapter with one is silently exempt from the tamper check
        that protects the operator."""
        if self.weights:
            path = Path(self.weights)
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 16), b""):
                        digest.update(chunk)
                return f"sha256:{digest.hexdigest()}"
        seed = f"{self.id}:{self.model_version}:{self.framework}".encode()
        return f"sha256:{hashlib.sha256(seed).hexdigest()}"

    def model_info(self) -> ModelInfo:
        """The ``/capabilities`` model block, derived from the
        declaration and the handler that was registered."""
        kind = self._handler.kind if self._handler else "data"
        size_mb = None
        if self.weights and Path(self.weights).is_file():
            size_mb = round(Path(self.weights).stat().st_size / (1024 * 1024), 2)
        modalities_in = list(self._modalities_in) or [
            {"image": "image", "audio": "audio", "text": "text"}.get(kind, "binary")]
        modalities_out = list(self._modalities_out) or [
            {"image": "bbox_classes", "audio": "text", "text": "text"}.get(kind, "json")]
        return ModelInfo(
            name=self.id,
            version=self.model_version,
            framework=self.framework,
            size_mb=size_mb,
            modalities_in=modalities_in,
            modalities_out=modalities_out,
            fingerprint=self.fingerprint,
        )

    @property
    def service(self) -> AdapterService:
        """The :class:`~.service.AdapterService` this adapter compiles
        to — anything that accepts one accepts this."""
        if self._service is None:
            if self._handler is None:
                raise RuntimeError(
                    f"Adapter({self.id!r}): no inference handler — decorate a "
                    f"function with @adapter.on_image() (or on_audio / "
                    f"on_text / on_data)."
                )
            self._service = _FacadeService(self)
        return self._service

    @property
    def adapter_app(self) -> AdapterApp:
        """The :class:`~.adapter_app.AdapterApp` wrapper."""
        if self._app is None:
            shape = _KIND_TO_SHAPE[self._handler.kind] if self._handler else BodyShape.GENERIC
            self._app = AdapterApp(
                service=self.service,
                name=self.id,
                version=self.version,
                vendor=self.vendor or self.id,
                license=self.license or "unspecified",
                model_card_url=self.model_card_url,
                tasks_advertised=list(self.tasks),
                body_shape=shape,
                max_body_bytes=self._max_body_bytes,
                permissions=Permissions(
                    gpu=self.gpu,
                    network_egress=list(self._network_egress),
                    host_filesystem=[],
                    shared_memory_paths=[],
                    host_metadata=False,
                ),
                scheduling=Scheduling(
                    max_inflight=self._max_inflight,
                    preferred_batch_size=1,
                    fair_queuing=FairQueuing.PER_CAMERA,
                ),
                cost=self._cost or Cost(currency="USD"),
                supports_stream=self._stream_fn is not None,
            )
        return self._app

    @property
    def app(self):
        """The ASGI application — point uvicorn at this."""
        return self.adapter_app.fastapi_app

    def run(self, host: str = "0.0.0.0", port: int = 9000) -> None:  # pragma: no cover
        """Serve with uvicorn. Convenience for local runs; in a
        container, point uvicorn at :attr:`app` directly."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)


# ── The compiled service ────────────────────────────────────────────


class _FacadeService(AdapterService):
    """What an :class:`Adapter` compiles to: an ordinary
    :class:`~.service.AdapterService` whose four required methods are
    answered from the declaration."""

    def __init__(self, adapter: Adapter) -> None:
        self._adapter = adapter
        self._state: HealthStatus = HealthStatus.LOADING
        self._error: str | None = None
        self._model: Any = None
        self._loaded_at: float | None = None

    # ── Lifecycle ──────────────────────────────────────────────────

    def load(self) -> None:
        if self._state == HealthStatus.OK:
            return
        loader = self._adapter._load_fn
        if loader is None:
            self._state = HealthStatus.OK
            self._loaded_at = time.time()
            return
        try:
            self._model = loader()
        except Exception as exc:  # noqa: BLE001 — the whole point
            self._state = HealthStatus.ERROR
            self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s failed to load", self._adapter.id)
            return
        self._state = HealthStatus.OK
        self._error = None
        self._loaded_at = time.time()
        logger.info("%s ready", self._adapter.id)

    def is_ready(self) -> bool:
        return self._state == HealthStatus.OK

    def shutdown(self) -> None:  # pragma: no cover — exercised by the app
        if self._adapter._shutdown_fn is not None:
            try:
                self._adapter._shutdown_fn(self._model)
            except Exception:
                logger.exception("%s shutdown hook failed", self._adapter.id)

    # ── Identity ───────────────────────────────────────────────────

    def fingerprint(self) -> str | None:
        return self._adapter.fingerprint

    def model_info(self) -> ModelInfo:
        return self._adapter.model_info()

    def hardware_evaluation(self) -> HardwareEvaluationResponse:
        override = self._adapter._hardware_fn
        if override is not None:
            try:
                verdict = override(self._model)
            except Exception as exc:  # noqa: BLE001
                return self._evaluation(
                    HardwareVerdict.WARN,
                    f"The adapter's hardware check raised: "
                    f"{type(exc).__name__}: {exc}")
            if isinstance(verdict, HardwareEvaluationResponse):
                return verdict
            if isinstance(verdict, tuple) and len(verdict) == 2:
                return self._evaluation(HardwareVerdict(verdict[0]), str(verdict[1]))
            if isinstance(verdict, bool):
                return self._evaluation(
                    HardwareVerdict.OK if verdict else HardwareVerdict.BLOCKED,
                    "The adapter's hardware check "
                    + ("passed." if verdict else "failed."))
            verdict = HardwareVerdict(verdict)
            return self._evaluation(
                verdict,
                f"The adapter's hardware check returned {verdict.value!r}.")

        if self._state == HealthStatus.OK:
            return self._evaluation(
                HardwareVerdict.OK,
                "The model loaded on this host."
                + (" A GPU is declared as required; verify the container sees "
                   "one before relying on throughput." if self._adapter.gpu else ""))
        if self._state == HealthStatus.LOADING:
            return self._evaluation(HardwareVerdict.WARN,
                                    "The model is still loading.")
        return self._evaluation(HardwareVerdict.BLOCKED,
                                f"The model failed to load: {self._error}")

    def _evaluation(self, verdict: HardwareVerdict,
                    reasoning: str) -> HardwareEvaluationResponse:
        # ``reasoning`` is what the operator's hardware dashboard shows,
        # and the contract requires it to be non-empty — an adapter that
        # answers "blocked" with no explanation is unactionable.
        return HardwareEvaluationResponse(
            verdict=verdict,
            reasoning=reasoning or f"Verdict: {verdict.value}.",
            checked_at=datetime.now(timezone.utc),
            details={
                "cpu_count": os.cpu_count() or 0,
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "gpu_declared": self._adapter.gpu,
                "framework": self._adapter.framework,
            },
        )

    # ── Inference ──────────────────────────────────────────────────

    def infer(self, payload: dict[str, Any]) -> InferResponse:
        if self._state != HealthStatus.OK:
            loading = self._state == HealthStatus.LOADING
            raise ServiceError(
                ErrorCategory.MODEL_ERROR,
                code="model_loading" if loading else "model_failed",
                message=(self._error or "The model is still loading."),
                transient=loading,
                http_status=503,
                retry_after_ms=2000 if loading else None,
            )

        handler = self._adapter._handler
        assert handler is not None  # guarded by Adapter.service
        call = InferCall(payload, self._model, self._adapter)
        started = time.monotonic()
        try:
            produced = handler.fn(call)
        except ServiceError:
            # The handler was precise about the failure; respect it.
            raise
        except Overloaded as exc:
            raise ServiceError(
                ErrorCategory.OVERLOADED, code="overloaded", message=str(exc),
                transient=True, http_status=503,
                retry_after_ms=exc.retry_after_ms,
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            # The caller sent something this model cannot use. A 400
            # tells KAI-C not to retry — which is the difference between
            # one bad frame and a retry storm.
            raise ServiceError(
                ErrorCategory.TRANSPORT_ERROR, code="malformed_input",
                message=str(exc) or type(exc).__name__,
                transient=False, http_status=400,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s inference failed", self._adapter.id)
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="inference_failed",
                message=f"{type(exc).__name__}: {exc}",
                transient=False, http_status=500,
            ) from exc

        return self._respond(produced, int((time.monotonic() - started) * 1000))

    def _respond(self, produced: Any, inference_ms: int) -> InferResponse:
        """Normalize whatever the handler returned.

        A list is the §5.1 detection convention, a dict is a result as
        written, and an InferResponse passes through — so the easy case
        stays one line and the precise case is still available."""
        if isinstance(produced, InferResponse):
            return produced
        if produced is None:
            result: dict[str, Any] = {}
        elif isinstance(produced, list):
            result = {"detections": produced}
        elif isinstance(produced, dict):
            result = produced
        else:
            raise ServiceError(
                ErrorCategory.MODEL_ERROR, code="bad_handler_result",
                message=(
                    f"The inference handler returned "
                    f"{type(produced).__name__}; return a list of detections, "
                    f"a result dict, or an InferResponse."
                ),
                transient=False, http_status=500,
            )
        return InferResponse(
            model_name=self._adapter.id,
            model_version=self._adapter.model_version,
            inference_ms=inference_ms,
            result=result,
        )

    async def handle_stream(self, websocket: Any) -> None:  # pragma: no cover
        stream = self._adapter._stream_fn
        if stream is None:
            return await super().handle_stream(websocket)
        return await stream(websocket)


__all__ = ["Adapter", "InferCall", "Overloaded"]
