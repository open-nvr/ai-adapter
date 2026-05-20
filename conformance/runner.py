# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ConformanceRunner — exercises an adapter URL against the contract.

Each ``check`` method probes one endpoint, validates the wire shape
against the Pydantic models in ``app/interfaces/contract.py``, and
records a :class:`CheckResult`. ``run_all()`` runs every check in
order and returns the aggregate report.

Result vocabulary:

* ``PASS`` — endpoint conforms.
* ``WARN`` — endpoint works but advisory issue (missing optional field,
  slow response, etc.).
* ``FAIL`` — endpoint violates the contract.
* ``SKIP`` — check cannot run (e.g., streaming check on an adapter
  that declared ``infer_stream.supported = false``).
"""
from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from opennvr_adapter_sdk.contract import (
    CapabilitiesResponse,
    FailureEnvelope,
    HardwareEvaluationResponse,
    HealthResponse,
    InferResponse,
)


def _looks_like_test_client(client: Any) -> bool:
    """True if the client appears to be a Starlette/FastAPI TestClient.

    TestClient subclasses httpx.Client but wraps an ASGI app accessible
    as ``client.app``; a vanilla httpx.Client has no such attribute.
    """
    return hasattr(client, "app") and client.app is not None


class CheckOutcome(str, enum.Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    outcome: CheckOutcome
    detail: str = ""
    latency_ms: int = 0
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConformanceReport:
    base_url: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome == CheckOutcome.PASS)

    @property
    def warned(self) -> int:
        return sum(1 for r in self.results if r.outcome == CheckOutcome.WARN)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome == CheckOutcome.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome == CheckOutcome.SKIP)

    @property
    def is_green(self) -> bool:
        return self.failed == 0


class ConformanceRunner:
    """
    Orchestrates the conformance checks. The client is injected so
    in-process tests can hand in a ``httpx.Client(transport=...)``
    backed by FastAPI's TestClient — no network required.
    """

    # Sample infer payloads keyed by ``tasks_advertised`` element. The
    # runner picks one based on what the adapter declares.
    SAMPLE_INFER_PAYLOADS: dict[str, dict[str, Any]] = {
        "speech_synthesis": {"text": "Conformance check: hello, world."},
        "object_detection": {"confidence_threshold": 0.5},
        # ASR adapters receive audio bytes via the multipart path (see
        # _post_multipart_audio below); this JSON payload is a no-op
        # fallback for adapters that don't accept multipart for some
        # reason. Real test goes through the audio multipart path.
        "audio_transcription": {"task": "audio_transcription"},
        "audio_translation": {"task": "audio_translation"},
        "echo": {"hello": "world"},
    }

    # Sample frames for the WS streaming check, keyed by task. A
    # detection adapter gets a small valid JPEG; tasks without a
    # registered sample are reported as WARN. The embedded JPEG below
    # is a 1x1 black image (~630 bytes) — small enough to ship inline
    # and still decode through OpenCV/PIL on the adapter side, so we
    # avoid pulling cv2 into the conformance kit's own deps.
    _SAMPLE_1x1_BLACK_JPEG_B64: str = (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgU"
        "GBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBw"
        "YHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/w"
        "AARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QA"
        "tRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2J"
        "yggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eX"
        "qDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2"
        "uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL"
        "/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvA"
        "VYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dX"
        "Z3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1"
        "dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+f+iiigD/2Q=="
    )
    SAMPLE_STREAM_FRAMES: dict[str, bytes] = {
        "object_detection": __import__("base64").b64decode(_SAMPLE_1x1_BLACK_JPEG_B64),
    }

    # ── Sample audio for ASR adapters ──────────────────────────────
    #
    # A 100ms silent WAV @ 8 kHz mono int16 (~1.6 KB). Whisper's VAD
    # gates out silence so the transcript is typically empty, but the
    # §5.3 result shape still validates. Good enough to confirm the
    # wire path works without bundling a real-speech asset and the
    # licensing baggage that goes with one. Generated at module-load
    # time via stdlib ``wave`` — no extra deps, no large source-tree
    # constants.
    @classmethod
    def _build_sample_silent_wav(cls) -> bytes:
        import io
        import wave as _wave
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        return buf.getvalue()

    # Computed once at class-body evaluation, NOT at every check_infer.
    # ``__class_getitem__``-style lazy attribute would be nicer but a
    # plain method-call-into-dict at module load is fine for the
    # conformance kit's scale.
    SAMPLE_AUDIO_BYTES: dict[str, bytes] = {}  # populated below

    def __init__(
        self,
        base_url: str,
        client: Any = None,
        token: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        ``client`` is duck-typed: anything exposing ``.get(url, headers=..., timeout=...)``
        and ``.post(url, content=..., headers=...)`` works. In production the
        CLI passes an ``httpx.Client``; tests pass FastAPI's
        ``TestClient`` directly so no real network is required.
        """
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        # ``trust_env=False`` so HTTP_PROXY / SOCKS_PROXY env vars do
        # not redirect our probes through some operator-side proxy
        # that's not part of the adapter under test. We're probing a
        # specific URL — go directly.
        self._client = client or httpx.Client(timeout=timeout_seconds, trust_env=False)
        self._token = token
        self._capabilities: CapabilitiesResponse | None = None
        self._report = ConformanceReport(base_url=self.base_url)

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()

    def __enter__(self) -> "ConformanceRunner":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────

    def run_all(self) -> ConformanceReport:
        self.check_base_url()
        self.check_health()
        self.check_capabilities()
        self.check_hardware_evaluation()
        self.check_metrics()
        self.check_infer()
        self.check_infer_stream()
        return self._report

    # ── Individual checks ──────────────────────────────────────────

    def check_base_url(self) -> CheckResult:
        # Empty base_url means the caller wired in a TestClient or other
        # in-process transport — skip the URL hygiene check.
        if not self.base_url:
            return self._record(
                "base_url", CheckOutcome.SKIP, detail="in-process client; URL check skipped"
            )
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https"):
            return self._record(
                "base_url",
                CheckOutcome.FAIL,
                detail=f"Base URL must use http or https; got {parsed.scheme!r}",
            )
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            return self._record(
                "base_url",
                CheckOutcome.WARN,
                detail="Non-loopback HTTP — production adapters should use HTTPS or be on loopback.",
            )
        return self._record("base_url", CheckOutcome.PASS, detail=parsed.scheme + "://" + (parsed.netloc or ""))

    def check_health(self) -> CheckResult:
        # /health is auth-exempt per §3.8; do not send the token here so
        # we exercise the "open" path the same way KAI-C would.
        ok, response, latency_ms, err = self._get("/health", with_auth=False, timeout=1.5)
        if not ok:
            return self._record("health", CheckOutcome.FAIL, detail=err, latency_ms=latency_ms)
        if latency_ms > 1000:
            return self._record(
                "health",
                CheckOutcome.WARN,
                detail=f"Response took {latency_ms}ms; spec says within 1000ms.",
                latency_ms=latency_ms,
            )
        try:
            HealthResponse.model_validate(response.json())
        except (ValidationError, json.JSONDecodeError) as exc:
            return self._record(
                "health",
                CheckOutcome.FAIL,
                detail=f"Body does not validate as HealthResponse: {exc}",
                latency_ms=latency_ms,
            )
        return self._record(
            "health",
            CheckOutcome.PASS,
            detail=f"status={response.json().get('status')}",
            latency_ms=latency_ms,
        )

    def check_capabilities(self) -> CheckResult:
        ok, response, latency_ms, err = self._get("/capabilities")
        if not ok:
            return self._record("capabilities", CheckOutcome.FAIL, detail=err, latency_ms=latency_ms)
        try:
            caps = CapabilitiesResponse.model_validate(response.json())
        except (ValidationError, json.JSONDecodeError) as exc:
            return self._record(
                "capabilities",
                CheckOutcome.FAIL,
                detail=f"Body does not validate as CapabilitiesResponse: {exc}",
                latency_ms=latency_ms,
            )
        self._capabilities = caps
        warn: str | None = None
        if caps.model.fingerprint is None:
            warn = "model.fingerprint is null — tamper detection unavailable."
        outcome = CheckOutcome.WARN if warn else CheckOutcome.PASS
        return self._record(
            "capabilities",
            outcome,
            detail=warn or f"adapter={caps.adapter.name}@{caps.adapter.version}",
            latency_ms=latency_ms,
            evidence={"tasks_advertised": list(caps.tasks_advertised)},
        )

    def check_hardware_evaluation(self) -> CheckResult:
        ok, response, latency_ms, err = self._get("/hardware/evaluation")
        if not ok:
            return self._record("hardware_evaluation", CheckOutcome.FAIL, detail=err, latency_ms=latency_ms)
        try:
            HardwareEvaluationResponse.model_validate(response.json())
        except (ValidationError, json.JSONDecodeError) as exc:
            return self._record(
                "hardware_evaluation",
                CheckOutcome.FAIL,
                detail=f"Body does not validate as HardwareEvaluationResponse: {exc}",
                latency_ms=latency_ms,
            )
        return self._record(
            "hardware_evaluation",
            CheckOutcome.PASS,
            detail=f"verdict={response.json().get('verdict')}",
            latency_ms=latency_ms,
        )

    def check_metrics(self) -> CheckResult:
        ok, response, latency_ms, err = self._get("/metrics")
        if not ok:
            return self._record("metrics", CheckOutcome.FAIL, detail=err, latency_ms=latency_ms)
        body = response.text
        required = (
            "adapter_infer_total",
            "adapter_infer_latency_seconds",
            "adapter_model_loaded",
        )
        missing = [m for m in required if m not in body]
        if missing:
            return self._record(
                "metrics",
                CheckOutcome.FAIL,
                detail=f"Missing required metrics: {missing}",
                latency_ms=latency_ms,
            )
        return self._record(
            "metrics",
            CheckOutcome.PASS,
            detail=f"{len(body.splitlines())} lines of Prometheus exposition",
            latency_ms=latency_ms,
        )

    def check_infer(self) -> CheckResult:
        if self._capabilities is None:
            return self._record("infer", CheckOutcome.SKIP, detail="capabilities check did not pass")
        if not self._capabilities.endpoints.infer.supported:
            return self._record("infer", CheckOutcome.SKIP, detail="adapter declares infer.supported = false")

        # §3.5 — adapters that take image or audio input prefer
        # multipart (real binary bytes); text-only adapters get the
        # JSON path. Pick based on declared modalities so the
        # conformance kit exercises whichever transport the adapter
        # actually expects.
        modalities = self._capabilities.model.modalities_in or []
        image_in = "image" in modalities
        audio_in = "audio" in modalities
        sample_frame = self._sample_stream_frame_for(self._capabilities) if image_in else None
        sample_audio = self._sample_audio_for(self._capabilities) if audio_in else None

        if image_in and sample_frame is not None:
            ok, response, latency_ms, err = self._post_multipart_frame("/infer", sample_frame)
        elif audio_in and sample_audio is not None:
            ok, response, latency_ms, err = self._post_multipart_audio("/infer", sample_audio)
        else:
            payload = self._sample_payload_for(self._capabilities)
            if payload is None:
                return self._record(
                    "infer",
                    CheckOutcome.WARN,
                    detail=(
                        "no sample payload registered for any of "
                        f"tasks_advertised={list(self._capabilities.tasks_advertised)}. "
                        "Add one to ConformanceRunner.SAMPLE_INFER_PAYLOADS to exercise this adapter."
                    ),
                )
            ok, response, latency_ms, err = self._post_json("/infer", payload)

        if not ok:
            return self._record("infer", CheckOutcome.FAIL, detail=err, latency_ms=latency_ms)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            return self._record("infer", CheckOutcome.FAIL, detail=f"non-JSON body: {exc}", latency_ms=latency_ms)

        # Adapter could return success OR a typed failure envelope. Both
        # are wire-conformant.
        try:
            InferResponse.model_validate(data)
        except ValidationError:
            try:
                FailureEnvelope.model_validate(data)
                return self._record(
                    "infer",
                    CheckOutcome.WARN,
                    detail=f"Adapter returned FailureEnvelope code={data.get('error', {}).get('code')}",
                    latency_ms=latency_ms,
                )
            except ValidationError as exc:
                return self._record(
                    "infer",
                    CheckOutcome.FAIL,
                    detail=f"Body validates as neither InferResponse nor FailureEnvelope: {exc}",
                    latency_ms=latency_ms,
                )
        return self._record(
            "infer",
            CheckOutcome.PASS,
            detail=f"inference_ms={data.get('inference_ms')}",
            latency_ms=latency_ms,
        )

    def check_infer_stream(self) -> CheckResult:
        if self._capabilities is None:
            return self._record(
                "infer_stream",
                CheckOutcome.SKIP,
                detail="capabilities check did not pass",
            )
        if not self._capabilities.endpoints.infer_stream.supported:
            # Per §3.6 the adapter MUST refuse the upgrade with HTTP 501
            # when not supported. We probe with a GET and expect 501.
            ok, response, latency_ms, err = self._get("/infer/stream", expect_status={501, 405})
            if not ok:
                return self._record(
                    "infer_stream",
                    CheckOutcome.FAIL,
                    detail=(
                        "infer_stream.supported=false but adapter did not refuse with "
                        f"HTTP 501: {err}"
                    ),
                    latency_ms=latency_ms,
                )
            if response.status_code == 405:
                return self._record(
                    "infer_stream",
                    CheckOutcome.WARN,
                    detail=(
                        "Adapter returned 405 Method Not Allowed instead of 501 — "
                        "harmless but §3.6 prefers 501 + stream_not_supported envelope."
                    ),
                    latency_ms=latency_ms,
                )
            try:
                FailureEnvelope.model_validate(response.json())
            except (ValidationError, json.JSONDecodeError):
                return self._record(
                    "infer_stream",
                    CheckOutcome.WARN,
                    detail="HTTP 501 returned but body is not a FailureEnvelope.",
                    latency_ms=latency_ms,
                )
            return self._record(
                "infer_stream",
                CheckOutcome.PASS,
                detail="HTTP 501 with FailureEnvelope as expected (adapter does not stream).",
                latency_ms=latency_ms,
            )

        # Adapter advertises streaming — exercise the §6 protocol
        # end-to-end: open WS, handshake, send one frame, receive one
        # result message, close cleanly.
        return self._check_stream_handshake()

    def _check_stream_handshake(self) -> CheckResult:
        """Open a real WebSocket, handshake, send a frame, expect a result.

        Probe payloads come from ``SAMPLE_STREAM_FRAMES`` keyed by task.
        Adapters whose first task has no sample frame are reported as
        WARN (we know they advertise streaming but we have nothing to
        send them).
        """
        sample_frame = self._sample_stream_frame_for(self._capabilities)
        if sample_frame is None:
            return self._record(
                "infer_stream",
                CheckOutcome.WARN,
                detail=(
                    "Adapter advertises streaming but no sample frame is "
                    f"registered for tasks_advertised={list(self._capabilities.tasks_advertised)}. "
                    "Add one to ConformanceRunner.SAMPLE_STREAM_FRAMES to exercise it."
                ),
            )

        # TestClient (in-process) has ``websocket_connect``; httpx.Client
        # does not — for httpx we use the ``websockets`` library if
        # available; otherwise skip the full handshake with a WARN.
        if _looks_like_test_client(self._client):
            return self._exercise_stream_via_testclient(sample_frame)
        return self._exercise_stream_via_websockets(sample_frame)

    def _exercise_stream_via_testclient(self, sample_frame: bytes) -> CheckResult:
        started = time.monotonic()
        try:
            headers = {}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            with self._client.websocket_connect("/infer/stream", headers=headers) as ws:
                ws.send_text(json.dumps({
                    "type": "handshake",
                    "client_id": "conformance",
                    "camera_id": "conformance-cam",
                    "frame_transport": "websocket",
                }))
                ack = json.loads(ws.receive_text())
                if ack.get("type") != "handshake_ack":
                    return self._record(
                        "infer_stream",
                        CheckOutcome.FAIL,
                        detail=f"first WS message was not handshake_ack: {ack}",
                        latency_ms=int((time.monotonic() - started) * 1000),
                    )
                ws.send_text(json.dumps({
                    "type": "frame", "seq": 1, "ts_ms": 0,
                    "content_type": "image/jpeg",
                }))
                ws.send_bytes(sample_frame)
                result = json.loads(ws.receive_text())
                ws.send_text(json.dumps({"type": "close", "reason": "conformance done"}))
            return self._validate_stream_result(result, int((time.monotonic() - started) * 1000))
        except Exception as exc:
            return self._record(
                "infer_stream",
                CheckOutcome.FAIL,
                detail=f"WS exercise failed: {type(exc).__name__}: {exc}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )

    def _validate_stream_result(self, result: Any, latency_ms: int) -> CheckResult:
        """Validate the shape of a streaming ``result`` message.

        The §6.3 envelope is: ``type=="result"``, echoed ``seq``, integer
        ``inference_ms``, and a ``result`` body. The body is either a
        §5-shaped success payload OR a §7 FailureEnvelope. Both must
        validate or we don't really know the adapter conforms — only
        that it sent us something JSON-shaped.
        """
        if not isinstance(result, dict):
            return self._record(
                "infer_stream", CheckOutcome.FAIL,
                detail=f"WS result was not a JSON object: {type(result).__name__}",
                latency_ms=latency_ms,
            )
        if result.get("type") != "result":
            return self._record(
                "infer_stream", CheckOutcome.FAIL,
                detail=f"expected type='result', got {result.get('type')!r}",
                latency_ms=latency_ms,
            )
        if result.get("seq") != 1:
            return self._record(
                "infer_stream", CheckOutcome.FAIL,
                detail=f"expected seq=1, got {result.get('seq')!r}",
                latency_ms=latency_ms,
            )
        body = result.get("result")
        if not isinstance(body, dict):
            return self._record(
                "infer_stream", CheckOutcome.FAIL,
                detail=f"result.result must be a JSON object, got {type(body).__name__}",
                latency_ms=latency_ms,
            )
        # Body must be either a success payload (any dict) or a typed
        # FailureEnvelope. Plain success bodies are guidance-only per
        # §5 — we accept anything dict-shaped. Error bodies must match
        # §7 exactly so audit pipelines parsing them work uniformly.
        if body.get("status") == "error":
            try:
                FailureEnvelope.model_validate(body)
            except ValidationError as exc:
                return self._record(
                    "infer_stream", CheckOutcome.FAIL,
                    detail=f"error result body is not a valid FailureEnvelope: {exc}",
                    latency_ms=latency_ms,
                )
            return self._record(
                "infer_stream", CheckOutcome.WARN,
                detail=(
                    f"Stream returned typed error envelope "
                    f"(code={body['error'].get('code')}); roundtrip protocol OK."
                ),
                latency_ms=latency_ms,
            )
        return self._record(
            "infer_stream", CheckOutcome.PASS,
            detail="Full §6 roundtrip OK (handshake → frame → result → close).",
            latency_ms=latency_ms,
        )

    def _exercise_stream_via_websockets(self, sample_frame: bytes) -> CheckResult:
        """Real-network WebSocket roundtrip using the ``websockets`` lib.

        Optional dep — if not installed, we WARN instead of FAIL so the
        rest of the conformance run still gives signal. A2.3 will land
        ``websockets`` as a hard dep on the conformance kit.
        """
        try:
            import asyncio
            import websockets  # type: ignore[import-not-found]
        except ImportError:
            return self._record(
                "infer_stream",
                CheckOutcome.WARN,
                detail=(
                    "Adapter advertises streaming. Install 'websockets' "
                    "(pip install websockets) for full conformance."
                ),
            )

        ws_url = self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/infer/stream"
        extra_headers = {}
        if self._token:
            extra_headers["Authorization"] = f"Bearer {self._token}"

        async def _exercise() -> tuple[bool, str]:
            try:
                async with websockets.connect(ws_url, extra_headers=extra_headers) as ws:
                    await ws.send(json.dumps({
                        "type": "handshake",
                        "client_id": "conformance",
                        "camera_id": "conformance-cam",
                        "frame_transport": "websocket",
                    }))
                    ack = json.loads(await ws.recv())
                    if ack.get("type") != "handshake_ack":
                        return False, f"first WS message was not handshake_ack: {ack}"
                    await ws.send(json.dumps({
                        "type": "frame", "seq": 1, "ts_ms": 0,
                        "content_type": "image/jpeg",
                    }))
                    await ws.send(sample_frame)
                    result = json.loads(await ws.recv())
                    if result.get("type") != "result" or result.get("seq") != 1:
                        return False, f"expected result(seq=1), got {result}"
                    await ws.send(json.dumps({"type": "close", "reason": "conformance done"}))
                return True, "Full §6 roundtrip OK."
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"

        started = time.monotonic()
        ok, detail = asyncio.run(_exercise())
        latency_ms = int((time.monotonic() - started) * 1000)
        return self._record(
            "infer_stream",
            CheckOutcome.PASS if ok else CheckOutcome.FAIL,
            detail=detail,
            latency_ms=latency_ms,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _sample_payload_for(self, caps: CapabilitiesResponse) -> dict[str, Any] | None:
        for task in caps.tasks_advertised:
            if task in self.SAMPLE_INFER_PAYLOADS:
                return self.SAMPLE_INFER_PAYLOADS[task]
        return None

    def _sample_stream_frame_for(self, caps: CapabilitiesResponse) -> bytes | None:
        for task in caps.tasks_advertised:
            if task in self.SAMPLE_STREAM_FRAMES:
                return self.SAMPLE_STREAM_FRAMES[task]
        return None

    def _sample_audio_for(self, caps: CapabilitiesResponse) -> bytes | None:
        for task in caps.tasks_advertised:
            if task in self.SAMPLE_AUDIO_BYTES:
                return self.SAMPLE_AUDIO_BYTES[task]
        return None

    def _headers(self, *, with_auth: bool) -> dict[str, str]:
        h = {}
        if with_auth and self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _client_get(self, url: str, headers: dict[str, str], timeout: float | None) -> Any:
        """Call ``client.get`` tolerating TestClient.

        Per-call ``timeout=`` was deprecated on Starlette's TestClient
        (which inherits from httpx.Client, so an isinstance check alone
        won't distinguish them). We sniff for ``client.app`` — TestClient
        wraps an ASGI app there; a real httpx.Client does not.
        """
        if timeout is not None and not _looks_like_test_client(self._client):
            return self._client.get(url, headers=headers, timeout=timeout)
        return self._client.get(url, headers=headers)

    def _client_post(self, url: str, headers: dict[str, str], content: bytes) -> Any:
        return self._client.post(url, content=content, headers=headers)

    def _get(
        self,
        path: str,
        *,
        with_auth: bool = True,
        expect_status: set[int] | None = None,
        timeout: float | None = None,
    ) -> tuple[bool, Any, int, str]:
        url = self.base_url + path
        start = time.monotonic()
        try:
            response = self._client_get(url, self._headers(with_auth=with_auth), timeout)
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return False, None, latency_ms, f"GET {path}: {type(exc).__name__}: {exc}"
        latency_ms = int((time.monotonic() - start) * 1000)
        allowed = expect_status or {200}
        if response.status_code not in allowed:
            return (
                False,
                response,
                latency_ms,
                f"GET {path}: expected status in {sorted(allowed)} got {response.status_code}",
            )
        return True, response, latency_ms, ""

    def _post_multipart_frame(
        self,
        path: str,
        frame_bytes: bytes,
    ) -> tuple[bool, Any, int, str]:
        """POST a frame as multipart/form-data. Works for both httpx.Client
        and Starlette's TestClient (both accept the same ``files=`` kwarg)."""
        return self._post_multipart_file(
            path, file_field="frame", filename="frame.jpg",
            content=frame_bytes, content_type="image/jpeg",
        )

    def _post_multipart_audio(
        self,
        path: str,
        audio_bytes: bytes,
    ) -> tuple[bool, Any, int, str]:
        """POST audio bytes as multipart/form-data with field name
        ``audio`` (the ASR-adapter convention; YOLOv8 uses ``frame``)."""
        return self._post_multipart_file(
            path, file_field="audio", filename="audio.wav",
            content=audio_bytes, content_type="audio/wav",
        )

    def _post_multipart_file(
        self,
        path: str,
        *,
        file_field: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> tuple[bool, Any, int, str]:
        url = self.base_url + path
        headers = self._headers(with_auth=True)
        start = time.monotonic()
        try:
            response = self._client.post(
                url,
                files={file_field: (filename, content, content_type)},
                headers=headers,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return False, None, latency_ms, f"POST {path}: {type(exc).__name__}: {exc}"
        latency_ms = int((time.monotonic() - start) * 1000)
        if response.status_code != 200:
            return (
                False,
                response,
                latency_ms,
                f"POST {path}: expected 200 got {response.status_code}: {response.text[:200]}",
            )
        return True, response, latency_ms, ""

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> tuple[bool, Any, int, str]:
        url = self.base_url + path
        headers = self._headers(with_auth=True)
        headers["Content-Type"] = "application/json"
        start = time.monotonic()
        try:
            response = self._client_post(url, headers, json.dumps(payload).encode("utf-8"))
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return False, None, latency_ms, f"POST {path}: {type(exc).__name__}: {exc}"
        latency_ms = int((time.monotonic() - start) * 1000)
        if response.status_code != 200:
            return (
                False,
                response,
                latency_ms,
                f"POST {path}: expected 200 got {response.status_code}: {response.text[:200]}",
            )
        return True, response, latency_ms, ""

    def _record(
        self,
        name: str,
        outcome: CheckOutcome,
        *,
        detail: str = "",
        latency_ms: int = 0,
        evidence: dict[str, Any] | None = None,
    ) -> CheckResult:
        result = CheckResult(
            name=name,
            outcome=outcome,
            detail=detail,
            latency_ms=latency_ms,
            evidence=evidence or {},
        )
        self._report.results.append(result)
        return result


# Populate the audio-sample dict once at module load. Has to happen
# AFTER the class is fully defined because we call the classmethod.
_silent_wav = ConformanceRunner._build_sample_silent_wav()
ConformanceRunner.SAMPLE_AUDIO_BYTES = {
    "audio_transcription": _silent_wav,
    "audio_translation": _silent_wav,
}
del _silent_wav
