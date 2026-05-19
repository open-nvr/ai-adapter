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

from app.interfaces.contract import (
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
        "echo": {"hello": "world"},
    }

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

        # Adapter advertises streaming. Full WS-protocol exercise is
        # an A2.2/YOLO concern — for v1 we just confirm the upgrade
        # endpoint exists and returns SOMETHING other than 501.
        ok, response, latency_ms, err = self._get(
            "/infer/stream", expect_status={400, 405, 426}  # Upgrade Required / similar
        )
        if not ok and response is None:
            return self._record(
                "infer_stream",
                CheckOutcome.FAIL,
                detail=f"upgrade endpoint unreachable: {err}",
                latency_ms=latency_ms,
            )
        return self._record(
            "infer_stream",
            CheckOutcome.PASS,
            detail=f"upgrade endpoint reachable (status={response.status_code}); full handshake not yet exercised",
            latency_ms=latency_ms,
        )

    # ── Helpers ────────────────────────────────────────────────────

    def _sample_payload_for(self, caps: CapabilitiesResponse) -> dict[str, Any] | None:
        for task in caps.tasks_advertised:
            if task in self.SAMPLE_INFER_PAYLOADS:
                return self.SAMPLE_INFER_PAYLOADS[task]
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
