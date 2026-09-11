# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the defects the pre-merge review of the adapter SDK
turned up.

Each one was a SILENT failure — the adapter reported healthy, the
conformance run went green, the catalog answered 200 — which is exactly
the class of bug a test suite has to hold down, because nothing else
will notice.
"""
from __future__ import annotations

import base64
import json
import re

import pytest
from fastapi.testclient import TestClient

from opennvr_adapter_sdk.contract import HealthStatus
from opennvr_adapter_sdk.facade import Adapter

#: Any decodable body — the parser rejects an empty one before
#: the handler is ever called.
_FRAME_B64 = base64.b64encode(b"not-really-a-jpeg").decode()


def _adapter(**kwargs) -> Adapter:
    return Adapter(
        "review-fixture", version="1.0.0", vendor="OpenNVR",
        license="Apache-2.0", tasks=("object_detection",), **kwargs)


# ── /health tells the truth about a failed load ────────────────────


def test_health_reports_error_when_the_model_fails_to_load() -> None:
    """A dead adapter used to answer `loading` forever: Docker's
    healthcheck passed, the conformance run recorded PASS, and the
    operator's only clue was that nothing ever inferred."""
    adapter = _adapter()

    @adapter.load()
    def load():
        raise RuntimeError("weights are missing")

    @adapter.on_image()
    def handle(call):
        return []

    with TestClient(adapter.app) as client:
        body = client.get("/health").json()

    assert body["status"] == HealthStatus.ERROR.value


def test_health_reports_ok_once_loaded() -> None:
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return []

    with TestClient(adapter.app) as client:
        assert client.get("/health").json()["status"] == HealthStatus.OK.value


def test_conformance_fails_a_dead_adapter() -> None:
    """The whole point of #1: `validate` must not go green on it."""
    from opennvr_adapter_sdk.conformance.runner import CheckOutcome, ConformanceRunner

    adapter = _adapter()

    @adapter.load()
    def load():
        raise RuntimeError("weights are missing")

    @adapter.on_image()
    def handle(call):
        return []

    with TestClient(adapter.app, base_url="http://adapter.test") as client:
        report = ConformanceRunner("", client=client).run_all()

    health = next(r for r in report.results if r.name == "health")
    assert health.outcome == CheckOutcome.FAIL
    assert not report.is_green


def test_in_process_client_skips_the_url_hygiene_check() -> None:
    """`validate` printed "Non-loopback HTTP" about a host it never
    bound, on every single run."""
    from opennvr_adapter_sdk.conformance.runner import CheckOutcome, ConformanceRunner

    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return []

    with TestClient(adapter.app, base_url="http://adapter.test") as client:
        report = ConformanceRunner("", client=client).run_all()

    base = next(r for r in report.results if r.name == "base_url")
    assert base.outcome == CheckOutcome.SKIP


# ── The fingerprint is not re-hashed on every poll ─────────────────


def test_fingerprint_is_cached_between_calls(tmp_path) -> None:
    """KAI-C polls /capabilities every 60s and /health more often than
    that; re-reading a multi-gigabyte weights file each time blew the
    contract's 1000 ms budget."""
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"x" * 4096)
    adapter = _adapter(weights=weights)

    first = adapter.fingerprint
    assert adapter._fp_cache is not None
    # Corrupt the cached digest: a second read must come from the cache.
    key, _ = adapter._fp_cache
    adapter._fp_cache = (key, "sha256:sentinel")
    assert adapter.fingerprint == "sha256:sentinel"
    assert first.startswith("sha256:")


def test_fingerprint_still_changes_when_the_weights_change(tmp_path) -> None:
    """Caching must not blind drift detection — that is the one thing
    the fingerprint exists for."""
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"x" * 4096)
    adapter = _adapter(weights=weights)
    before = adapter.fingerprint

    weights.write_bytes(b"y" * 8192)
    assert adapter.fingerprint != before


def test_fingerprint_is_never_null_without_weights() -> None:
    assert _adapter().fingerprint.startswith("sha256:")


# ── AsyncAPI resolves ──────────────────────────────────────────────


def test_asyncapi_has_no_dangling_refs() -> None:
    """`$defs` were dropped, so FrameTransport and StreamCloseCode were
    referenced by three messages and defined nowhere — every generator
    pointed at the document failed on it."""
    from opennvr_adapter_sdk.openapi import adapter_asyncapi

    doc = adapter_asyncapi(
        name="x", version="1.0.0", tasks=("object_detection",),
        supports_stream=True)
    text = json.dumps(doc)
    referenced = set(re.findall(r'"#/components/schemas/([A-Za-z0-9_]+)"', text))
    defined = set(doc["components"]["schemas"])
    assert referenced - defined == set()
    assert {"FrameTransport", "StreamCloseCode"} <= defined


def test_served_asyncapi_also_resolves() -> None:
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return []

    with TestClient(adapter.app) as client:
        doc = client.get("/asyncapi.json").json()

    text = json.dumps(doc)
    referenced = set(re.findall(r'"#/components/schemas/([A-Za-z0-9_]+)"', text))
    assert referenced - set(doc["components"]["schemas"]) == set()


# ── Registration after the app is built is loud ────────────────────


@pytest.mark.parametrize("register", [
    lambda a: a.on_stream()(lambda ws: None),
    lambda a: a.load()(lambda: None),
    lambda a: a.on_shutdown()(lambda model: None),
    lambda a: a.check_hardware()(lambda model: True),
])
def test_registering_after_the_app_is_built_raises(register) -> None:
    """`supports_stream`, the body shape and the advertised tasks all
    freeze when the ASGI app is built. The scaffold ends with
    `app = adapter.app`, so anything below it silently did nothing."""
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return []

    _ = adapter.app

    with pytest.raises(RuntimeError, match="after the app was built"):
        register(adapter)


def test_stream_registered_before_the_app_is_advertised() -> None:
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return []

    @adapter.on_stream()
    def stream(ws):
        return None

    with TestClient(adapter.app) as client:
        caps = client.get("/capabilities").json()

    assert caps["endpoints"]["infer_stream"]["supported"] is True


# ── The body shape reaches the tooling ─────────────────────────────


@pytest.mark.parametrize("register,expected", [
    ("on_image", "image"),
    ("on_audio", "audio"),
    ("on_text", "text"),
    ("on_data", "generic"),
])
def test_app_state_carries_the_body_shape(register, expected) -> None:
    """`opennvr-adapter dev` sent `frame_b64` to every adapter, so an
    audio or text adapter was reported broken by its own dev runner."""
    adapter = _adapter()
    getattr(adapter, register)()(lambda call: [])
    assert adapter.app.state.opennvr_body_shape.value == expected


# ── A bbox in pixels is not silently clamped ───────────────────────


def test_pixel_coordinates_warn(caplog) -> None:
    import opennvr_adapter_sdk.facade as facade
    from opennvr_adapter_sdk.facade import InferCall

    facade._WARNED_BBOX = False
    with caplog.at_level("WARNING"):
        item = InferCall.detection("person", 0.9, 100, 200, 50, 60)
    assert item["bbox"] == {"x": 1.0, "y": 1.0, "w": 1.0, "h": 1.0}
    assert "NORMALIZED" in caplog.text


def test_normalized_coordinates_do_not_warn(caplog) -> None:
    import opennvr_adapter_sdk.facade as facade
    from opennvr_adapter_sdk.facade import InferCall

    facade._WARNED_BBOX = False
    with caplog.at_level("WARNING"):
        InferCall.detection("person", 0.9, 0.1, 0.2, 0.3, 0.4)
    assert caplog.text == ""


# ── A handler bug is a 500, not a 400 blamed on the caller ─────────


def test_type_error_in_the_handler_is_a_model_error() -> None:
    """A TypeError is almost always the handler's own bug. Reporting it
    as `transport_error` told KAI-C not to retry AND told the author
    the caller was at fault, with no traceback anywhere."""
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        return 1 + None  # type: ignore[operator]

    with TestClient(adapter.app) as client:
        response = client.post(
            "/infer", json={"camera_id": "cam-1", "frame_b64": _FRAME_B64},
            headers={"Authorization": "Bearer test"})

    assert response.status_code == 500
    assert response.json()["error"]["category"] == "model_error"


def test_value_error_in_the_handler_is_still_a_transport_error() -> None:
    adapter = _adapter()

    @adapter.on_image()
    def handle(call):
        raise ValueError("confidence_threshold must be a number")

    with TestClient(adapter.app) as client:
        response = client.post(
            "/infer", json={"camera_id": "cam-1", "frame_b64": _FRAME_B64},
            headers={"Authorization": "Bearer test"})

    assert response.status_code == 400
    assert response.json()["error"]["category"] == "transport_error"


# ── The scaffolded project is installable ──────────────────────────


def test_scaffolded_console_script_points_at_a_callable(tmp_path) -> None:
    """`__ADAPTER_MODULE__ = "module:app"` named a FastAPI instance, so
    `pip install -e .` produced an entry point that raised on the first
    invocation."""
    from opennvr_adapter_sdk.scaffold import generate

    target = generate("fall-detection", tmp_path)

    pyproject = (target / "pyproject.toml").read_text()
    module = next(
        p.stem for p in target.glob("*.py") if not p.name.startswith("test"))
    assert f'{module} = "{module}:main"' in pyproject
    assert "def main() -> None:" in (target / f"{module}.py").read_text()
