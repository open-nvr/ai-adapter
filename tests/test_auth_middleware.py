# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Regression tests for AuthAndCorrelationMiddleware's fail-closed
registration grace window (auth.py) plus the WebSocket auth helper
matrix.

The §3.8 grace window lets an adapter answer unauthenticated
/capabilities and /hardware/evaluation probes for the first 5 minutes
after boot so KAI-C can register it before a token has been provisioned.
The security invariant is that the window CLOSES: once
REGISTRATION_GRACE_SECONDS has elapsed those two endpoints must require
the bearer token like every other route. A window that never closes is
a silent, permanent auth bypass. We prove closure by constructing the
middleware with ``started_at`` pinned far in the past.

WebSocket upgrades are not intercepted by Starlette's BaseHTTPMiddleware,
so /infer/stream calls ``websocket_auth_failure()`` directly. We table-
test its documented branches against the exact source return values.
"""
from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from opennvr_adapter_sdk.auth import (
    REGISTRATION_GRACE_SECONDS,
    AuthAndCorrelationMiddleware,
    websocket_auth_failure,
)


# ── Grace-window fail-closed ────────────────────────────────────────


def _build_app_with_started_at(started_at: float) -> FastAPI:
    """Minimal contract-shaped app guarded by the real middleware.

    We mount AuthAndCorrelationMiddleware directly and inject
    ``started_at`` (mirrors how AdapterApp adds it, but lets us control
    the boot time). The route bodies are stubs — only the middleware's
    auth gating is under test here.
    """
    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/capabilities")
    def capabilities() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/hardware/evaluation")
    def hardware_evaluation() -> JSONResponse:
        return JSONResponse({"ok": True})

    # add_middleware forwards **options to the middleware constructor,
    # so started_at reaches AuthAndCorrelationMiddleware.__init__.
    app.add_middleware(AuthAndCorrelationMiddleware, started_at=started_at)
    return app


@pytest.fixture
def _auth_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Auth enabled: the middleware reads OPENNVR_ADAPTER_TOKEN when its
    stack is built (first request), so setting it here is sufficient."""
    token = "test-token"
    monkeypatch.setenv("OPENNVR_ADAPTER_TOKEN", token)
    return token


def test_grace_window_closes_after_expiry(_auth_env):
    """Fail-closed invariant: with the middleware booted >5min ago, the
    grace-window endpoints require auth (401) while /health stays open."""
    started_at = time.monotonic() - REGISTRATION_GRACE_SECONDS - 100
    client = TestClient(_build_app_with_started_at(started_at))

    assert client.get("/capabilities").status_code == 401
    assert client.get("/hardware/evaluation").status_code == 401
    # /health is ALWAYS open, grace window or not — must not have closed.
    assert client.get("/health").status_code == 200


def test_grace_window_open_while_fresh(_auth_env):
    """Inside the window (booted ~now), the same two endpoints answer
    unauthenticated probes — proving the window, not a static allowlist,
    is what gates them."""
    client = TestClient(_build_app_with_started_at(time.monotonic()))

    assert client.get("/capabilities").status_code == 200
    assert client.get("/hardware/evaluation").status_code == 200
    assert client.get("/health").status_code == 200


def test_closed_grace_window_still_accepts_the_token(_auth_env):
    """After the window closes the endpoints aren't dead — a correct
    bearer token still gets through (so closure gates *unauthenticated*
    access, not all access)."""
    started_at = time.monotonic() - REGISTRATION_GRACE_SECONDS - 100
    client = TestClient(_build_app_with_started_at(started_at))
    headers = {"Authorization": f"Bearer {_auth_env}"}

    assert client.get("/capabilities", headers=headers).status_code == 200
    assert client.get("/hardware/evaluation", headers=headers).status_code == 200


# ── websocket_auth_failure() branch matrix ──────────────────────────
#
# Exact return values from auth.py:
#   expected is None (dev mode)          -> None
#   missing / empty header               -> "auth_missing"
#   non-bearer scheme                    -> "auth_missing"
#   bearer scheme but empty token        -> "auth_missing"
#   bearer + wrong token                 -> "auth_invalid"
#   bearer + correct token (any case)    -> None


@pytest.mark.parametrize(
    "expected, auth_header, want",
    [
        # dev mode: no expected token configured -> never rejects.
        (None, None, None),
        (None, "Bearer anything", None),
        # missing / malformed Authorization header.
        ("secret", None, "auth_missing"),
        ("secret", "", "auth_missing"),
        ("secret", "Bearer", "auth_missing"),  # scheme only, no token
        ("secret", "Basic secret", "auth_missing"),  # wrong scheme
        # present token but wrong.
        ("secret", "Bearer wrong", "auth_invalid"),
        # correct token — scheme is case-insensitive per RFC 7235.
        ("secret", "Bearer secret", None),
        ("secret", "bearer secret", None),
        ("secret", "BEARER secret", None),
    ],
)
def test_websocket_auth_failure_matrix(expected, auth_header, want):
    assert websocket_auth_failure(expected, auth_header) == want
