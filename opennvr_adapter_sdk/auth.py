# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Auth + correlation_id middleware (SDK consolidated).

Replaces the per-adapter ``auth.py`` files that A2.1 / A2.2 / A2.3-prep
shipped. Same §3.8 wire spec — bearer token from
``OPENNVR_ADAPTER_TOKEN``, 5-minute registration grace window for
``/capabilities`` and ``/hardware/evaluation``, constant-time HMAC
compare, case-insensitive Bearer scheme per RFC 7235, dev-mode bypass
when the env var is unset.
"""
from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from typing import Callable

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

ALWAYS_OPEN_PATHS: frozenset[str] = frozenset({"/health"})
GRACE_WINDOW_PATHS: frozenset[str] = frozenset(
    {"/capabilities", "/hardware/evaluation"}
)
REGISTRATION_GRACE_SECONDS: int = 5 * 60
CORRELATION_ID_HEADER: str = "X-Correlation-Id"
_ENV_TOKEN_VAR: str = "OPENNVR_ADAPTER_TOKEN"


def expected_token() -> str | None:
    """Read the expected bearer token from env, or None in dev mode."""
    token = os.environ.get(_ENV_TOKEN_VAR, "").strip()
    return token or None


def _is_within_grace_window(started_at: float) -> bool:
    return (time.monotonic() - started_at) < REGISTRATION_GRACE_SECONDS


class AuthAndCorrelationMiddleware(BaseHTTPMiddleware):
    """See ``open-nvr/docs/AI_ADAPTER_CONTRACT.md`` §3.8 for the wire spec.

    Single middleware that handles both auth and correlation_id so
    ordering is unambiguous: correlation_id is available to handlers
    even when auth rejects, so audit log lines for the rejection
    carry the same id as the caller's log line.
    """

    def __init__(self, app, started_at: float | None = None) -> None:
        super().__init__(app)
        self._started_at: float = started_at if started_at is not None else time.monotonic()
        self._expected: str | None = expected_token()
        if self._expected is None:
            logger.warning(
                "OPENNVR_ADAPTER_TOKEN not set — adapter running in dev mode "
                "with auth disabled. Do not use this configuration in production."
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or uuid.uuid4().hex
        request.state.correlation_id = correlation_id

        path = request.url.path
        if self._auth_required_for(path):
            auth_error = self._check_auth(request)
            if auth_error is not None:
                logger.info(
                    "auth rejected path=%s code=%s correlation_id=%s remote=%s",
                    path,
                    auth_error[1].get("error", {}).get("code", "?"),
                    correlation_id,
                    request.client.host if request.client else "?",
                )
                from fastapi.responses import JSONResponse

                response = JSONResponse(status_code=auth_error[0], content=auth_error[1])
                response.headers[CORRELATION_ID_HEADER] = correlation_id
                return response

        response: Response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response

    def _auth_required_for(self, path: str) -> bool:
        if self._expected is None:
            return False
        if path in ALWAYS_OPEN_PATHS:
            return False
        if path in GRACE_WINDOW_PATHS and _is_within_grace_window(self._started_at):
            return False
        return True

    def _check_auth(self, request: Request) -> tuple[int, dict] | None:
        header = request.headers.get("Authorization", "").strip()
        # RFC 7235 §2.1: auth-scheme is case-insensitive.
        scheme, _, raw_token = header.partition(" ")
        if scheme.lower() != "bearer" or not raw_token:
            return (
                status.HTTP_401_UNAUTHORIZED,
                _auth_envelope("auth_missing", "Authorization: Bearer <token> required"),
            )
        token = raw_token.strip()
        assert self._expected is not None
        # Constant-time compare so response latency doesn't leak how
        # many leading characters of a brute-force token match.
        if not hmac.compare_digest(token, self._expected):
            return (
                status.HTTP_401_UNAUTHORIZED,
                _auth_envelope("auth_invalid", "Invalid bearer token"),
            )
        return None


def _auth_envelope(code: str, message: str) -> dict:
    return {
        "status": "error",
        "error": {
            "category": "permission_denied",
            "code": code,
            "message": message,
            "transient": False,
            "details": {},
        },
    }


def websocket_auth_failure(
    expected: str | None,
    auth_header: str | None,
) -> str | None:
    """WebSocket auth check — Starlette's BaseHTTPMiddleware doesn't
    intercept WS upgrades, so /infer/stream calls this directly.

    Returns a short failure reason on rejection, or None on success.
    WebSocket auth is mandatory regardless of grace window — there's
    no "registration probe" for WS, only real-time inference streams
    that always come from KAI-C with a token.
    """
    if expected is None:
        return None  # dev mode
    header = (auth_header or "").strip()
    scheme, _, raw_token = header.partition(" ")
    if scheme.lower() != "bearer" or not raw_token:
        return "auth_missing"
    if not hmac.compare_digest(raw_token.strip(), expected):
        return "auth_invalid"
    return None
