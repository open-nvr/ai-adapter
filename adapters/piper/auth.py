# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Auth + correlation_id middleware for the Piper contract service.

Implements the §3.8 wire spec:

* ``Authorization: Bearer <token>`` required on all endpoints except
  ``/health``. Token comes from the ``OPENNVR_ADAPTER_TOKEN`` env.
* During a 5-minute registration grace window after process start,
  ``/capabilities`` and ``/hardware/evaluation`` accept unauthenticated
  probes so KAI-C can discover the adapter before issuing a token.
* ``X-Correlation-Id`` is read on every request and echoed on every
  response. If absent, the service mints one (KAI-C SHOULD always send,
  but adapters MUST tolerate ad-hoc clients).
* In dev (no ``OPENNVR_ADAPTER_TOKEN`` set), auth is bypassed and a
  warning is logged once at startup. This matches the §3.7
  minimum-viable-adapter pattern; production must set the token.
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

# Endpoints exempt from auth even after the registration window closes.
# ``/health`` MUST always be open per §3.8 (KAI-C needs to probe liveness
# before re-issuing a token).
ALWAYS_OPEN_PATHS: frozenset[str] = frozenset({"/health"})

# Endpoints that accept unauthenticated probes during the 5-minute
# registration grace window after process start.
GRACE_WINDOW_PATHS: frozenset[str] = frozenset(
    {"/capabilities", "/hardware/evaluation"}
)

REGISTRATION_GRACE_SECONDS: int = 5 * 60

CORRELATION_ID_HEADER: str = "X-Correlation-Id"

_ENV_TOKEN_VAR: str = "OPENNVR_ADAPTER_TOKEN"


def _expected_token() -> str | None:
    """Read the expected bearer token from env, or None in dev mode."""
    token = os.environ.get(_ENV_TOKEN_VAR, "").strip()
    return token or None


def _is_within_grace_window(started_at: float) -> bool:
    return (time.monotonic() - started_at) < REGISTRATION_GRACE_SECONDS


class AuthAndCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Single middleware that handles both auth and correlation_id plumbing.

    Kept as one class so ordering is unambiguous: correlation_id is
    available to handlers even when auth rejects the request (so the
    audit log line for the rejection has the same id as the caller's
    log line).
    """

    def __init__(self, app, started_at: float | None = None) -> None:
        super().__init__(app)
        self._started_at: float = started_at if started_at is not None else time.monotonic()
        self._expected: str | None = _expected_token()
        if self._expected is None:
            logger.warning(
                "OPENNVR_ADAPTER_TOKEN not set — adapter running in dev mode "
                "with auth disabled. Do not use this configuration in production."
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Extract or mint correlation_id. Stash on request.state so
        # handlers can attach it to audit/log events. Echo on every
        # response — success or failure.
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or uuid.uuid4().hex
        request.state.correlation_id = correlation_id

        # 2. Auth check. Skip for always-open paths; skip for grace-window
        # paths during the window; otherwise require the token.
        path = request.url.path
        auth_required = self._auth_required_for(path)
        if auth_required:
            auth_error = self._check_auth(request)
            if auth_error is not None:
                # Build the FailureEnvelope ourselves rather than raising
                # HTTPException — we need to set the X-Correlation-Id
                # header on the rejection.
                from fastapi.responses import JSONResponse

                response = JSONResponse(status_code=auth_error[0], content=auth_error[1])
                response.headers[CORRELATION_ID_HEADER] = correlation_id
                return response

        response: Response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response

    def _auth_required_for(self, path: str) -> bool:
        if self._expected is None:
            # Dev mode — auth disabled everywhere.
            return False
        if path in ALWAYS_OPEN_PATHS:
            return False
        if path in GRACE_WINDOW_PATHS and _is_within_grace_window(self._started_at):
            return False
        return True

    def _check_auth(self, request: Request) -> tuple[int, dict] | None:
        """Returns (status_code, body) on failure, None on success."""
        header = request.headers.get("Authorization", "").strip()
        # RFC 7235 §2.1: auth-scheme is case-insensitive ("Bearer" / "bearer" / "BEARER").
        scheme, _, raw_token = header.partition(" ")
        if scheme.lower() != "bearer" or not raw_token:
            return (
                status.HTTP_401_UNAUTHORIZED,
                {
                    "status": "error",
                    "error": {
                        "category": "permission_denied",
                        "code": "auth_missing",
                        "message": "Authorization: Bearer <token> required",
                        "transient": False,
                        "details": {},
                    },
                },
            )
        token = raw_token.strip()
        # Constant-time comparison so the response latency does not leak
        # how many leading characters of a brute-force token match.
        # ``compare_digest`` requires both sides to be the same type;
        # ``self._expected`` is set in __init__ above (we'd be in dev mode
        # otherwise and ``_auth_required_for`` would return False).
        assert self._expected is not None
        if not hmac.compare_digest(token, self._expected):
            return (
                status.HTTP_401_UNAUTHORIZED,
                {
                    "status": "error",
                    "error": {
                        "category": "permission_denied",
                        "code": "auth_invalid",
                        "message": "Invalid bearer token",
                        "transient": False,
                        "details": {},
                    },
                },
            )
        return None
