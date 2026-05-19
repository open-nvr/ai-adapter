# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Auth + correlation_id middleware for the YOLOv8 contract service.

Implements the §3.8 wire spec — same shape as
``adapters/piper/auth.py``. See that module for the design notes;
this module is an intentional copy.

A2.3 will extract this (plus the matching auth module from Piper and
any future adapters) into ``opennvr-adapter-sdk``. Until then we
copy + paste to keep each adapter self-contained as a deployable
unit. Changes must be applied symmetrically across both modules until
the SDK lands.
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


def _expected_token() -> str | None:
    token = os.environ.get(_ENV_TOKEN_VAR, "").strip()
    return token or None


def _is_within_grace_window(started_at: float) -> bool:
    return (time.monotonic() - started_at) < REGISTRATION_GRACE_SECONDS


class AuthAndCorrelationMiddleware(BaseHTTPMiddleware):
    """See ``adapters/piper/auth.py`` for the design notes."""

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
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or uuid.uuid4().hex
        request.state.correlation_id = correlation_id

        path = request.url.path
        if self._auth_required_for(path):
            auth_error = self._check_auth(request)
            if auth_error is not None:
                # Audit-log the rejection. We log the code (auth_missing
                # / auth_invalid) but never the supplied token — that
                # could leak credentials into logs. Path + remote IP
                # give an operator enough to spot brute-force probes.
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


# WebSocket auth helper. The middleware above handles HTTP requests
# but Starlette's BaseHTTPMiddleware does NOT intercept WebSocket
# upgrades, so /infer/stream must check the bearer token itself.
def websocket_auth_failure(
    expected_token: str | None,
    auth_header: str | None,
) -> str | None:
    """
    Returns a short failure reason on auth rejection, or None on success.

    WebSocket auth is mandatory regardless of grace window — there's no
    "registration probe" pattern for WS, only real-time inference
    streams that always come from KAI-C with a token. Adapters in dev
    mode (no expected_token set) allow all WS upgrades, same as the HTTP
    middleware does.
    """
    if expected_token is None:
        return None  # dev mode
    header = (auth_header or "").strip()
    scheme, _, raw_token = header.partition(" ")
    if scheme.lower() != "bearer" or not raw_token:
        return "auth_missing"
    if not hmac.compare_digest(raw_token.strip(), expected_token):
        return "auth_invalid"
    return None
