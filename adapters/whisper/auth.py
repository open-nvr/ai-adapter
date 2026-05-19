# Copyright (c) 2026 OpenNVR
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Auth + correlation_id middleware for the Whisper contract service.

Same shape as ``adapters/yolov8/auth.py`` and ``adapters/piper/auth.py``.
A2.3 (the next milestone) extracts this into ``opennvr-adapter-sdk``
so every adapter shares one implementation. Until then, changes must
be applied symmetrically across all three modules.

See ``adapters/piper/auth.py`` for design notes on the §3.8 wire spec.
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
    """See ``adapters/piper/auth.py`` for design notes."""

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
