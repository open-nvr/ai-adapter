import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional

# Configured via environment variables
# If REQUIRE_AUTH is true, an API key must be provided in exactly one of these ways:
# 1. As an HTTP header: X-API-Key
# 2. Or if configured differently but we'll stick to X-API-Key
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
API_KEY = os.getenv("API_KEY")  # No default — must be explicitly set

# Fail fast at startup: if auth is required but no key configured, refuse to boot.
# This prevents silent deployments where REQUIRE_AUTH=true but API_KEY was forgotten,
# which would leave the server either locked out or protected by a known default key.
if REQUIRE_AUTH and not API_KEY:
    raise RuntimeError(
        "REQUIRE_AUTH is enabled but API_KEY environment variable is not set. "
        "Set API_KEY to a strong secret before starting the server."
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)) -> Optional[str]:
    if not REQUIRE_AUTH:
        return None  # Auth is disabled, let anyone through

    if api_key_header == API_KEY:
        return api_key_header
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
        headers={"WWW-Authenticate": "ApiKey"},
    )
