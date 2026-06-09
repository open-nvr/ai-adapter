import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Optional API key protection for adapter endpoints.
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
API_KEY = os.getenv("API_KEY")

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
