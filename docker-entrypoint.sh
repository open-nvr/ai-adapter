#!/bin/sh
# Start the configured adapter server.

set -e

HOST="${ADAPTER_HOST:-0.0.0.0}"
PORT="${PORT:-9100}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Use the venv baked into the image (/opt/venv) directly. Do NOT use
# `uv run`, which re-resolves and re-downloads the project's deps from
# PyPI on every boot (failing in any egress-restricted deploy).
if ! /opt/venv/bin/python3 /app/download_models.py; then
    echo "Model pre-download failed; models may load on first inference." >&2
fi

exec /opt/venv/bin/python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level "$(printf '%s' "$LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"
