#!/bin/sh
# Entrypoint script for AI Adapters container

set -e

echo "🚀 Starting AI Adapters container..."

# Run the baked virtualenv's Python directly. Using `uv run` here would
# re-resolve the project and hit PyPI on every boot (crash-looping on a
# slow network); the deps are already installed into /opt/venv at build
# time, so invoke that interpreter straight.
PY=/opt/venv/bin/python3

# Pre-fetch model weights in the BACKGROUND. download_models.py uses
# urlretrieve with no timeout, so a stalled download could otherwise hang
# the entrypoint forever and fail the healthcheck. Adapters also lazily
# download their weights on first inference, so this is best-effort — the
# server must not wait on it.
( "$PY" /app/download_models.py \
    || echo "⚠ Model pre-download failed — models will be fetched on first inference." >&2 ) &

# Start the application from the baked venv (no network resolution).
echo "🔧 Starting uvicorn server..."
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port 9100
