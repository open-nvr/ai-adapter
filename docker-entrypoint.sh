#!/bin/sh
# Entrypoint script for AI Adapters container

set -e

echo "🚀 Starting AI Adapters container..."

# Download models using uv — non-fatal so transient network errors
# don't crash-loop the container; adapters will download lazily on first use.
if ! uv run python3 /app/download_models.py; then
    echo "⚠ Model pre-download failed — continuing; models will be fetched on first inference." >&2
fi

# Start the application exclusively via uv
echo "🔧 Starting uvicorn server..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 9100
