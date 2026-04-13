#!/bin/sh
# Entrypoint script for AI Adapters container

set -e

echo "🚀 Starting AI Adapters container..."

# Download models using uv
uv run python3 /app/download_models.py

# Start the application exclusively via uv
echo "🔧 Starting uvicorn server..."
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 9100
