#!/bin/sh
# Entrypoint script for AI Adapters container
# Downloads models on startup, then starts the application

set -e

echo "🚀 Starting AI Adapters container..."

# Download models if they don't exist
python3 /app/download_models.py

# Start the application
echo "🔧 Starting uvicorn server..."
exec uvicorn adapter.main:app --host 0.0.0.0 --port 9100
