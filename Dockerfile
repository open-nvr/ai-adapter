# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# Base image ONLY loads the ~15MB web framework
RUN uv pip install --system --no-cache-dir -r requirements.txt

# ==============================================================================
# DYNAMIC DEPENDENCY RESOLUTION
# ==============================================================================
# Users pass --build-arg ADAPTER_REQ="vision/requirements-yolo.txt"
# If passed, we install those specific heavy ML dependencies (PyTorch etc).
# If omitted, the Docker image remains under 100MB and serves Lean adapters!
ARG ADAPTER_REQ=""
COPY ./app/adapters ./app/adapters
RUN if [ -n "$ADAPTER_REQ" ] && [ -f "./app/adapters/$ADAPTER_REQ" ]; then \
        echo "Installing specific adapter requirements: $ADAPTER_REQ"; \
        uv pip install --system --no-cache-dir -r ./app/adapters/$ADAPTER_REQ; \
    else \
        echo "No optional heavy ML framework dependencies requested. Building Lean core."; \
    fi

# Copy source code
COPY . .

# SECURITY: Remove environment config leakage
RUN find /app -name ".env" -type f -delete && \
    find /app -name ".env.*" -type f -delete && \
    find /app -name "*.env" -type f -delete && \
    find /app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "env.example" -type f -delete

# SECURITY: Run as non-root user
RUN useradd -m -u 1000 aiuser && chown -R aiuser:aiuser /app
USER aiuser

# Expose the API Port
EXPOSE 9100

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9100"]
