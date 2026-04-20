# Stage 1: Build Environment
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies (the heavy stuff we don't want in runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Allow overriding GPU flag
ARG USE_GPU=false

# Copy project manifest
COPY pyproject.toml .

# Create virtual environment and install natively with uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN if [ "$USE_GPU" = "true" ]; then \
        echo "Installing all adapters with GPU PyTorch..." && \
        uv sync --no-dev --extra all --extra gpu; \
    else \
        echo "Installing all adapters with CPU-only PyTorch..." && \
        uv sync --no-dev --extra all --extra cpu; \
    fi

# Copy application source code into builder stage
COPY . .

# Stage 2: Clean Runtime Environment
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install ONLY runtime dependencies (OpenCV requires these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Re-copy uv for compatibility with entrypoints, even though dependencies are already installed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# SECURITY: Create non-root user BEFORE copying files
RUN useradd -m -u 1000 aiuser

# Copy pre-downloaded environment and models from builder with native ownership
COPY --from=builder --chown=aiuser:aiuser /opt/venv /opt/venv
COPY --from=builder --chown=aiuser:aiuser /app /app

RUN chown aiuser:aiuser /app && \
    chmod +x /app/docker-entrypoint.sh && \
    find /app -name ".env" -type f -delete && \
    find /app -name ".env.*" -type f -delete && \
    find /app -name "*.env" -type f -delete && \
    find /app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

USER aiuser

EXPOSE 9100

# Add HEALTHCHECK so orchestrators know if the async loop freezes
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9100/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
