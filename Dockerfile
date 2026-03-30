# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
# We include libraries required for OpenCV and other image processing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    python3-dev \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# ARG allows users to override this at build time (e.g. --build-arg USE_GPU=true)
# We default to "false" (CPU-only) to save space (removes ~4GB download)
ARG USE_GPU=false

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
RUN if [ "$USE_GPU" = "false" ]; then \
        echo "Installing CPU-only PyTorch to save space..." && \
        uv pip install --system --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy model download script and entrypoint
COPY download_models.py /app/download_models.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# SECURITY: Remove any .env files and .git directories that may have slipped through
RUN find /app -name ".env" -type f -delete && \
    find /app -name ".env.*" -type f -delete && \
    find /app -name "*.env" -type f -delete && \
    find /app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /app -name "env.example" -type f -delete && \
    echo "✓ Cleaned .env files and .git directories from AI image"

# SECURITY: Run as non-root user
RUN useradd -m -u 1000 aiuser && chown -R aiuser:aiuser /app
USER aiuser

# Expose the port (AI Adapters runs on 9100)
EXPOSE 9100

# Use entrypoint script to download models and start application
CMD ["/app/docker-entrypoint.sh"]
