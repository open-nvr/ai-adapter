# Amazon Rekognition adapter (face_identify, off-site).
#
# Build from the repo root:
#   docker build -f adapters/rekognition/Dockerfile -t opennvr/rekognition-adapter:local .
#
# No model in the image: the faces are matched by AWS. Credentials come from
# the standard AWS environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_REGION) or an instance role; never bake them in.

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        fastapi==0.127.0 \
        uvicorn==0.40.0 \
        pydantic==2.12.5 \
        httpx==0.28.1 \
        python-multipart==0.0.20 \
        pillow==12.0.0 \
        boto3==1.40.70

COPY opennvr_adapter_sdk/    ./opennvr_adapter_sdk/
RUN pip install --no-cache-dir ./opennvr_adapter_sdk
COPY adapters/__init__.py     ./adapters/__init__.py
COPY adapters/rekognition/    ./adapters/rekognition/

ENV PYTHONPATH=/app \
    OPENNVR_ADAPTER_TOKEN="" \
    OPENNVR_REKOGNITION_COLLECTION=opennvr-attendance \
    OPENNVR_REKOGNITION_INDEX=/data/index.json

EXPOSE 9019
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import httpx, sys; sys.exit(0 if httpx.get('http://127.0.0.1:9019/health', timeout=2).status_code == 200 else 1)"
CMD ["uvicorn", "adapters.rekognition.main:app", "--host", "0.0.0.0", "--port", "9019"]
