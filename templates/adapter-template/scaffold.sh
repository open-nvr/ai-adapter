#!/usr/bin/env bash
# ============================================================
# OpenNVR Adapter Template — scaffold script
# ============================================================
# Generates a new contract v1 adapter from the _template/ skeleton.
#
# Usage:
#     ./templates/adapter-template/scaffold.sh <slug> <port> <body-shape>
#
# Examples:
#     ./templates/adapter-template/scaffold.sh fall-detection 9008 IMAGE
#     ./templates/adapter-template/scaffold.sh audio-events 9009 AUDIO
#     ./templates/adapter-template/scaffold.sh pose-estimation 9010 IMAGE
#
# Run from the ai-adapter repo root. The script refuses to overwrite
# an existing adapter directory; delete it manually first if you're
# starting over.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <slug> <port> <body-shape>" >&2
    echo "       slug: lowercase, hyphenated, e.g. fall-detection" >&2
    echo "       port: 4-digit port number (9008+ recommended; 9001-9007 taken)" >&2
    echo "       body-shape: IMAGE | AUDIO | TEXT | GENERIC" >&2
    exit 64
fi

SLUG="$1"
PORT="$2"
BODY_SHAPE="$3"

# ── Argument validation ─────────────────────────────────────
# Slug: starts with a letter, alphanumeric + hyphens, no trailing
# hyphen. The optional second class makes single-character slugs
# legal (``./scaffold.sh x 9008 IMAGE``) — they're unusual but the
# README's "lowercase, hyphenated, alphanumeric" wording doesn't
# explicitly forbid them.
if [[ ! "$SLUG" =~ ^[a-z]([a-z0-9-]*[a-z0-9])?$ ]]; then
    echo "error: slug must be lowercase, hyphenated, alphanumeric" >&2
    echo "       got: '$SLUG'" >&2
    exit 64
fi

if [[ ! "$PORT" =~ ^[0-9]{4}$ ]]; then
    echo "error: port must be 4 digits, got '$PORT'" >&2
    exit 64
fi

case "$BODY_SHAPE" in
    IMAGE|AUDIO|TEXT|GENERIC) ;;
    *)
        echo "error: body-shape must be IMAGE | AUDIO | TEXT | GENERIC" >&2
        echo "       got: '$BODY_SHAPE'" >&2
        exit 64
        ;;
esac

# ── Derived names ───────────────────────────────────────────
# Python module dirs can't contain hyphens, so the directory name
# uses underscores. The GHCR image name keeps hyphens for Docker /
# CLI convention. Same split fast_plate_ocr/fast-plate-ocr-adapter
# uses.
DIR_NAME="${SLUG//-/_}"
IMAGE_NAME="${SLUG}-adapter"
SERVICE_CLASS="${DIR_NAME^}Service"      # e.g. fall_detection → Fall_detectionService
# Title-case each segment: fall_detection → FallDetection
SERVICE_CLASS=""
IFS='_' read -ra parts <<< "$DIR_NAME"
for part in "${parts[@]}"; do
    SERVICE_CLASS+="${part^}"
done
SERVICE_CLASS+="Service"

# ── Locate the template + target ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/_template"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAPTER_DIR="$REPO_ROOT/adapters/$DIR_NAME"
TEST_FILE="$REPO_ROOT/tests/test_${DIR_NAME}_service.py"

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "error: template skeleton missing at $TEMPLATE_DIR" >&2
    echo "       did you delete templates/adapter-template/_template/ by accident?" >&2
    exit 1
fi

if [[ -e "$ADAPTER_DIR" ]]; then
    echo "error: $ADAPTER_DIR already exists" >&2
    echo "       delete it manually first if you want to regenerate" >&2
    exit 1
fi

if [[ -e "$TEST_FILE" ]]; then
    echo "error: $TEST_FILE already exists" >&2
    exit 1
fi

# ── Copy + substitute ───────────────────────────────────────
echo "Scaffolding adapter '$SLUG'..."
echo "  directory:     adapters/$DIR_NAME/"
echo "  test file:     tests/test_${DIR_NAME}_service.py"
echo "  port:          $PORT"
echo "  body shape:    BodyShape.$BODY_SHAPE"
echo "  GHCR image:    ghcr.io/open-nvr/$IMAGE_NAME"
echo "  service class: $SERVICE_CLASS"

mkdir -p "$ADAPTER_DIR"

# Helper: substitute placeholders into one file.
substitute() {
    local src="$1"
    local dst="$2"
    sed \
        -e "s|__SLUG__|$SLUG|g" \
        -e "s|__DIR_NAME__|$DIR_NAME|g" \
        -e "s|__IMAGE_NAME__|$IMAGE_NAME|g" \
        -e "s|__PORT__|$PORT|g" \
        -e "s|__BODY_SHAPE__|$BODY_SHAPE|g" \
        -e "s|__SERVICE_CLASS__|$SERVICE_CLASS|g" \
        "$src" > "$dst"
}

substitute "$TEMPLATE_DIR/adapter/__init__.py" "$ADAPTER_DIR/__init__.py"
substitute "$TEMPLATE_DIR/adapter/main.py"     "$ADAPTER_DIR/main.py"
substitute "$TEMPLATE_DIR/adapter/service.py"  "$ADAPTER_DIR/service.py"
substitute "$TEMPLATE_DIR/adapter/Dockerfile"  "$ADAPTER_DIR/Dockerfile"
substitute "$TEMPLATE_DIR/adapter/README.md"   "$ADAPTER_DIR/README.md"
substitute "$TEMPLATE_DIR/test_service.py"     "$TEST_FILE"

echo
echo "Done. Next steps:"
echo "  1. Replace the # TODO markers in adapters/$DIR_NAME/service.py with your model wrapper."
echo "  2. Add your ML libraries to adapters/$DIR_NAME/Dockerfile's pip install line."
echo "  3. Fill in the README's 'What it does' / 'Why this model' sections."
echo "  4. Replace the toy assertions in tests/test_${DIR_NAME}_service.py with real ones."
echo "  5. When ready, append the adapter to .github/workflows/publish-images.yml's matrix."
echo
echo "Test it boots locally:"
echo "  OPENNVR_ADAPTER_TOKEN=dev-token \\"
echo "    python -m uvicorn adapters.$DIR_NAME.main:app --port $PORT"
echo "  curl http://localhost:$PORT/health"
echo
echo "Conformance check (once /health is green):"
echo "  python -m conformance http://localhost:$PORT --token \$OPENNVR_ADAPTER_TOKEN"
