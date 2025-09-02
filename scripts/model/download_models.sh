#!/bin/sh

# Resolve repository root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/src/.env"

# OCR 模型
huggingface-cli download SWHL/RapidOCR --local-dir "${MODEL_DIR}/SWHL/RapidOCR"
