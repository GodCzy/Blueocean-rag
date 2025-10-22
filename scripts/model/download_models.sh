#!/bin/sh

# Resolve repository root relative to this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load shared environment configuration if present.
ENV_FILE="$PROJECT_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  . "$ENV_FILE"
fi

# OCR 模型
huggingface-cli download SWHL/RapidOCR --local-dir "${MODEL_DIR}/SWHL/RapidOCR"
