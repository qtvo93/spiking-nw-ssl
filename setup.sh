#!/bin/bash

set -euo pipefail

DATASET_URL="https://huggingface.co/datasets/qtvo/processed_6folds_VLA/resolve/main/swellex-6folds-real-physics.pkl?download=1"
DATASET_FILE="swellex-6folds-real-physics.pkl"
TARGET_DIR="$(pwd)"
TARGET_PATH="$TARGET_DIR/$DATASET_FILE"
MIN_BYTES=1048576

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv..."
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Syncing project dependencies with uv..."
export UV_LINK_MODE=copy
uv sync

if [ -f "$TARGET_PATH" ]; then
  echo "Dataset already exists: $TARGET_PATH"
else
  echo "Downloading dataset..."
  curl -fL "$DATASET_URL" -o "$TARGET_PATH"
  actual_size=$(wc -c < "$TARGET_PATH")
  if [ "$actual_size" -lt "$MIN_BYTES" ]; then
    echo "Download failed or truncated (size: ${actual_size} bytes)."
    echo "Please retry or check network access to Hugging Face."
    exit 1
  fi
  echo "Saved dataset to: $TARGET_PATH (${actual_size} bytes)"
fi

echo "Setup complete."
echo "Run:"
echo "  ./run.sh train_model"
