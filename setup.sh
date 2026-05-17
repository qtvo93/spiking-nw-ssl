#!/bin/bash

set -euo pipefail

DATASET_URL="https://huggingface.co/datasets/qtvo/processed_6folds_VLA/resolve/main/swellex-6folds-real-physics.pkl"
DATASET_FILE="swellex-6folds-real-physics.pkl"
TARGET_DIR="$(pwd)"
TARGET_PATH="$TARGET_DIR/$DATASET_FILE"

if [ -f "$TARGET_PATH" ]; then
  echo "Dataset already exists: $TARGET_PATH"
else
  echo "Downloading dataset..."
  curl -L "$DATASET_URL" -o "$TARGET_PATH"
  echo "Saved dataset to: $TARGET_PATH"
fi

echo "Setup complete."
echo "Run:"
echo "  ./run.sh train_model"
