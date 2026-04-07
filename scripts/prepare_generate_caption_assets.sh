#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEESR_DIR="$ROOT_DIR/thirdparty/SeeSR"
MODEL_DIR="$SEESR_DIR/preset/models"
DAPE_URL="https://huggingface.co/alexnasa/SEESR/resolve/main/DAPE.pth"
RAM_URL="https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth"

mkdir -p "$ROOT_DIR/thirdparty"

if [[ ! -d "$SEESR_DIR/.git" ]]; then
  echo "Cloning SeeSR into $SEESR_DIR"
  git clone --depth 1 https://github.com/cswry/SeeSR.git "$SEESR_DIR"
else
  echo "SeeSR already present at $SEESR_DIR"
fi

mkdir -p "$MODEL_DIR"

if [[ ! -f "$MODEL_DIR/DAPE.pth" ]]; then
  echo "Downloading DAPE checkpoint"
  curl -L --fail "$DAPE_URL" -o "$MODEL_DIR/DAPE.pth"
else
  echo "DAPE checkpoint already present"
fi

cat <<EOF

Assets prepared:
  SeeSR source: $SEESR_DIR
  DAPE model:   $MODEL_DIR/DAPE.pth

Manual step still required:
  Download RAM checkpoint to:
    $MODEL_DIR/ram_swin_large_14m.pth

Suggested command:
  curl -L --fail "$RAM_URL" -o "$MODEL_DIR/ram_swin_large_14m.pth"

Note:
  The RAM checkpoint is large. Make sure you have enough disk space before downloading it.
EOF
