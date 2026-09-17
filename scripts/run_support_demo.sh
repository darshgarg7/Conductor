#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDUCTOR_PYTHON="${CONDUCTOR_PYTHON:-.venv/bin/python}"
SUPPORT_BOOTSTRAP_DEFAULT=true
for SUPPORT_ARGUMENT in "$@"; do
  case "$SUPPORT_ARGUMENT" in
    --checkpoint|--checkpoint=*|--config|--config=*|--help|-h)
      SUPPORT_BOOTSTRAP_DEFAULT=false ;;
  esac
done
if [[ "$SUPPORT_BOOTSTRAP_DEFAULT" == true &&
      ! -e outputs/checkpoints/tiny-sft/controller.json &&
      ! -e outputs/checkpoints/tiny-sft/checkpoint_pointer.json ]]; then
  printf '%s\n' 'Preparing synthetic development data and trained tiny SFT checkpoint (no model download).'
  "$CONDUCTOR_PYTHON" -m conductor.generate --config configs/generation.yaml
  "$CONDUCTOR_PYTHON" -m conductor.train --config configs/training/tiny_sft.yaml
fi
"$CONDUCTOR_PYTHON" -m conductor.demo --config configs/demos/support.yaml "$@"
