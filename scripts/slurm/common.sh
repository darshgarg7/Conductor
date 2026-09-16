#!/usr/bin/env bash
set -euo pipefail
CONDUCTOR_ROOT="${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$CONDUCTOR_ROOT"
CONDUCTOR_PYTHON="${CONDUCTOR_PYTHON:-$CONDUCTOR_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CONDUCTOR_TORCH_THREADS="$OMP_NUM_THREADS"
if [[ ! -x "$CONDUCTOR_PYTHON" ]]; then
  printf 'Python environment missing: %s\n' "$CONDUCTOR_PYTHON" >&2
  exit 1
fi
