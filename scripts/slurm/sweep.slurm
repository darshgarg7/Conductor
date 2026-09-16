#!/usr/bin/env bash
#SBATCH --job-name=conductor-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=conductor-sweep-%A_%a.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
CONDUCTOR_MANIFEST="${1:?Pass a manifest of experiment YAML files}"
CONDUCTOR_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CONDUCTOR_CONFIG="$($CONDUCTOR_PYTHON -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text().splitlines()[int(sys.argv[2])])' "$CONDUCTOR_MANIFEST" "$CONDUCTOR_INDEX")"
"$CONDUCTOR_PYTHON" -m "${CONDUCTOR_MODULE:-conductor.train}" --config "$CONDUCTOR_CONFIG"
