#!/usr/bin/env bash
#SBATCH --job-name=conductor-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=conductor-data-%j.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
"$CONDUCTOR_PYTHON" -m conductor.generate --config "${1:-configs/generation.yaml}"
