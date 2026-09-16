#!/usr/bin/env bash
#SBATCH --job-name=conductor-infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=conductor-infer-%j.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
"$CONDUCTOR_PYTHON" -m conductor.benchmark --config "${1:-configs/inference/olmoe.yaml}"
