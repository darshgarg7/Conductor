#!/usr/bin/env bash
#SBATCH --job-name=conductor-dpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=conductor-dpo-%j.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
"$CONDUCTOR_PYTHON" -m conductor.train --config "${1:-configs/training/olmoe_preference.yaml}" --checkpoint "${2:-outputs/checkpoints/conductor-sft}"
