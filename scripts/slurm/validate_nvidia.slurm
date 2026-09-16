#!/usr/bin/env bash
#SBATCH --job-name=conductor-gpu-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=conductor-gpu-validation-%j.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
export CONDUCTOR_PYTHON
export CONDUCTOR_VALIDATION_OUTPUT="outputs/validation/nvidia-${SLURM_JOB_ID}"
bash scripts/validate_nvidia.sh "${1:-configs/inference/nvidia.yaml}" "${2:-outputs/checkpoints/conductor-preference}"
