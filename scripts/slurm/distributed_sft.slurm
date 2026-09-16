#!/usr/bin/env bash
#SBATCH --job-name=conductor-ddp-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=conductor-ddp-sft-%j.log
set -euo pipefail
source "${CONDUCTOR_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
CONDUCTOR_DDP_PROCESSES="${CONDUCTOR_GPU_COUNT:-2}"
if [[ ! "$CONDUCTOR_DDP_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CONDUCTOR_GPU_COUNT must be a positive integer\n' >&2
  exit 2
fi
CONDUCTOR_DDP_OUTPUT="outputs/validation/ddp-${SLURM_JOB_ID:-local}"
mkdir -p "$CONDUCTOR_DDP_OUTPUT"
"$CONDUCTOR_PYTHON" - "$CONDUCTOR_DDP_PROCESSES" "${1:-configs/training/olmoe_sft.yaml}" "$CONDUCTOR_DDP_OUTPUT/doctor.json" <<'PY'
import sys
import torch
from conductor.doctor import inspect
from conductor.utils.config import load_config
from conductor.utils.runs import write_json
processes = int(sys.argv[1])
configuration = load_config(sys.argv[2])
model = configuration.get("model", {})
if torch.device(model.get("device", "cpu")).type != "cuda" or model.get("require_cuda") is not True:
    raise SystemExit("Distributed NVIDIA SFT requires model.device=cuda and model.require_cuda=true")
if configuration.get("training", {}).get("stage") != "sft":
    raise SystemExit("distributed_sft.slurm requires training.stage=sft")
if torch.cuda.device_count() < processes:
    raise SystemExit("Requested process count exceeds GPUs visible in this SLURM allocation")
records = [inspect(f"cuda:{index}", model.get("dtype", "float32"), require_cuda=True, probe=True)
           for index in range(processes)]
write_json(sys.argv[3], records)
if any(record["status"] != "compatible" for record in records):
    raise SystemExit("An allocated GPU failed compatibility validation; see doctor.json")
PY
# common.sh exports the full allocation; divide it across the torchrun workers.
export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-8} / CONDUCTOR_DDP_PROCESSES ))"
if (( OMP_NUM_THREADS < 1 )); then export OMP_NUM_THREADS=1; fi
export CONDUCTOR_TORCH_THREADS="$OMP_NUM_THREADS"
"$CONDUCTOR_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$CONDUCTOR_DDP_PROCESSES" --module conductor.train --config "${1:-configs/training/olmoe_sft.yaml}" "${@:2}"
