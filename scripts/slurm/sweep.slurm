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
CONDUCTOR_SWEEP_SELECTION="$("$CONDUCTOR_PYTHON" - "$CONDUCTOR_MANIFEST" "$CONDUCTOR_INDEX" "${CONDUCTOR_MODULE:-}" <<'PY'
from pathlib import Path
import sys
from conductor.utils.config import load_config
paths = Path(sys.argv[1]).read_text().splitlines()
try:
    index = int(sys.argv[2])
except ValueError:
    raise SystemExit("SLURM_ARRAY_TASK_ID must be a nonnegative manifest index")
if not 0 <= index < len(paths):
    raise SystemExit("SLURM_ARRAY_TASK_ID is outside the sweep manifest")
path = Path(paths[index]).resolve()
configuration = load_config(path)
module = configuration.get("sweep", {}).get("module")
if module not in {"conductor.train", "conductor.evaluate", "conductor.benchmark"}:
    raise SystemExit("Sweep configuration lacks a supported declared module; regenerate with make_sweep.py --module")
if sys.argv[3] and sys.argv[3] != module:
    raise SystemExit("CONDUCTOR_MODULE conflicts with the configuration's declared sweep module")
print(module)
print(path)
PY
)"
{
  IFS= read -r CONDUCTOR_SWEEP_MODULE
  IFS= read -r CONDUCTOR_CONFIG
} <<< "$CONDUCTOR_SWEEP_SELECTION"
"$CONDUCTOR_PYTHON" -m "$CONDUCTOR_SWEEP_MODULE" --config "$CONDUCTOR_CONFIG"
