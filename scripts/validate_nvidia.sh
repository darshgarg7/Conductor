#!/usr/bin/env bash
# Fail closed without real CUDA. Run on an NVIDIA host, never substitute CPU metrics.
set -euo pipefail
CONDUCTOR_VALIDATION_PYTHON="${CONDUCTOR_PYTHON:-.venv/bin/python}"
CONDUCTOR_VALIDATION_OUTPUT="${CONDUCTOR_VALIDATION_OUTPUT:-outputs/validation/nvidia}"
if [[ $# -ne 0 && $# -ne 2 ]]; then
  printf 'Usage: %s [BENCHMARK_CONFIG CHECKPOINT]\n' "$0" >&2
  exit 2
fi
if [[ ! -x "$CONDUCTOR_VALIDATION_PYTHON" ]]; then
  printf 'Python environment missing: %s\n' "$CONDUCTOR_VALIDATION_PYTHON" >&2
  exit 1
fi
CONDUCTOR_VALIDATION_DEVICE="${CONDUCTOR_VALIDATION_DEVICE:-cuda:0}"
CONDUCTOR_VALIDATION_DTYPE="${CONDUCTOR_VALIDATION_DTYPE:-bfloat16}"
if [[ $# -eq 2 ]]; then
  # A GPU doctor must never authorize a CPU-configured model benchmark.
  CONDUCTOR_VALIDATION_MODEL_INFO="$("$CONDUCTOR_VALIDATION_PYTHON" - "$1" "$2" <<'PY'
import sys
import torch
from conductor.controller.artifacts import resolve_checkpoint
from conductor.utils.config import load_config
from conductor.utils.hardware import dtype_name
model = load_config(sys.argv[1]).get("model", {})
device = torch.device(model.get("device", "cpu"))
if device.type != "cuda" or model.get("require_cuda") is not True:
    raise SystemExit("NVIDIA validation requires benchmark model.device=cuda and model.require_cuda=true")
resolve_checkpoint(sys.argv[2])
print(str(device), dtype_name(model.get("dtype", "float32")))
PY
)"
  read -r CONDUCTOR_VALIDATION_DEVICE CONDUCTOR_VALIDATION_DTYPE <<< "$CONDUCTOR_VALIDATION_MODEL_INFO"
fi
mkdir -p "$CONDUCTOR_VALIDATION_OUTPUT"
"$CONDUCTOR_VALIDATION_PYTHON" -m conductor.doctor --device "$CONDUCTOR_VALIDATION_DEVICE" --dtype "$CONDUCTOR_VALIDATION_DTYPE" --require-cuda --probe --output "$CONDUCTOR_VALIDATION_OUTPUT/doctor.json"
"$CONDUCTOR_VALIDATION_PYTHON" -m pip freeze > "$CONDUCTOR_VALIDATION_OUTPUT/dependencies.txt"
git rev-parse HEAD > "$CONDUCTOR_VALIDATION_OUTPUT/git_commit.txt"
git status --porcelain > "$CONDUCTOR_VALIDATION_OUTPUT/git_status.txt"
"$CONDUCTOR_VALIDATION_PYTHON" -m pytest -m cuda -q --junitxml="$CONDUCTOR_VALIDATION_OUTPUT/cuda-tests.xml"
"$CONDUCTOR_VALIDATION_PYTHON" - "$CONDUCTOR_VALIDATION_OUTPUT/cuda-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).findall(".//testcase")
if not cases or any(case.find("skipped") is not None for case in cases):
    raise SystemExit("GPU release gate requires executed CUDA tests without skips; install all test dependencies")
PY
if [[ $# -eq 0 ]]; then
  printf 'CUDA tests completed. Supply BENCHMARK_CONFIG CHECKPOINT to validate actual controller inference.\n'
  exit 0
fi
"$CONDUCTOR_VALIDATION_PYTHON" -m conductor.benchmark --config "$1" --checkpoint "$2"
