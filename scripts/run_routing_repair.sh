#!/usr/bin/env bash
# CPU repair experiment. Output directories are deliberately immutable.
set -euo pipefail
cd "$(dirname "$0")/.."
conductor_python="${CONDUCTOR_PYTHON:-.venv/bin/python}"
"$conductor_python" -m conductor.generate --config configs/research/routing_repair_generation.yaml
"$conductor_python" -m conductor.training.curate --config configs/research/routing_repair_curate.yaml
"$conductor_python" -m conductor.training.probe --config configs/research/routing_repair_probe.yaml
"$conductor_python" -m conductor.train --config configs/research/routing_repair_sft.yaml
"$conductor_python" -m conductor.train --config configs/research/routing_repair_preference.yaml
"$conductor_python" -m conductor.evaluate --config configs/research/routing_repair_reference.yaml
"$conductor_python" -m conductor.evaluate --config configs/research/routing_repair_frozen.yaml
"$conductor_python" -m conductor.evaluate --config configs/research/routing_repair_evaluate.yaml
"$conductor_python" -m conductor.evaluate --config configs/research/routing_repair_dense.yaml
