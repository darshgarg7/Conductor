#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDUCTOR_PYTHON="${CONDUCTOR_PYTHON:-.venv/bin/python}"
"$CONDUCTOR_PYTHON" -m conductor.generate --config configs/generation.yaml
"$CONDUCTOR_PYTHON" -m conductor.train --config configs/training/tiny_sft.yaml
"$CONDUCTOR_PYTHON" -m conductor.train --config configs/training/tiny_preference.yaml --checkpoint outputs/checkpoints/tiny-sft
"$CONDUCTOR_PYTHON" -m conductor.evaluate --config configs/evaluation/heldout.yaml
"$CONDUCTOR_PYTHON" -m conductor.benchmark --config configs/inference/development.yaml
"$CONDUCTOR_PYTHON" scripts/benchmark_optimizations.py --config configs/inference/optimization_study.yaml
"$CONDUCTOR_PYTHON" -m conductor.analyze --evaluation outputs/evaluation/dev --benchmark outputs/benchmarks/dev --output outputs/reports/dev
