#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONDUCTOR_PILOT_PYTHON="${CONDUCTOR_PYTHON:-.venv/bin/python}"
"$CONDUCTOR_PILOT_PYTHON" -m conductor.doctor --device cpu --dtype float32 --probe --output outputs/research/granite-pilot/doctor.json
"$CONDUCTOR_PILOT_PYTHON" -m conductor.generate --config configs/research/granite_generation.yaml
"$CONDUCTOR_PILOT_PYTHON" -m conductor.train --config configs/training/granite_sft.yaml
"$CONDUCTOR_PILOT_PYTHON" -m conductor.train --config configs/training/granite_preference.yaml
"$CONDUCTOR_PILOT_PYTHON" -m conductor.evaluate --config configs/evaluation/granite_pilot.yaml
"$CONDUCTOR_PILOT_PYTHON" -m conductor.benchmark --config configs/inference/granite_pilot.yaml
"$CONDUCTOR_PILOT_PYTHON" -m conductor.analyze --evaluation outputs/research/granite-pilot/evaluation --benchmark outputs/research/granite-pilot/inference --output outputs/research/granite-pilot/analysis
