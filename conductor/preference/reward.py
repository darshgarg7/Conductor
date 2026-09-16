"""A configurable, dimensionless success/cost utility used to label preferences.

Default scales: 1000 tokens, 1 wall-clock second, 12 activations, 20 directed
communication edges. Coefficients/scales are design choices, never results.
Latencies are measured; timing-sensitive labels should be repeated at scale.
"""
from __future__ import annotations

from typing import Any

from conductor.schema import Trajectory

DEFAULT_WEIGHTS = {"token_weight": 0.05, "latency_weight": 0.02, "agent_call_weight": 0.04,
                   "communication_weight": 0.01, "token_scale": 1000.0, "latency_scale": 1.0,
                   "agent_call_scale": 12.0, "communication_scale": 20.0}


def reward(trajectory: Trajectory, weights: dict[str, Any] | None = None) -> float:
    values = {**DEFAULT_WEIGHTS, **(weights or {})}
    for alias, canonical in {"lambda_tokens": "token_weight", "lambda_latency": "latency_weight",
                             "lambda_agent_calls": "agent_call_weight", "lambda_communication": "communication_weight"}.items():
        if alias in (weights or {}):
            values[canonical] = weights[alias]
    for key in ("token_scale", "latency_scale", "agent_call_scale", "communication_scale"):
        if float(values[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("token_weight", "latency_weight", "agent_call_weight", "communication_weight"):
        if float(values[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    tokens = sum(step.controller_tokens + sum(int(output["tokens"]) for output in step.agent_outputs)
                 for step in trajectory.steps)
    calls = sum(len(step.agent_outputs) for step in trajectory.steps)
    return (float(trajectory.task_success)
            - values["token_weight"] * tokens / values["token_scale"]
            - values["latency_weight"] * trajectory.wall_clock_latency / values["latency_scale"]
            - values["agent_call_weight"] * calls / values["agent_call_scale"]
            - values["communication_weight"] * len(trajectory.communication_graph) / values["communication_scale"])
