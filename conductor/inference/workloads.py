"""Replay real controller states without leaking private task labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from conductor.datasets.io import read_jsonl
from conductor.evaluation.provenance import canonical_hash, file_sha256
from conductor.schema import ExecutionState


def replay_states(path: str | Path, limit: int | None = None) -> tuple[list[ExecutionState], list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    states, metadata = [], []
    for trajectory_index, item in enumerate(records):
        for step_index, step in enumerate(item.get("steps", [])):
            state = ExecutionState(**step["state"])
            states.append(state)
            metadata.append({"request_id": f"replay-{trajectory_index}-{step_index}",
                             "task_id": item.get("task", {}).get("id"), "task_type": state.task_type,
                             "step": step_index, "state_sha256": canonical_hash(state.to_dict())})
            if limit is not None and len(states) >= limit:
                break
        if limit is not None and len(states) >= limit:
            break
    if not states:
        raise ValueError(f"No trajectory execution states to replay: {path}")
    return states, metadata, {"kind": "trajectory_replay", "source": str(path), "source_sha256": file_sha256(path),
                              "state_count": len(states), "task_types": sorted({state.task_type for state in states})}
