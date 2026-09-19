"""Strict offline records and task-disjoint internal validation partitions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from conductor.schema import ExecutionState


def read_records(path: str | Path, stage: str, *, allowed_splits: tuple[str, ...] = ("train",)) -> list[dict[str, Any]]:
    required = {"state", "decision"} if stage == "sft" else {"state", "chosen", "rejected"}
    records = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not required <= record.keys():
                raise ValueError(f"{path}:{number} missing {required - record.keys()}")
            if record.get("split", "train") not in allowed_splits:
                raise ValueError(f"{path}:{number}: held-out evaluation records must never train the controller")
            ExecutionState(**record["state"])
            records.append(record)
    if not records:
        raise ValueError(f"empty {stage} dataset: {path}; generate real trajectories/preference pairs first")
    return records


def split_by_task(records: list[dict[str, Any]], fraction: float, seed: int
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0,1)")
    def identity(record: dict[str, Any]) -> str:
        return str(record.get("task_id", record["state"]["user_task"]))
    tasks = sorted({identity(record) for record in records},
                   key=lambda task: hashlib.sha256(f"{seed}:{task}".encode()).hexdigest())
    number = min(len(tasks) - 1, max(1, int(len(tasks) * fraction))) if fraction and len(tasks) > 1 else 0
    validation_ids = set(tasks[:number])
    return ([record for record in records if identity(record) not in validation_ids],
            [record for record in records if identity(record) in validation_ids])


def batches(records: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for offset in range(0, len(records), batch_size):
        yield records[offset:offset + batch_size]
