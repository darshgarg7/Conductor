"""Validated external task sources; task labels never enter execution state."""
from __future__ import annotations
from pathlib import Path
import unicodedata
from typing import Iterator
from conductor.datasets.integrity import digest, iter_records
from conductor.schema import Task


def external_tasks(path: str | Path) -> Iterator[Task]:
    ids: set[str] = set()
    prompts: set[str] = set()
    for value in iter_records(path):
        for field in ("id", "user_task", "task_type", "split", "expected_answer"):
            if not isinstance(value.get(field), str):
                raise ValueError(f"external task {field} must be an explicit string")
        if not isinstance(value.get("metadata", {}), dict):
            raise ValueError("external task metadata must be an object")
        task = Task(**value)
        if not task.id or not task.user_task.strip() or not task.task_type or task.split not in {"train", "eval", "test"}:
            raise ValueError("external tasks require id, public text, category and explicit train/eval/test split")
        if "expected_answer" not in value:
            raise ValueError("external tasks require private expected_answer for the configured exact grader")
        normalized = " ".join(unicodedata.normalize("NFC", task.user_task).casefold().split())
        key = digest(normalized)
        if task.id in ids or key in prompts:
            raise ValueError(f"duplicate external task ID/text (including train/heldout leakage): {task.id}")
        ids.add(task.id)
        prompts.add(key)
        yield task
