"""Explicit heldout selection with ID and prompt leakage checks."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from conductor.schema import Task


def normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def heldout_tasks(records: list[Any], splits: tuple[str, ...] = ("eval", "test")) -> tuple[list[Task], dict[str, Any]]:
    tasks = [item if isinstance(item, Task) else Task(**item) for item in records]
    train = [task for task in tasks if task.split == "train"]
    heldout = [task for task in tasks if task.split in splits]
    if not heldout:
        raise ValueError(f"No heldout tasks found in explicit splits {splits}; refusing to evaluate training data")
    train_ids = {task.id for task in train}
    train_texts = {normalized_text(task.user_task) for task in train}
    ids: set[str] = set()
    texts: set[str] = set()
    for task in heldout:
        prompt = normalized_text(task.user_task)
        if task.id in train_ids or prompt in train_texts:
            raise ValueError(f"Train/heldout leakage detected for task {task.id}")
        if task.id in ids or prompt in texts:
            raise ValueError(f"Duplicate heldout task ID or prompt: {task.id}")
        ids.add(task.id)
        texts.add(prompt)
    return heldout, {"train_count": len(train), "heldout_count": len(heldout), "splits": list(splits),
                     "train_overlap_ids": 0, "train_overlap_texts": 0,
                     "heldout_task_ids": [task.id for task in heldout],
                     "tasks": [asdict(task) for task in heldout]}
