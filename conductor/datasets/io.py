"""Streaming JSONL writing and strict record loading."""
from __future__ import annotations
import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(record)
    return records
