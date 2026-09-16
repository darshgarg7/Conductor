"""Atomic immutable artifact directories and movable checkpoint pointers."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [directory for directory in path.rglob("*") if directory.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(path)


def resolve_checkpoint(path: str | Path) -> Path:
    directory = Path(path)
    pointer = directory / "checkpoint_pointer.json"
    if pointer.exists():
        target = Path(json.loads(pointer.read_text())["path"])
        directory = target if target.is_absolute() else directory / target
    return directory


@contextmanager
def atomic_directory(path: str | Path) -> Iterator[Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"immutable checkpoint directory already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield temporary
        # Rename publishes the entire controller and optimizer state together.
        _fsync_tree(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
