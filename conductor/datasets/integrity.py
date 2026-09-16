"""Crash-recoverable, checksum-verified JSONL journals and stable shard assignment."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Iterator


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def fsync_directory(directory: str | Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def durable_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SingleWriterLock:
    """A persistent lock inode held across inventory, all journals and manifests."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f"another generation job owns {self.path}") from None
        except Exception:
            self.handle.close()
            raise

    def __enter__(self) -> SingleWriterLock:
        return self

    def __exit__(self, *_: Any) -> None:
        self.handle.close()  # Never unlink a lock inode while contenders may hold it.


def shard_for(task_id: str, shards: int) -> int:
    if shards < 1:
        raise ValueError("jobshards must be positive")
    return int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % shards


def task_seed(seed: int, task_id: str, policy: str) -> int:
    return int(hashlib.sha256(f"{seed}:{task_id}:{policy}".encode()).hexdigest()[:8], 16)


def record_checksum(record: dict[str, Any]) -> str:
    clean = copy.deepcopy(record)
    clean.get("metadata", {}).pop("record_checksum_sha256", None)
    return digest(clean)


def seal(record: dict[str, Any], record_id: str) -> dict[str, Any]:
    result = copy.deepcopy(record)
    result.setdefault("metadata", {})["record_id"] = record_id
    result["metadata"]["record_checksum_sha256"] = record_checksum(result)
    return result


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: JSON record must be an object")
                yield value


class Journal:
    """Single-writer append journal; a torn final line is discarded on restart.

    Every acknowledged append is fsynced. A held advisory lock prevents two
    workers from accidentally writing the same shard. Interior corruption fails
    closed, and duplicate IDs with different content are rejected.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f"another writer owns {self.path}")
        self.ids: dict[str, str] = {}
        self.offsets: dict[str, int] = {}
        self.recovered_bytes = 0
        self.handle.seek(0)
        try:
            while True:
                offset = self.handle.tell()
                line = self.handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    self.recovered_bytes = len(line)
                    self.handle.truncate(offset)
                    self.handle.flush()
                    os.fsync(self.handle.fileno())
                    break
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
                    raise ValueError(f"invalid journal record at offset {offset}: {self.path}")
                metadata = value["metadata"]
                key, checksum = metadata.get("record_id"), metadata.get("record_checksum_sha256")
                if not isinstance(key, str) or not key or checksum != record_checksum(value):
                    raise ValueError(f"invalid journal integrity at offset {offset}: {self.path}")
                if key in self.ids:
                    raise ValueError(f"duplicate journal ID {key}: {self.path}")
                self.ids[key] = checksum
                self.offsets[key] = offset
            self.handle.seek(0, os.SEEK_END)
            fsync_directory(self.path.parent)
        except Exception:
            self.handle.close()
            raise

    def append(self, record: dict[str, Any], record_id: str) -> bool:
        if record_id in self.ids:
            if record_checksum(seal(record, record_id)) != self.ids[record_id]:
                raise ValueError(f"duplicate record ID with different content: {record_id}")
            return False
        value = seal(record, record_id)
        offset = self.handle.tell()
        self.handle.write((canonical(value) + "\n").encode())
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.ids[record_id] = value["metadata"]["record_checksum_sha256"]
        self.offsets[record_id] = offset
        return True

    def get(self, record_id: str) -> dict[str, Any]:
        end = self.handle.tell()
        try:
            self.handle.seek(self.offsets[record_id])
            value = json.loads(self.handle.readline())
            if record_checksum(value) != self.ids[record_id]:
                raise ValueError(f"journal record changed after validation: {record_id}")
        finally:
            self.handle.seek(end)
        return value

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
