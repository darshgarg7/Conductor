"""YAML configs with explicit relative includes, resolved from the including file."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    source = Path(path).resolve()
    if source in _seen:
        raise ValueError(f"cyclic configuration include: {source}")
    with source.open() as handle:
        own = yaml.safe_load(handle) or {}
    includes = own.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    result: dict[str, Any] = {}
    for include in includes:
        result = merge(result, load_config(source.parent / include, _seen | {source}))
    return merge(result, own)

