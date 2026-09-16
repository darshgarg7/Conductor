"""Canonical complete execution state, with bounded serialization memoization."""
from __future__ import annotations

import json
from collections import OrderedDict
from conductor.schema import ExecutionState


def serialize_state(state: ExecutionState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class StateSerializer:
    def __init__(self, cache_size: int = 1024) -> None:
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        self._cache: OrderedDict[str, str] = OrderedDict()

    def serialize(self, state: ExecutionState) -> str:
        # State is mutable: canonical content, never object identity, is the key.
        # Key creation is included in benchmarks; memoization is not assumed faster.
        key = serialize_state(state)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        if self.cache_size:
            self._cache[key] = key
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return key

    def clear(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0

