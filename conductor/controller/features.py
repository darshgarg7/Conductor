"""Deterministic whole-state features for the cheap, randomly initialized MoE.

This is a debugging representation, not a pretrained language representation.
All fields are hashed; the first dimensions expose execution/budget indicators.
Expected answers and task grader metadata never enter ExecutionState.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

import torch

from conductor.schema import AGENT_NAMES, ExecutionState


def serialize_state(state: ExecutionState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def token_count(text: str) -> int:
    """Explicit proxy for tiny non-tokenizer backend; never passed off as LM tokens."""
    return len(re.findall(r"\w+|[^\w\s]", text))


class StateFeatures:
    def __init__(self, dimension: int = 256, cache_size: int = 0) -> None:
        if dimension < 64:
            raise ValueError("feature_dim must be >=64")
        self.dimension = dimension
        self.encode_text = lru_cache(maxsize=cache_size)(self._encode) if cache_size else self._encode

    def _encode(self, text: str) -> torch.Tensor:
        state = json.loads(text)
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        vector[0] = min(state["current_step"] / 8, 2)
        vector[1] = min(state["remaining_budget"].get("agent_calls", 0) / 12, 2)
        vector[2] = min(state["remaining_budget"].get("tokens", 0) / 8192, 2)
        vector[3] = min(len(state["previous_agent_outputs"]) / 12, 2)
        vector[4] = min(len(state["tool_results"]) / 8, 2)
        vector[5] = min(len(state["previous_routing_decisions"]) / 8, 2)
        vector[6] = float(bool(state["previous_agent_outputs"]))
        vector[7] = 1
        for index, agent in enumerate(AGENT_NAMES):
            vector[8 + index] = float(agent in state["agents_already_called"])
            vector[16 + index] = min(state["agents_already_called"].count(agent) / 4, 1)
        words = re.findall(r"\w+|[^\w\s]", text.lower())
        terms = words + [f"{a}:{b}" for a, b in zip(words, words[1:])]
        for term in terms:
            hashed = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "little")
            vector[32 + hashed % (self.dimension - 32)] += 1 if hashed & (1 << 63) else -1
        vector[32:] /= max(float(vector[32:].norm()), 1)
        return vector

    def batch(self, states: list[ExecutionState]) -> torch.Tensor:
        if not states:
            return torch.empty((0, self.dimension))
        return torch.stack([self.encode_text(serialize_state(state)) for state in states])
