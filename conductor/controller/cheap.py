"""Small learned public-state controls for the coordination experiment.

These models have no language-model backbone and no experts.  Both controls
share a fixed representation: separate signed lexical hashes for the request
and public evidence, plus execution/budget indicators.  Public status fields
are evidence, never an answer checker.  The representation excludes task family,
grader labels, dependency graphs and measured timing/cost/token bookkeeping.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn

from conductor.controller.actions import ActionCatalog
from conductor.controller.artifacts import resolve_checkpoint
from conductor.controller.features import token_count
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


PUBLIC_FEATURE_VERSION = "public_split_hash_v1"
_EXCLUDED = frozenset({
    "expected_answer", "grader_score", "task_success", "reward", "chosen_reward", "rejected_reward",
    "required_agents", "family", "workflow_family", "template_family", "dependency_graph", "structure",
    "source_group", "task_id", "split", "optimal_route", "oracle_completion", "private_metadata",
    "tokens", "input_tokens", "output_tokens", "total_tokens", "token_usage", "token_accounting",
    "latency", "latency_seconds", "elapsed_seconds", "wall_clock_latency", "wall_clock_latency_seconds",
    "cost", "cost_usd", "estimated_cost_usd", "estimated_inference_cost", "controller_tokens",
    "controller_latency_seconds", "controller_cost_usd", "confidence",
})


def public_evidence(value: Any) -> Any:
    """Sanitize structured fields and JSON artifact strings, without editing prose.

    Text supplied by a real agent is inherently public evidence.  Valid JSON
    content is recursively sanitized so embedded telemetry cannot become a
    policy fingerprint.  Budget fields are restored separately by public_state.
    """
    if isinstance(value, dict):
        return {key: public_evidence(item) for key, item in value.items()
                if key not in _EXCLUDED and not key.startswith("private_")}
    if isinstance(value, list):
        return [public_evidence(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        return json.dumps(public_evidence(parsed), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value


def public_state(state: ExecutionState) -> dict[str, Any]:
    """The common public input; task_type is omitted rather than used as an oracle."""
    value = public_evidence(state.to_dict())
    value.pop("task_type", None)
    value["remaining_budget"] = {
        name: _budget(state.remaining_budget.get(name, 0), name) for name in ("agent_calls", "tokens")
    }
    return value


def serialize_public_state(state: ExecutionState) -> str:
    return json.dumps(public_state(state), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _budget(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"remaining {name} budget must be a finite nonnegative number")
    return float(value)


class PublicStateFeatures:
    """Fixed separate lexical channels, preserving request signal in long histories.

    No learned vocabulary is fitted on held-out inputs.  Hash collisions are
    possible and are part of this explicitly cheap baseline.  Source IDs and
    numeric values do not have privileged positions in the representation.
    """
    def __init__(self, dimension: int = 512, cache_size: int = 0) -> None:
        if dimension < 128:
            raise ValueError("public feature dimension must be >=128")
        if cache_size < 0:
            raise ValueError("feature_cache_size must be nonnegative")
        self.dimension = dimension
        self.encode_text = lru_cache(maxsize=cache_size)(self._encode) if cache_size else self._encode

    @staticmethod
    def _hash(text: str, target: torch.Tensor) -> None:
        words = re.findall(r"[a-z_]+|[0-9]+|[^\w\s]", text.lower())
        terms = words + [f"{left}:{right}" for left, right in zip(words, words[1:])]
        for term in terms:
            digest = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "little")
            target[digest % len(target)] += 1 if digest & (1 << 63) else -1
        target /= max(float(target.norm()), 1)

    def _encode(self, text: str) -> torch.Tensor:
        state = json.loads(text)
        vector = torch.zeros(self.dimension, dtype=torch.float32)
        calls = _budget(state["remaining_budget"].get("agent_calls", 0), "agent_calls")
        tokens = _budget(state["remaining_budget"].get("tokens", 0), "tokens")
        vector[0] = 1
        vector[1] = min(state["current_step"] / 6, 2)
        vector[2] = min(calls / 12, 2)
        vector[3] = min(tokens / 16384, 2)
        outputs = state.get("previous_agent_outputs", [])
        tool_results = state.get("tool_results", [])
        called = state.get("agents_already_called", [])
        vector[4] = min(len(outputs) / 12, 2)
        vector[5] = min(len(tool_results) / 12, 2)
        vector[6] = min(len(state.get("previous_routing_decisions", [])) / 6, 2)
        vector[7] = float(calls < 1 or tokens <= 0)
        for index, agent in enumerate(AGENT_NAMES):
            vector[8 + index] = float(agent in called)
            vector[16 + index] = min(called.count(agent) / 4, 2)
            vector[24 + index] = float(bool(outputs) and outputs[-1].get("agent") == agent)
        # Count explicit public statuses; a "verified" artifact is still not a
        # correctness certificate and its provenance/content remain in hashes.
        statuses: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("status"), str):
                    statuses[value["status"].lower()] += 1
                if isinstance(value.get("kind"), str):
                    kinds[value["kind"].lower()] += 1
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                try:
                    visit(json.loads(value))
                except ValueError:
                    pass
        visit(outputs)
        visit(tool_results)
        for offset, status in enumerate(("ok", "success", "failed", "error", "blocked", "partial", "verified", "invalid")):
            vector[32 + offset] = min(statuses[status] / 4, 2)
        # Operation and verification are user requirements in the public
        # structured request, not workflow-family labels or source-store access.
        try:
            request = json.loads(state.get("user_task", ""))
        except ValueError:
            request = {}
        if isinstance(request, dict):
            for offset, operation in enumerate(("calculate", "transform", "research", "combine", "resolve_conflict", "read_tool")):
                vector[40 + offset] = float(request.get("operation") == operation)
            vector[46] = float(request.get("verification") is True)
        for offset, kind in enumerate(("fact", "research", "program", "candidate", "verification", "execution", "error",
                                       "plan", "critique", "selector", "threshold", "tool_result", "partial", "repair")):
            vector[48 + offset] = min(kinds[kind] / 4, 2)
        boundary = 64 + (self.dimension - 64) // 2
        self._hash(state.get("user_task", ""), vector[64:boundary])
        evidence = {key: value for key, value in state.items() if key not in {"user_task", "remaining_budget"}}
        self._hash(json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False), vector[boundary:])
        return vector


class CheapRoutingModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, architecture: str, catalog: ActionCatalog,
                 action_head: str) -> None:
        super().__init__()
        if architecture == "linear":
            self.encoder = nn.Identity()
            representation_dim = feature_dim
        elif architecture == "mlp":
            if hidden_dim < 1:
                raise ValueError("hidden_dim must be positive")
            self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            representation_dim = hidden_dim
        else:
            raise ValueError("cheap architecture must be linear or mlp")
        self.action_head = action_head
        if action_head == "catalog":
            self.head = nn.Linear(representation_dim, len(catalog))
        elif action_head == "factorized":
            from conductor.controller.factorized import FactorizedHead
            self.head = FactorizedHead(representation_dim, max_agents=catalog.max_agents, agents=catalog.agents)
        else:
            raise ValueError("action_head must be catalog or factorized")

    def forward(self, inputs: dict[str, torch.Tensor], k: int) -> torch.Tensor:
        hidden = self.encoder(inputs["features"])
        if self.action_head == "factorized":
            return self.head(hidden, k=k, call_budgets=inputs["agent_calls"], token_budgets=inputs["token_budget"])
        return self.head(hidden)


class CheapController:
    feature_version = PUBLIC_FEATURE_VERSION
    supported_optimizations = {"batching": True, "feature_cache": True, "prefix_kv_cache": False,
                               "mixed_precision": False, "dynamic_batching": True}
    pretrained = False
    token_accounting = "sanitized public-state lexical token proxy; no language-model tokenizer"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        model_config = self.config.get("model", self.config)
        self.architecture = model_config.get("architecture", "linear")
        self.action_head = model_config.get("action_head", "catalog")
        self.catalog = ActionCatalog(int(model_config.get("max_agents", 3)))
        self.device = torch.device(model_config.get("device", "cpu"))
        if model_config.get("dtype", "float32") != "float32":
            raise ValueError("cheap controls use float32 for comparable CPU experiments")
        from conductor.utils.hardware import validate_device
        self.hardware_validation = validate_device(str(self.device), "float32", bool(model_config.get("require_cuda", False)))
        self.dtype = torch.float32
        self.compute_precision = "float32"
        self.features = PublicStateFeatures(int(model_config.get("feature_dim", 512)),
                                           int(config.get("inference", {}).get("feature_cache_size", 0)))
        self.model = CheapRoutingModel(self.features.dimension, int(model_config.get("hidden_dim", 64)),
                                       self.architecture, self.catalog, self.action_head).to(self.device)
        self.stage = "random_initialization"
        self.name = f"cheap-{self.architecture}-{self.action_head}-untrained"
        self.last_tokens = 0
        self.last_cost_usd = 0.0
        self.last_batch_tokens: list[int] = []
        self.last_batch_costs: list[float] = []
        self.last_invalid = False
        self.invalid_decisions = 0
        self.sample = bool(config.get("inference", {}).get("sample", False))
        self.temperature = float(config.get("inference", {}).get("temperature", 1))
        if not 0 < self.temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        self.model.eval()

    serialize = staticmethod(serialize_public_state)

    def tokenize_serialized(self, texts: list[str]) -> dict[str, torch.Tensor]:
        states = [json.loads(text) for text in texts]
        return {"features": torch.stack([self.features.encode_text(text) for text in texts])
                if texts else torch.empty((0, self.features.dimension)),
                "agent_calls": torch.tensor([_budget(state["remaining_budget"].get("agent_calls", 0), "agent_calls")
                                             for state in states], dtype=torch.float32),
                "token_budget": torch.tensor([_budget(state["remaining_budget"].get("tokens", 0), "tokens")
                                              for state in states], dtype=torch.float32)}

    def tokenize_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.tokenize_serialized([self.serialize(state) for state in states])

    def move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in inputs.items()}

    def encode_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.move_inputs(self.tokenize_states(states))

    def forward_encoded(self, inputs: dict[str, torch.Tensor], k: int | None = None, track: bool = False,
                        task_types: list[str] | None = None) -> torch.Tensor:
        del track, task_types
        limit = self.catalog.max_agents if k is None else k
        logits = self.model(inputs, limit)
        if self.action_head == "catalog":
            call_limit = inputs["agent_calls"].floor().clamp(max=limit)
            sizes = torch.tensor([len(action.selected_agents) for action in self.catalog.actions], device=self.device)
            valid = self.catalog.mask(limit, self.device)[None, :] & (sizes[None, :] <= call_limit[:, None])
            valid &= (inputs["token_budget"][:, None] > 0) | (sizes[None, :] == 0)
            logits = logits.masked_fill(~valid, float("-inf"))
        return logits

    def forward_states(self, states: list[ExecutionState], k: int | None = None, track: bool = False) -> torch.Tensor:
        return self.forward_encoded(self.encode_states(states), k, track)

    def decide(self, logits: torch.Tensor, k: int) -> list[RoutingDecision]:
        probabilities = (logits.float().masked_fill(~self.catalog.mask(k, logits.device), float("-inf"))
                         / self.temperature).softmax(-1)
        decisions = []
        self.last_invalid = False
        for distribution in probabilities:
            if not bool(torch.isfinite(distribution).all()):
                self.last_invalid = True
                self.invalid_decisions += 1
                decisions.append(RoutingDecision([], confidence=0, terminate=True))
            else:
                index = int(torch.multinomial(distribution, 1)[0] if self.sample else distribution.argmax())
                decisions.append(self.catalog.decision(index, float(distribution[index])).validate(k))
        return decisions

    def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
        self.model.eval()
        with torch.inference_mode():
            decisions = self.decide(self.forward_states(states, k), k) if states else []
        self.last_batch_tokens = [token_count(self.serialize(state)) for state in states]
        self.last_batch_costs = [0.0] * len(states)
        self.last_tokens = sum(self.last_batch_tokens)
        self.last_cost_usd = 0.0
        return decisions

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        return self.batch_route([state], k)[0]

    def enable_training(self, gradient_checkpointing: bool = False) -> None:
        if gradient_checkpointing:
            raise ValueError("cheap control has no transformer checkpointing")
        self.model.requires_grad_(True)
        self.model.train()

    def expert_stats(self) -> dict[str, Any]:
        return {"available": False, "reason": "dense cheap control has no MoE experts"}

    def reset_expert_stats(self) -> None:
        pass

    def token_cache_clear(self) -> None:
        if hasattr(self.features.encode_text, "cache_clear"):
            self.features.encode_text.cache_clear()

    def token_cache_stats(self) -> dict[str, Any]:
        if hasattr(self.features.encode_text, "cache_info"):
            info = self.features.encode_text.cache_info()
            return {"entries": info.currsize, "max_entries": info.maxsize, "hits": info.hits,
                    "misses": info.misses, "scope": "sanitized public-state feature cache"}
        return {"entries": 0, "max_entries": 0, "hits": 0, "misses": 0}

    def save(self, path: str | Path, stage: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        head_metadata = self.model.head.metadata() if hasattr(self.model.head, "metadata") else {
            "head_type": "catalog", "action_catalog_size": len(self.catalog)}
        write_json(directory / "controller.json", {"backend": "cheap", "configuration": self.config,
                   "stage": stage, "pretrained": False, "action_catalog_size": len(self.catalog),
                   "feature_version": PUBLIC_FEATURE_VERSION, "action_head": self.action_head,
                   "head_metadata": head_metadata,
                   "token_accounting": self.token_accounting})

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> CheapController:
        from conductor.utils.config import merge
        directory = resolve_checkpoint(path)
        metadata = json.loads((directory / "controller.json").read_text())
        if metadata.get("backend") != "cheap" or metadata.get("feature_version") != PUBLIC_FEATURE_VERSION:
            raise ValueError("unsupported cheap-controller checkpoint representation")
        config = copy.deepcopy(metadata["configuration"])
        if overrides:
            config = merge(config, {"inference": overrides.get("inference", {})})
            for key in ("device", "dtype"):
                if key in overrides.get("model", {}):
                    config["model"][key] = overrides["model"][key]
        instance = cls(config)
        if metadata.get("action_head") != instance.action_head or metadata["action_catalog_size"] != len(instance.catalog):
            raise ValueError("cheap-controller checkpoint action space changed")
        instance.model.load_state_dict(torch.load(directory / "model.pt", map_location=instance.device, weights_only=True))
        instance.model.eval()
        instance.stage = metadata["stage"]
        instance.name = f"cheap-{instance.architecture}-{instance.action_head}-{instance.stage}"
        return instance
