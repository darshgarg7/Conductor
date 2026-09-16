"""An actually sparse neural MoE policy, randomly initialized for development.

Only selected experts execute. This backend validates the research machinery;
it cannot provide evidence about post-training a pretrained language model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from conductor.controller.actions import ActionCatalog
from conductor.controller.experts import ExpertTracker
from conductor.controller.features import StateFeatures, serialize_state, token_count
from conductor.schema import ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


class SparseMoE(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_experts: int, expert_top_k: int, num_actions: int) -> None:
        super().__init__()
        if not 1 <= expert_top_k <= num_experts:
            raise ValueError("expert_top_k must be within expert count")
        self.expert_top_k = expert_top_k
        self.encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_experts)
        ])
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        router_logits = self.gate(hidden)
        scores, selected = router_logits.topk(self.expert_top_k, dim=-1)
        weights = scores.softmax(-1)
        mixed = torch.zeros_like(hidden)
        for expert_id, expert in enumerate(self.experts):
            sample_ids, slots = torch.where(selected == expert_id)
            if sample_ids.numel():
                expert_outputs = expert(hidden.index_select(0, sample_ids))
                contribution = expert_outputs * weights[sample_ids, slots, None]
                mixed = mixed.index_add(0, sample_ids, contribution)
        return self.head(hidden + mixed), router_logits


class TinyController:
    supported_optimizations = {"batching": True, "feature_cache": True, "prefix_kv_cache": False,
                               "mixed_precision": True, "dynamic_batching": True}

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        model_config = config.get("model", config)
        self.catalog = ActionCatalog(int(model_config.get("max_agents", 3)))
        self.device = torch.device(model_config.get("device", "cpu"))
        precision = model_config.get("dtype", "float32")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if precision == "float16" and self.device.type != "cuda":
            raise ValueError("tiny float16 autocast requires CUDA; use bfloat16 on CPU")
        self.dtype = getattr(torch, precision)
        from conductor.utils.hardware import validate_device
        self.hardware_validation = validate_device(str(self.device), precision, require_cuda=bool(model_config.get("require_cuda", False)))
        self.compute_precision = precision
        self.features = StateFeatures(int(model_config.get("feature_dim", 256)),
                                      int(config.get("inference", {}).get("feature_cache_size", 0)))
        self.model = SparseMoE(self.features.dimension, int(model_config.get("hidden_dim", 64)),
                               int(model_config.get("num_experts", 4)), int(model_config.get("expert_top_k", 2)),
                               len(self.catalog)).to(self.device)
        self.actual_expert_top_k = self.model.expert_top_k
        self._compiled_model: Any = None
        self.compile_status: dict[str, Any] = {"enabled": False, "validated": False}
        self.name = "tiny-random-initialized"
        self.stage = "random_initialization"
        self.last_tokens = 0
        self.last_cost_usd = 0.0
        self.last_batch_tokens: list[int] = []
        self.last_batch_costs: list[float] = []
        self.last_invalid = False
        self.invalid_decisions = 0
        self.tracker = ExpertTracker()
        self.instrument_experts = bool(config.get("inference", {}).get("instrument_experts", True))
        self.sample = bool(config.get("inference", {}).get("sample", False))
        self.temperature = float(config.get("inference", {}).get("temperature", 1))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.token_accounting = "whole-state lexical token proxy; no language-model tokenization"
        self.model.eval()

    @staticmethod
    def serialize(state: ExecutionState) -> str:
        return serialize_state(state)

    def encode_states(self, states: list[ExecutionState]) -> torch.Tensor:
        return self.move_inputs(self.tokenize_states(states))

    def tokenize_states(self, states: list[ExecutionState]) -> torch.Tensor:
        return self.tokenize_serialized([self.serialize(state) for state in states])

    def tokenize_serialized(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.features.encode_text(text) for text in texts])

    def move_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.to(self.device)

    def enable_training(self, gradient_checkpointing: bool = False) -> None:
        if gradient_checkpointing:
            raise ValueError("tiny controller has no transformer checkpointing; disable gradient_checkpointing")
        self._compiled_model = None
        self.model.requires_grad_(True)
        self.model.train()

    def compile_for_inference(self, backend: str = "inductor", mode: str = "default", fullgraph: bool = False) -> None:
        if self.model.training:
            raise ValueError("compile_for_inference requires eval mode")
        self._compiled_model = torch.compile(self.model, backend=backend, mode=mode, fullgraph=fullgraph)
        self.compile_status = {"enabled": True, "validated": False, "backend": backend, "mode": mode, "fullgraph": fullgraph}

    def token_cache_clear(self) -> None:
        if hasattr(self.features.encode_text, "cache_clear"):
            self.features.encode_text.cache_clear()

    def token_cache_stats(self) -> dict[str, Any]:
        if hasattr(self.features.encode_text, "cache_info"):
            value = self.features.encode_text.cache_info()
            return {"entries": value.currsize, "max_entries": value.maxsize, "hits": value.hits, "misses": value.misses,
                    "scope": "hashed feature cache; tiny backend has no language tokenizer"}
        return {"entries": 0, "max_entries": 0, "hits": 0, "misses": 0}

    def forward_encoded(self, inputs: torch.Tensor, k: int | None = None, track: bool = False,
                        task_types: list[str] | None = None) -> torch.Tensor:
        runner = self._compiled_model if not self.model.training and self._compiled_model is not None else self.model
        try:
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                logits, expert_logits = runner(inputs)
            if runner is self._compiled_model:
                self.compile_status["validated"] = True
        except Exception as error:
            if runner is self._compiled_model:
                raise RuntimeError("requested torch.compile inference failed; no eager fallback was used") from error
            raise
        if track and self.instrument_experts:
            self.tracker.add(expert_logits, self.actual_expert_top_k, task_types or ["unspecified"] * len(inputs))
        if k is not None:
            logits = logits.masked_fill(~self.catalog.mask(k, logits.device), float("-inf"))
        return logits

    def forward_states(self, states: list[ExecutionState], k: int | None = None,
                       track: bool = False) -> torch.Tensor:
        return self.forward_encoded(self.encode_states(states), k, track, [state.task_type for state in states])

    def decide(self, logits: torch.Tensor, k: int) -> list[RoutingDecision]:
        probabilities = (logits.float().masked_fill(~self.catalog.mask(k, logits.device), float("-inf")) / self.temperature).softmax(-1)
        decisions = []
        self.last_invalid = False
        for distribution in probabilities:
            if not bool(torch.isfinite(distribution).all()):
                self.last_invalid = True
                self.invalid_decisions += 1
                decisions.append(RoutingDecision([], confidence=0.0, terminate=True))
            else:
                index = int(torch.multinomial(distribution, 1)[0] if self.sample else distribution.argmax())
                decisions.append(self.catalog.decision(index, float(distribution[index])).validate(k))
        return decisions

    def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
        if not states:
            self.last_batch_tokens = []
            self.last_batch_costs = []
            self.last_tokens = 0
            return []
        self.model.eval()
        with torch.inference_mode():
            decisions = self.decide(self.forward_states(states, k=k, track=True), k)
        self.last_batch_tokens = [token_count(self.serialize(state)) for state in states]
        self.last_batch_costs = [0.0] * len(states)
        self.last_tokens = sum(self.last_batch_tokens)
        self.last_cost_usd = 0.0
        return decisions

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        return self.batch_route([state], k)[0]

    def expert_stats(self) -> dict[str, Any]:
        return self.tracker.export()

    def reset_expert_stats(self) -> None:
        self.tracker.reset()

    def save(self, path: str | Path, stage: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "model.pt")
        write_json(directory / "controller.json", {"backend": "tiny", "configuration": self.config,
                   "stage": stage, "pretrained": False, "action_catalog_size": len(self.catalog),
                   "token_accounting": self.token_accounting})

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TinyController:
        from conductor.utils.config import merge
        directory = Path(path)
        metadata = json.loads((directory / "controller.json").read_text())
        config = metadata["configuration"]
        if overrides:
            # Architecture comes from checkpoint. Inference and device can vary.
            config = merge(config, {"inference": overrides.get("inference", {})})
            for key in ("device", "dtype"):
                if key in overrides.get("model", {}):
                    config["model"][key] = overrides["model"][key]
        instance = cls(config)
        instance.model.load_state_dict(torch.load(directory / "model.pt", map_location=instance.device, weights_only=True))
        instance.stage = metadata["stage"]
        instance.name = f"tiny-{instance.stage}"
        return instance
