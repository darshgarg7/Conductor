"""Pretrained Hugging Face MoE backbone with a learned constrained action head.

SFT/DPO operate on probabilities of complete discrete routing actions. Backbone
LoRA adapters and the action head belong to the coordinator. Specialists are
never referenced by training. Untrained action heads are labeled explicitly.
"""
from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from conductor.controller.actions import ActionCatalog
from conductor.controller.experts import ExpertTracker
from conductor.controller.features import serialize_state
from conductor.schema import ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


class HFRoutingModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_actions: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, num_actions)

    def forward(self, inputs: dict[str, torch.Tensor], output_router_logits: bool = False
                ) -> tuple[torch.Tensor, Any]:
        output = self.backbone(**inputs, use_cache=False, return_dict=True,
                               output_router_logits=output_router_logits)
        positions = torch.arange(inputs["attention_mask"].shape[1], device=inputs["input_ids"].device)
        last = (positions[None, :] * inputs["attention_mask"]).max(-1).values.long()
        hidden = output.last_hidden_state[torch.arange(len(last), device=last.device), last]
        return self.head(hidden.to(self.head.weight.dtype)), getattr(output, "router_logits", None)


class HFController:
    supported_optimizations = {"batching": True, "feature_cache": False, "prefix_kv_cache": False,
                               "mixed_precision": True, "dynamic_batching": True}

    def __init__(self, config: dict[str, Any], checkpoint: str | Path | None = None) -> None:
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError("HF controllers require `pip install -e '.[hf]'`") from error
        self.config = copy.deepcopy(config)
        model_config = self.config["model"]
        self.base_model_name = model_config.get("name", "allenai/OLMoE-1B-7B-0924")
        if Path(self.base_model_name).is_dir():
            self.base_model_name = str(Path(self.base_model_name).resolve())
            model_config["name"] = self.base_model_name
        self.revision = model_config.get("resolved_revision") or model_config.get("revision", "main")
        self.pretrained = bool(model_config.get("pretrained", True))
        self.catalog = ActionCatalog(int(model_config.get("max_agents", 3)))
        self.device = torch.device(model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        precision = model_config.get("dtype", "float32")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        self.dtype = getattr(torch, precision)
        self.compute_precision = f"{precision} frozen backbone; float32 trainable head and adapters"
        source_kwargs = {"revision": self.revision, "local_files_only": bool(model_config.get("local_files_only", False)),
                         "trust_remote_code": bool(model_config.get("trust_remote_code", False))}
        architecture = AutoConfig.from_pretrained(self.base_model_name, **source_kwargs)
        self.resolved_revision = getattr(architecture, "_commit_hash", None)
        self.local_checkpoint_sha256 = None
        if Path(self.base_model_name).is_dir():
            digest = hashlib.sha256()
            for file in sorted(Path(self.base_model_name).rglob("*")):
                if file.is_file():
                    digest.update(str(file.relative_to(self.base_model_name)).encode())
                    with file.open("rb") as handle:
                        while block := handle.read(1024 * 1024):
                            digest.update(block)
            self.local_checkpoint_sha256 = digest.hexdigest()
            expected = model_config.get("local_checkpoint_sha256")
            if expected and expected != self.local_checkpoint_sha256:
                raise ValueError("local HF base checkpoint changed since the coordinator checkpoint was saved")
            model_config["local_checkpoint_sha256"] = self.local_checkpoint_sha256
        if self.resolved_revision:
            self.revision = self.resolved_revision
            source_kwargs["revision"] = self.resolved_revision
            model_config["resolved_revision"] = self.resolved_revision
        if not hasattr(architecture, "num_experts") and not hasattr(architecture, "num_local_experts"):
            raise ValueError("HF base must be an actual MoE with num_experts/num_local_experts")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, **source_kwargs)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Token-budgeted header + most recent state tail: the task is never lost
        # merely because sort_keys placed user_task after a large output history.
        self.max_length = int(model_config.get("max_length", 1024))
        if self.max_length < 16:
            raise ValueError("HF max_length must be >=16")
        self.context_strategy = "task/type header capped at 1/3 context; newest serialized execution-state tail uses remainder"
        backbone = AutoModel.from_pretrained(self.base_model_name, torch_dtype=self.dtype, **source_kwargs)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        self.lora_config = model_config.get("lora", {"enabled": True})
        if self.lora_config.get("enabled", True):
            try:
                from peft import LoraConfig, PeftModel, TaskType, get_peft_model
            except ImportError as error:
                raise ImportError("LoRA requires `pip install -e '.[hf]'`") from error
            if checkpoint is not None:
                backbone = PeftModel.from_pretrained(backbone, str(Path(checkpoint) / "adapter"), is_trainable=True)
            else:
                adapter = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                     r=int(self.lora_config.get("r", 8)),
                                     lora_alpha=int(self.lora_config.get("alpha", 16)),
                                     lora_dropout=float(self.lora_config.get("dropout", 0.0)),
                                     target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj", "gate"]),
                                     bias="none")
                backbone = get_peft_model(backbone, adapter)
        # Float32 action head and adapters; pretrained backbone runs at selected
        # precision. Keep trainable parameters in float32 for stable optimization.
        self.model = HFRoutingModel(backbone, architecture.hidden_size, len(self.catalog)).to(device=self.device)
        self.stage = "base-with-random-action-head"
        self.name = "base-moe-random-action-head" if self.pretrained else "hf-random-smoke"
        if checkpoint is not None:
            metadata = json.loads((Path(checkpoint) / "controller.json").read_text())
            self.model.head.load_state_dict(torch.load(Path(checkpoint) / "head.pt", map_location=self.device, weights_only=True))
            self.stage = metadata["stage"]
            self.name = f"hf-{self.stage}"
        self.tracker = ExpertTracker()
        self.instrument_experts = bool(config.get("inference", {}).get("instrument_experts", True))
        self.expert_top_k = int(getattr(architecture, "num_experts_per_tok", 1))
        self.sample = bool(config.get("inference", {}).get("sample", False))
        self.temperature = float(config.get("inference", {}).get("temperature", 1.0))
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.last_tokens = 0
        self.last_cost_usd = 0.0
        self.last_batch_tokens: list[int] = []
        self.last_batch_costs: list[float] = []
        self.last_invalid = False
        self._last_encoded_tokens: list[int] = []
        self.invalid_decisions = 0
        self.token_accounting = "HF tokenizer input tokens; classifier emits no language tokens"
        self.model.eval()

    @staticmethod
    def serialize(state: ExecutionState) -> str:
        return serialize_state(state)

    def encode_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        items = []
        reserve = self.tokenizer.num_special_tokens_to_add(pair=False)
        budget = self.max_length - reserve
        for state in states:
            header = self.tokenizer.encode(f"User task: {state.user_task}\nTask type: {state.task_type}\nExecution state:\n", add_special_tokens=False)
            header = header[:max(1, budget // 3)]
            whole = state.to_dict()
            fields = ("conversation_state", "agents_already_called", "remaining_budget", "current_step",
                      "previous_routing_decisions", "tool_results", "previous_agent_outputs")
            tail = self.tokenizer.encode(json.dumps({key: whole[key] for key in fields}, separators=(",", ":"), ensure_ascii=False), add_special_tokens=False)
            inputs = self.tokenizer.build_inputs_with_special_tokens(header + tail[-(budget - len(header)):])
            items.append(inputs)
        encoded = self.tokenizer.pad({"input_ids": items}, padding=True, return_tensors="pt")
        self._last_encoded_tokens = encoded["attention_mask"].sum(-1).tolist()
        return {key: value.to(self.device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}

    def forward_states(self, states: list[ExecutionState], k: int | None = None,
                       track: bool = False) -> torch.Tensor:
        inputs = self.encode_states(states)
        logits, router_logits = self.model(inputs, output_router_logits=track and self.instrument_experts)
        if track and self.instrument_experts and router_logits:
            batch_size, sequence_length = inputs["input_ids"].shape
            for layer, raw in enumerate(router_logits):
                values = raw.reshape(batch_size, sequence_length, -1)
                self.tracker.add(values, self.expert_top_k, [state.task_type for state in states], str(layer), inputs["attention_mask"])
        if k is not None:
            logits = logits.masked_fill(~self.catalog.mask(k, logits.device), float("-inf"))
        return logits

    def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
        if not states:
            self.last_batch_tokens, self.last_batch_costs = [], []
            self.last_tokens, self.last_cost_usd = 0, 0.0
            return []
        self.model.eval()
        with torch.inference_mode():
            probabilities = (self.forward_states(states, k=k, track=True).float() / self.temperature).softmax(-1)
        self.last_batch_tokens = list(self._last_encoded_tokens)
        rate = float(self.config["model"].get("input_cost_per_million_tokens", 0.0))
        self.last_batch_costs = [tokens * rate / 1_000_000 for tokens in self.last_batch_tokens]
        self.last_tokens, self.last_cost_usd = sum(self.last_batch_tokens), sum(self.last_batch_costs)
        decisions = []
        self.last_invalid = False
        for distribution in probabilities:
            if not bool(torch.isfinite(distribution).all()):
                # Fail closed, with an explicit flag/counter. Never repair with a heuristic.
                self.last_invalid = True
                self.invalid_decisions += 1
                decisions.append(RoutingDecision([], confidence=0.0, terminate=True))
                continue
            index = int(torch.multinomial(distribution, 1)[0] if self.sample else distribution.argmax())
            decisions.append(self.catalog.decision(index, float(distribution[index])).validate(k))
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
        if self.lora_config.get("enabled", True):
            self.model.backbone.save_pretrained(directory / "adapter")
        torch.save(self.model.head.state_dict(), directory / "head.pt")
        write_json(directory / "controller.json", {"backend": "hf", "configuration": self.config, "stage": stage,
                   "pretrained": self.pretrained, "base_model": self.base_model_name, "revision": self.revision,
                   "resolved_revision": self.resolved_revision, "local_checkpoint_sha256": self.local_checkpoint_sha256,
                   "action_catalog_size": len(self.catalog), "token_accounting": self.token_accounting,
                   "context_strategy": self.context_strategy, "max_length": self.max_length,
                   "method": "constrained categorical routing head with optional coordinator-backbone LoRA"})

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> HFController:
        from conductor.utils.config import merge
        metadata = json.loads((Path(path) / "controller.json").read_text())
        config = metadata["configuration"]
        if overrides:
            config = merge(config, {"inference": overrides.get("inference", {})})
            for key in ("device", "dtype", "max_length"):
                if key in overrides.get("model", {}):
                    config["model"][key] = overrides["model"][key]
        return cls(config, checkpoint=path)
