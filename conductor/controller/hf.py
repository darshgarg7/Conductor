"""Pretrained Hugging Face MoE backbone with a learned constrained action head.

SFT/DPO operate on probabilities of complete discrete routing actions. Backbone
LoRA adapters and the action head belong to the coordinator. Specialists are
never referenced by training. Untrained action heads are labeled explicitly.
"""
from __future__ import annotations

import json
import copy
import hashlib
import platform
import shutil
from importlib.metadata import version
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from conductor.controller.actions import ActionCatalog
from conductor.controller.artifacts import atomic_directory, atomic_json
from conductor.controller.experts import ExpertTracker
from conductor.schema import ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
        self.device = torch.device(model_config.get("device", "cuda"))
        precision = model_config.get("dtype", "float32")
        if precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        self.dtype = getattr(torch, precision)
        from conductor.utils.hardware import validate_device
        self.hardware_validation = validate_device(str(self.device), precision, require_cuda=bool(model_config.get("require_cuda", False)))
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
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
        attention = model_config.get("attention_implementation", "sdpa")
        if attention not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("attention_implementation must be eager, sdpa, or flash_attention_2")
        backbone = AutoModel.from_pretrained(self.base_model_name, torch_dtype=self.dtype,
                                             attn_implementation=attention, **source_kwargs)
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
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.actual_expert_top_k = int(getattr(architecture, "num_experts_per_tok", 1))
        self._compiled_model: Any = None
        self.compile_status: dict[str, Any] = {"enabled": False, "validated": False}
        self._token_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._token_cache_size = int(config.get("inference", {}).get("token_cache_size", 0))
        if self._token_cache_size < 0:
            raise ValueError("token_cache_size must be nonnegative")
        self._token_cache_hits = self._token_cache_misses = 0
        self.source_checkpoint = str(checkpoint) if checkpoint is not None else None
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
        whole = state.to_dict()
        fields = ("conversation_state", "agents_already_called", "remaining_budget", "current_step",
                  "previous_routing_decisions", "tool_results", "previous_agent_outputs")
        header = f"User task: {state.user_task}\nTask type: {state.task_type}\nExecution state:\n"
        tail = json.dumps({key: whole[key] for key in fields}, separators=(",", ":"), ensure_ascii=False)
        return header + "\0CONDUCTOR_STATE\0" + tail

    def tokenize_serialized(self, texts: list[str]) -> dict[str, torch.Tensor]:
        items = []
        reserve = self.tokenizer.num_special_tokens_to_add(pair=False)
        budget = self.max_length - reserve
        for text in texts:
            cached = self._token_cache.get(text)
            if cached is not None:
                self._token_cache_hits += 1
                self._token_cache.move_to_end(text)
                items.append(list(cached))
                continue
            self._token_cache_misses += 1
            header_text, tail_text = text.rsplit("\0CONDUCTOR_STATE\0", 1)
            header = self.tokenizer.encode(header_text, add_special_tokens=False)
            header = header[:max(1, budget // 3)]
            tail = self.tokenizer.encode(tail_text, add_special_tokens=False)
            inputs = self.tokenizer.build_inputs_with_special_tokens(header + tail[-(budget - len(header)):])
            items.append(inputs)
            if self._token_cache_size:
                self._token_cache[text] = tuple(inputs)
                while len(self._token_cache) > self._token_cache_size:
                    self._token_cache.popitem(last=False)
        encoded = self.tokenizer.pad({"input_ids": items}, padding=True, return_tensors="pt")
        self._last_encoded_tokens = encoded["attention_mask"].sum(-1).tolist()
        return {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}

    def tokenize_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.tokenize_serialized([self.serialize(state) for state in states])

    def move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=self.device.type == "cuda" and value.is_pinned()) for key, value in inputs.items()}

    def encode_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.move_inputs(self.tokenize_states(states))

    def token_cache_clear(self) -> None:
        self._token_cache.clear()
        self._token_cache_hits = self._token_cache_misses = 0

    def token_cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._token_cache), "max_entries": self._token_cache_size,
                "hits": self._token_cache_hits, "misses": self._token_cache_misses}

    def compile_for_inference(self, backend: str = "inductor", mode: str = "default", fullgraph: bool = False) -> None:
        if self.model.training:
            raise ValueError("compile_for_inference requires eval mode")
        self._compiled_model = torch.compile(self.model, backend=backend, mode=mode, fullgraph=fullgraph)
        self.compile_status = {"enabled": True, "validated": False, "backend": backend, "mode": mode, "fullgraph": fullgraph}

    def enable_training(self, gradient_checkpointing: bool = False) -> None:
        self._compiled_model = None
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith("head.") or "lora_" in name)
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if gradient_checkpointing:
            self.model.backbone.enable_input_require_grads()
            self.model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()

    def forward_encoded(self, inputs: dict[str, torch.Tensor], k: int | None = None,
                        track: bool = False, task_types: list[str] | None = None) -> torch.Tensor:
        auxiliary_weight = float(self.config.get("training", {}).get("router_auxiliary_weight", 0)) if self.model.training else 0.0
        if auxiliary_weight < 0:
            raise ValueError("router_auxiliary_weight cannot be negative")
        needs_router = (track and self.instrument_experts) or auxiliary_weight > 0
        runner = self._compiled_model if not self.model.training and self._compiled_model is not None else self.model
        try:
            logits, router_logits = runner(inputs, output_router_logits=needs_router)
            if runner is self._compiled_model:
                self.compile_status["validated"] = True
        except Exception as error:
            if runner is self._compiled_model:
                raise RuntimeError("requested torch.compile inference failed; no eager fallback was used") from error
            raise
        self.last_auxiliary_loss = None
        if auxiliary_weight and router_logits:
            losses = []
            valid = inputs["attention_mask"].reshape(-1).float()
            for raw in router_logits:
                probabilities = raw.reshape(-1, raw.shape[-1]).float().softmax(-1)
                selected = probabilities.topk(self.actual_expert_top_k, -1).indices
                fraction = F.one_hot(selected, probabilities.shape[-1]).float().mean(1)
                loads = (fraction * valid[:, None]).sum(0) / valid.sum().clamp_min(1)
                mean_probability = (probabilities * valid[:, None]).sum(0) / valid.sum().clamp_min(1)
                losses.append(probabilities.shape[-1] * (loads.detach() * mean_probability).sum())
            self.last_auxiliary_loss = torch.stack(losses).mean()
        elif auxiliary_weight:
            raise ValueError("requested router auxiliary loss, but this HF backbone returned no router logits")
        if track and self.instrument_experts and router_logits:
            batch_size, sequence_length = inputs["input_ids"].shape
            for layer, raw in enumerate(router_logits):
                self.tracker.add(raw.reshape(batch_size, sequence_length, -1), self.expert_top_k,
                                 task_types or ["unspecified"] * batch_size, str(layer), inputs["attention_mask"])
        if k is not None:
            logits = logits.masked_fill(~self.catalog.mask(k, logits.device), float("-inf"))
        return logits

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

    def forward_states(self, states: list[ExecutionState], k: int | None = None,
                       track: bool = False) -> torch.Tensor:
        return self.forward_encoded(self.encode_states(states), k, track, [state.task_type for state in states])

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

    def export_merged(self, path: str | Path, validation_states: list[ExecutionState] | None = None) -> dict[str, Any]:
        """Self-contained inference artifact; preserve the original source adapter.

        Merging clones the coordinator backbone and can temporarily double its
        resident weight memory. Publication fails if measured logits/routes on
        the supplied probes are not equivalent within numerical tolerances.
        """
        if self.model.training:
            raise ValueError("merged export requires eval mode")
        target = Path(path).resolve()
        probes = validation_states or [ExecutionState("Calculate 2 plus 3", "math")]
        backbone = copy.deepcopy(self.model.backbone)
        if hasattr(backbone, "merge_and_unload"):
            backbone = backbone.merge_and_unload(safe_merge=True)
        merged = HFRoutingModel(backbone, self.model.head.in_features, len(self.catalog)).to(self.device)
        merged.head.load_state_dict(self.model.head.state_dict())
        merged.eval()
        inputs = self.encode_states(probes)
        with torch.inference_mode():
            before, _ = self.model(inputs, output_router_logits=False)
            after, _ = merged(inputs, output_router_logits=False)
        if not bool(torch.isfinite(after).all()):
            raise FloatingPointError("merged export produced non-finite validation logits")
        torch.testing.assert_close(before.float(), after.float(), rtol=2e-3 if self.dtype != torch.float32 else 1e-4,
                                   atol=2e-3 if self.dtype != torch.float32 else 1e-5)
        if not torch.equal(before.argmax(-1), after.argmax(-1)):
            raise ValueError("merged export changed a routing action on validation probes")
        configuration = copy.deepcopy(self.config)
        configuration["model"].update(name=str(target / "backbone"), revision="local-export", local_files_only=True,
                                        lora={"enabled": False})
        configuration["model"].pop("resolved_revision", None)
        configuration["model"].pop("local_checkpoint_sha256", None)
        configuration["inference"] = {**configuration.get("inference", {}), "compile": {"enabled": False}}
        metadata = {"backend": "hf", "configuration": configuration, "stage": self.stage, "pretrained": self.pretrained,
                    "format": "merged-hf-v1", "original_base_model": self.base_model_name,
                    "original_resolved_revision": self.resolved_revision, "source_checkpoint": self.source_checkpoint,
                    "validation_probe_count": len(probes), "maximum_logit_difference": float((before.float() - after.float()).abs().max()),
                    "action_catalog_size": len(self.catalog), "token_accounting": self.token_accounting,
                    "context_strategy": self.context_strategy, "max_length": self.max_length}
        with atomic_directory(target) as temporary:
            backbone.save_pretrained(temporary / "backbone", safe_serialization=True)
            self.tokenizer.save_pretrained(temporary / "backbone")
            torch.save(self.model.head.state_dict(), temporary / "head.pt")
            if self.source_checkpoint and (Path(self.source_checkpoint) / "adapter").exists():
                shutil.copytree(Path(self.source_checkpoint) / "adapter", temporary / "source_adapter")
            elif self.lora_config.get("enabled", True):
                self.model.backbone.save_pretrained(temporary / "source_adapter")
            atomic_json(temporary / "controller.json", metadata)
            files = {str(file.relative_to(temporary)): _file_sha256(file)
                     for file in sorted(temporary.rglob("*")) if file.is_file()}
            manifest = {"format": "merged-hf-v1", "files_sha256": files, "validation": metadata,
                        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                     "transformers": version("transformers"), "peft": version("peft")}}
            atomic_json(temporary / "manifest.json", manifest)
        return manifest

    @classmethod
    def load(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> HFController:
        from conductor.utils.config import merge
        metadata = json.loads((Path(path) / "controller.json").read_text())
        if metadata.get("format") == "merged-hf-v1":
            manifest = json.loads((Path(path) / "manifest.json").read_text())
            for relative, expected in manifest["files_sha256"].items():
                if _file_sha256(Path(path) / relative) != expected:
                    raise ValueError(f"merged export checksum mismatch: {relative}")
        config = metadata["configuration"]
        if metadata.get("format") == "merged-hf-v1":
            config["model"]["name"] = str((Path(path) / "backbone").resolve())
        if overrides:
            config = merge(config, {"inference": overrides.get("inference", {})})
            for key in ("device", "dtype", "max_length", "attention_implementation", "require_cuda"):
                if key in overrides.get("model", {}):
                    config["model"][key] = overrides["model"][key]
        return cls(config, checkpoint=path)
