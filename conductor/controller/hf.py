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
from conductor.controller.artifacts import atomic_directory, atomic_json, resolve_checkpoint
from conductor.controller.experts import ExpertTracker
from conductor.schema import ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


MERGED_EXPORT_FP32_TOLERANCES = {"logits": {"rtol": 1e-4, "atol": 1e-4},
                                 "probabilities": {"rtol": 1e-4, "atol": 1e-5}}
MERGED_EXPORT_LOW_PRECISION_TOLERANCES = {"logits": {"rtol": 2e-3, "atol": 2e-3},
                                         "probabilities": {"rtol": 2e-3, "atol": 2e-3}}


def _validate_merged_logits(before: torch.Tensor, after: torch.Tensor, catalog: ActionCatalog,
                            dtype: torch.dtype, temperature: float) -> dict[str, Any]:
    """Gate export on bounded logit drift, probabilities, and every legal budget.

    FP32 permits absolute near-zero logit drift up to 1e-4 from rearranged
    multi-layer sums. That allowance is independently constrained by tighter
    probability tolerances and unchanged greedy decisions for every agent k.
    This verifies supplied probes, not a universal floating-point error bound
    or equivalence of individual stochastic sampling draws.
    """
    if not 0 < temperature < float("inf"):
        raise ValueError("merged validation requires a finite positive temperature")
    if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
        raise FloatingPointError("merged export produced non-finite validation logits")
    before, after = before.float(), after.float()
    defaults = MERGED_EXPORT_FP32_TOLERANCES if dtype == torch.float32 else MERGED_EXPORT_LOW_PRECISION_TOLERANCES
    tolerances = {key: dict(value) for key, value in defaults.items()}
    torch.testing.assert_close(before, after, **tolerances["logits"])
    k_values = list(range(1, catalog.max_agents + 1))
    maximum_probability_difference = 0.0
    for k in k_values:
        mask = catalog.mask(k, before.device)
        before_masked = before.masked_fill(~mask, float("-inf"))
        after_masked = after.masked_fill(~mask, float("-inf"))
        before_probability = (before_masked / temperature).softmax(-1)
        after_probability = (after_masked / temperature).softmax(-1)
        if not bool(torch.isfinite(before_probability).all()) or not bool(torch.isfinite(after_probability).all()):
            raise FloatingPointError(f"merged export produced non-finite probabilities for k={k}")
        torch.testing.assert_close(before_probability, after_probability, **tolerances["probabilities"])
        if (not torch.equal(before_masked.argmax(-1), after_masked.argmax(-1))
                or not torch.equal(before_probability.argmax(-1), after_probability.argmax(-1))):
            raise ValueError(f"merged export changed a routing action on validation probes for k={k}")
        maximum_probability_difference = max(maximum_probability_difference,
                                             float((before_probability - after_probability).abs().max()))
    return {"validation_tolerances": tolerances, "validation_k_values": k_values,
            "maximum_logit_difference": float((before - after).abs().max()),
            "maximum_probability_difference": maximum_probability_difference,
            "probability_temperature": temperature,
            "routing_validation": "unchanged masked-logit and serving-probability greedy argmax for every supported k"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _public_evidence(value: Any) -> Any:
    """Remove private grading fields and measured run accounting recursively."""
    excluded = {"expected_answer", "grader_score", "task_success", "reward", "chosen_reward", "rejected_reward",
                "tokens", "input_tokens", "output_tokens", "total_tokens", "token_usage", "token_accounting",
                "latency", "latency_seconds", "elapsed_seconds", "wall_clock_latency", "wall_clock_latency_seconds",
                "cost", "cost_usd", "estimated_cost_usd", "estimated_inference_cost", "controller_tokens",
                "controller_latency_seconds", "controller_cost_usd"}
    if isinstance(value, dict):
        return {key: _public_evidence(item) for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [_public_evidence(item) for item in value]
    return value


class HFRoutingModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_actions: int,
                 pooling: str = "last", head_input_normalization: str = "none") -> None:
        super().__init__()
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be last or mean")
        if head_input_normalization not in {"none", "layer_norm"}:
            raise ValueError("head_input_normalization must be none or layer_norm")
        self.backbone = backbone
        self.pooling = pooling
        self.head_input_normalization = head_input_normalization
        self.head = nn.Linear(hidden_size, num_actions)

    def pooled_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The exact FP32 head input; usable for frozen-representation ablations."""
        if bool((mask.sum(-1) == 0).any()):
            raise ValueError("routing inputs must contain at least one unmasked token")
        if self.pooling == "mean":
            # FP32 reduction excludes either left or right padding. The head and
            # stateless normalization have no additional checkpoint tensors.
            weights = mask.unsqueeze(-1).to(torch.float32)
            unpadded = hidden_states.float().masked_fill(~mask.bool().unsqueeze(-1), 0)
            hidden = unpadded.sum(1) / weights.sum(1)
        else:
            positions = torch.arange(mask.shape[1], device=mask.device)
            last = (positions[None, :] * mask).max(-1).values.long()
            hidden = hidden_states[torch.arange(len(last), device=last.device), last].float()
        if self.head_input_normalization == "layer_norm":
            hidden = F.layer_norm(hidden, (hidden.shape[-1],))
        return hidden

    def represent(self, inputs: dict[str, torch.Tensor], output_router_logits: bool = False
                  ) -> tuple[torch.Tensor, Any]:
        output = self.backbone(**inputs, use_cache=False, return_dict=True,
                               output_router_logits=output_router_logits)
        return self.pooled_hidden(output.last_hidden_state, inputs["attention_mask"]), getattr(output, "router_logits", None)

    def forward(self, inputs: dict[str, torch.Tensor], output_router_logits: bool = False
                ) -> tuple[torch.Tensor, Any]:
        hidden, routers = self.represent(inputs, output_router_logits)
        return self.head(hidden.to(self.head.weight.dtype)), routers


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
        self.state_serialization = model_config.get("state_serialization", "legacy")
        self.pooling = model_config.get("pooling", "last")
        self.head_input_normalization = model_config.get("head_input_normalization", "none")
        if self.state_serialization not in {"legacy", "priority_v1"}:
            raise ValueError("state_serialization must be legacy or priority_v1")
        if self.pooling not in {"last", "mean"}:
            raise ValueError("pooling must be last or mean")
        if self.head_input_normalization not in {"none", "layer_norm"}:
            raise ValueError("head_input_normalization must be none or layer_norm")
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
        self.context_strategy = ("task/type header capped at 1/3 context; newest serialized execution-state tail uses remainder"
                                 if self.state_serialization == "legacy" else
                                 "priority_v1: complete compact progress/type; full task when it fits, otherwise task prefix; newest sanitized evidence tail")
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
        self.model = HFRoutingModel(backbone, architecture.hidden_size, len(self.catalog), self.pooling,
                                    self.head_input_normalization).to(device=self.device)
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.actual_expert_top_k = int(getattr(architecture, "num_experts_per_tok", 1))
        self._compiled_model: Any = None
        self.compile_status: dict[str, Any] = {"enabled": False, "validated": False}
        self._token_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._token_cache_details: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.last_tokenization_details: list[dict[str, Any]] = []
        self._token_cache_size = int(config.get("inference", {}).get("token_cache_size", 0))
        if self._token_cache_size < 0:
            raise ValueError("token_cache_size must be nonnegative")
        self._token_cache_hits = self._token_cache_misses = 0
        self.source_checkpoint = str(checkpoint) if checkpoint is not None else None
        self._in_place_merged = False
        self._in_place_merge_validated = False
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

    def serialize(self=None, state: ExecutionState | None = None) -> str:
        # Keep the historical HFController.serialize(state) class call usable.
        # Bound instance calls select their checkpoint's serialization version.
        if isinstance(self, ExecutionState) and state is None:
            state, mode = self, "legacy"
        else:
            mode = getattr(self, "state_serialization", "legacy")
        if state is None:
            raise TypeError("serialize requires an ExecutionState")
        if mode == "priority_v1":
            routes = [{key: route[key] for key in ("selected_agents", "execution_mode", "terminate") if key in route}
                      for route in state.previous_routing_decisions[-1:]]
            progress = {"step": state.current_step, "called": list(dict.fromkeys(state.agents_already_called)),
                        "budget": {key: state.remaining_budget[key] for key in ("agent_calls", "tokens") if key in state.remaining_budget},
                        "recent_routes": routes,
                        "answer_present": any(isinstance(output.get("metadata", {}).get("answer"), str)
                                              for output in state.previous_agent_outputs)}
            # Public agent evidence remains available; private labels and run
            # accounting cannot become predictive shortcuts in this version.
            history = _public_evidence({"conversation": state.conversation_state, "tools": state.tool_results,
                                        "outputs": state.previous_agent_outputs})
            return json.dumps({"version": "priority_v1", "progress": progress, "task_type": state.task_type,
                               "user_task": state.user_task, "history": history}, separators=(",", ":"), ensure_ascii=False)
        whole = state.to_dict()
        fields = ("conversation_state", "agents_already_called", "remaining_budget", "current_step",
                  "previous_routing_decisions", "tool_results", "previous_agent_outputs")
        header = f"User task: {state.user_task}\nTask type: {state.task_type}\nExecution state:\n"
        tail = json.dumps({key: whole[key] for key in fields}, separators=(",", ":"), ensure_ascii=False)
        return header + "\0CONDUCTOR_STATE\0" + tail

    def tokenize_serialized(self, texts: list[str]) -> dict[str, torch.Tensor]:
        items = []
        details = []
        reserve = self.tokenizer.num_special_tokens_to_add(pair=False)
        budget = self.max_length - reserve
        for text in texts:
            cached = self._token_cache.get(text)
            if cached is not None:
                self._token_cache_hits += 1
                self._token_cache.move_to_end(text)
                items.append(list(cached))
                details.append(copy.deepcopy(self._token_cache_details.get(text, {"version": "legacy"})))
                continue
            self._token_cache_misses += 1
            if getattr(self, "state_serialization", "legacy") == "priority_v1":
                content, detail = self._priority_tokens(text, budget)
            else:
                header_text, tail_text = text.rsplit("\0CONDUCTOR_STATE\0", 1)
                header = self.tokenizer.encode(header_text, add_special_tokens=False)
                header = header[:max(1, budget // 3)]
                tail = self.tokenizer.encode(tail_text, add_special_tokens=False)
                content = header + tail[-(budget - len(header)):]
                detail = {"version": "legacy"}
            inputs = self.tokenizer.build_inputs_with_special_tokens(content)
            if len(inputs) > self.max_length:
                raise ValueError("tokenizer special-token accounting exceeded max_length")
            items.append(inputs)
            details.append(detail)
            if self._token_cache_size:
                self._token_cache[text] = tuple(inputs)
                self._token_cache_details[text] = copy.deepcopy(detail)
                while len(self._token_cache) > self._token_cache_size:
                    evicted, _ = self._token_cache.popitem(last=False)
                    self._token_cache_details.pop(evicted, None)
        encoded = self.tokenizer.pad({"input_ids": items}, padding=True, return_tensors="pt")
        self._last_encoded_tokens = encoded["attention_mask"].sum(-1).tolist()
        self.last_tokenization_details = details
        return {key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}

    def _priority_tokens(self, text: str, budget: int) -> tuple[list[int], dict[str, Any]]:
        record = json.loads(text)
        if record.get("version") != "priority_v1":
            raise ValueError("priority_v1 tokenizer requires priority_v1 serialization")
        def encode(value: str) -> list[int]:
            return self.tokenizer.encode(value, add_special_tokens=False)
        prefix = encode("Progress: " + json.dumps(record["progress"], separators=(",", ":"), ensure_ascii=False)
                        + "\nTask type: " + record["task_type"] + "\n")
        task_prefix = encode("User task: ")
        evidence_prefix = encode("\nRecent evidence: ")
        remaining = budget - len(prefix) - len(task_prefix) - len(evidence_prefix)
        if remaining < 1:
            # Never silently truncate fields advertised as preserved. A small
            # context or unusually large called/route fields must fail clearly.
            raise ValueError("priority_v1 progress/type fields do not fit max_length; increase model.max_length")
        task = encode(record["user_task"])
        history = encode(json.dumps(record["history"], separators=(",", ":"), ensure_ascii=False))
        task_count = min(len(task), remaining if len(task) <= remaining else max(1, remaining * 2 // 3))
        history_count = min(len(history), remaining - task_count)
        content = prefix + task_prefix + task[:task_count] + evidence_prefix
        if history_count:
            content += history[-history_count:]
        return content, {"version": "priority_v1", "priority_fields": copy.deepcopy(record["progress"]),
                         "task_type": record["task_type"], "priority_tokens": len(prefix),
                         "task_tokens_total": len(task), "task_tokens_kept": task_count,
                         "history_tokens_total": len(history), "history_tokens_kept": history_count,
                         "task_truncated": task_count < len(task), "history_truncated": history_count < len(history)}

    def tokenize_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.tokenize_serialized([self.serialize(state) for state in states])

    def move_inputs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device, non_blocking=self.device.type == "cuda" and value.is_pinned()) for key, value in inputs.items()}

    def encode_states(self, states: list[ExecutionState]) -> dict[str, torch.Tensor]:
        return self.move_inputs(self.tokenize_states(states))

    def token_cache_clear(self) -> None:
        self._token_cache.clear()
        self._token_cache_details.clear()
        self._token_cache_hits = self._token_cache_misses = 0

    def token_cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._token_cache), "max_entries": self._token_cache_size,
                "hits": self._token_cache_hits, "misses": self._token_cache_misses}

    def compile_for_inference(self, backend: str = "inductor", mode: str = "default", fullgraph: bool = False) -> None:
        self._ensure_usable()
        if self.model.training:
            raise ValueError("compile_for_inference requires eval mode")
        self._compiled_model = torch.compile(self.model, backend=backend, mode=mode, fullgraph=fullgraph)
        self.compile_status = {"enabled": True, "validated": False, "backend": backend, "mode": mode, "fullgraph": fullgraph}

    def _ensure_usable(self) -> None:
        if self._in_place_merged and not self._in_place_merge_validated:
            raise RuntimeError("in-place merge is unverified; discard this instance and reload the source checkpoint")

    def enable_training(self, gradient_checkpointing: bool = False) -> None:
        if self._in_place_merged:
            raise ValueError("in-place merged controller is consumed; reload the source adapter checkpoint for training")
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
        self._ensure_usable()
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
        if self._in_place_merged:
            raise ValueError("in-place merged controller cannot save adapters; reload the source adapter checkpoint")
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
                   "state_serialization": self.state_serialization, "pooling": self.pooling,
                   "head_input_normalization": self.head_input_normalization,
                   "method": "constrained categorical routing head with optional coordinator-backbone LoRA"})

    def export_merged(self, path: str | Path, validation_states: list[ExecutionState] | None = None,
                      *, preserve_model: bool = True) -> dict[str, Any]:
        """Self-contained inference artifact; preserve the original source adapter.

        The default clones the backbone, temporarily increasing weight memory.
        preserve_model=False merges the loaded eval controller in place instead:
        it requires a saved adapter checkpoint and consumes this instance's
        adapter training/save capability, including if validation later fails.
        The on-disk source checkpoint remains unchanged. Safe merge still needs
        temporary memory per adapted layer. Publication requires equivalent
        logits, probabilities, and greedy routing actions for every supported
        agent budget on the supplied validation probes. Named FP32 tolerances
        permit logit rtol/atol 1e-4/1e-4 and probability 1e-4/1e-5; both are
        2e-3/2e-3 for reduced precision. Discard the instance after a failed merge
        or validation; inference is blocked.
        """
        self._ensure_usable()
        if self.model.training:
            raise ValueError("merged export requires eval mode")
        target = Path(path).resolve()
        if target.exists():
            raise FileExistsError(f"immutable checkpoint directory already exists: {target}")
        source = resolve_checkpoint(self.source_checkpoint).resolve() if self.source_checkpoint else None
        source_adapter = source / "adapter" if source is not None else None
        if source is not None and (target == source or source in target.parents):
            raise ValueError("merged export must not publish inside its immutable source checkpoint")
        if not preserve_model:
            if (source is None or not (source / "controller.json").is_file()
                    or source_adapter is None or not (source_adapter / "adapter_config.json").is_file()
                    or not any(source_adapter.glob("*.safetensors")) and not any(source_adapter.glob("*.bin"))):
                raise ValueError("in-place export requires an existing immutable source checkpoint with adapter files")
            if self._in_place_merged or not hasattr(self.model.backbone, "merge_and_unload"):
                raise ValueError("in-place export requires a loaded unmerged adapter controller")
        probes = validation_states or [ExecutionState("Calculate 2 plus 3", "math")]
        inputs = self.encode_states(probes)
        with torch.inference_mode():
            before, _ = self.model(inputs, output_router_logits=False)
        if not bool(torch.isfinite(before).all()):
            raise FloatingPointError("source controller produced non-finite validation logits")
        backbone = copy.deepcopy(self.model.backbone) if preserve_model else self.model.backbone
        if not preserve_model:
            # A requested safe merge may partially mutate layers before failing;
            # invalidate adapter training/save and compiled aliases beforehand.
            self._in_place_merged = True
            self._compiled_model = None
            self.compile_status = {"enabled": False, "validated": False}
        if hasattr(backbone, "merge_and_unload"):
            backbone = backbone.merge_and_unload(safe_merge=True)
        if preserve_model:
            merged = HFRoutingModel(backbone, self.model.head.in_features, len(self.catalog), self.pooling,
                                    self.head_input_normalization).to(self.device)
            merged.head.load_state_dict(self.model.head.state_dict())
        else:
            self.model.backbone = backbone
            self.compute_precision = f"{str(self.dtype).removeprefix('torch.')} merged backbone; float32 action head"
            merged = self.model
        merged.eval()
        with torch.inference_mode():
            after, _ = merged(inputs, output_router_logits=False)
        validation = _validate_merged_logits(before, after, self.catalog, self.dtype, self.temperature)
        if not preserve_model:
            self._in_place_merge_validated = True
        configuration = copy.deepcopy(self.config)
        configuration["model"].update(name=str(target / "backbone"), revision="local-export", local_files_only=True,
                                        lora={"enabled": False})
        configuration["model"].pop("resolved_revision", None)
        configuration["model"].pop("local_checkpoint_sha256", None)
        configuration["inference"] = {**configuration.get("inference", {}), "compile": {"enabled": False}}
        metadata = {"backend": "hf", "configuration": configuration, "stage": self.stage, "pretrained": self.pretrained,
                    "format": "merged-hf-v1", "original_base_model": self.base_model_name,
                    "original_resolved_revision": self.resolved_revision, "source_checkpoint": self.source_checkpoint,
                    "preserve_model": preserve_model, "in_place_merge": not preserve_model,
                    "validation_probe_count": len(probes), **validation,
                    "action_catalog_size": len(self.catalog), "token_accounting": self.token_accounting,
                    "context_strategy": self.context_strategy, "max_length": self.max_length,
                    "state_serialization": self.state_serialization, "pooling": self.pooling,
                    "head_input_normalization": self.head_input_normalization}
        with atomic_directory(target) as temporary:
            backbone.save_pretrained(temporary / "backbone", safe_serialization=True)
            self.tokenizer.save_pretrained(temporary / "backbone")
            torch.save(self.model.head.state_dict(), temporary / "head.pt")
            if source_adapter is not None and source_adapter.exists():
                shutil.copytree(source_adapter, temporary / "source_adapter")
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
                if key == "max_length" and config.get("inference", {}).get("preserve_checkpoint_context", False):
                    continue
                if key in overrides.get("model", {}):
                    config["model"][key] = overrides["model"][key]
        return cls(config, checkpoint=path)
