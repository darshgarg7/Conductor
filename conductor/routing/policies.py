"""Orchestration baselines; model policies are implemented independently."""
from __future__ import annotations

import json
import random
from typing import Any

from conductor.schema import AGENT_NAMES, ExecutionState, Policy, RoutingDecision
from conductor.routing.serialization import serialize_state


class Baseline:
    last_tokens = 0
    last_cost_usd = 0.0


class AllAgentPolicy(Baseline):
    name = "all_agent"
    def __init__(self, available: tuple[str, ...] = AGENT_NAMES) -> None:
        self.available = available

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        if state.current_step > 0:
            return RoutingDecision([], terminate=True)
        return RoutingDecision(list(self.available), "parallel")


class RuleBasedPolicy(Baseline):
    name = "rule_based"
    DEFAULT_ROUTES = {
        "arithmetic": ["math"], "math": ["math"], "lookup": ["retriever"],
        "retrieval": ["retriever"], "string_transform": ["coder"], "code": ["coder"],
        "research": ["retriever", "researcher"], "reasoning": ["planner", "math"],
        "composed": ["retriever", "math", "verifier"],
        "composed_lookup_math": ["retriever", "math", "verifier"],
        "composed_retrieval_math": ["retriever", "math"],
        "composed_math_string": ["math", "coder"],
    }

    def __init__(self, routes: dict[str, list[str]] | None = None, available: tuple[str, ...] = AGENT_NAMES) -> None:
        self.routes = {**self.DEFAULT_ROUTES, **(routes or {})}
        self.available = available

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        needed = self.routes.get(state.task_type, ["planner", "researcher", "verifier"])
        selected = [name for name in needed if name not in state.agents_already_called and name in self.available][:k]
        if not selected:
            return RoutingDecision([], terminate=True)
        # Dependent specialist outputs must be available to the next agent.
        return RoutingDecision(selected, "sequential").validate(k)


class RandomTopKPolicy(Baseline):
    name = "random_top_k"
    def __init__(self, seed: int = 42, rounds: int = 3, available: tuple[str, ...] = AGENT_NAMES) -> None:
        self.rng = random.Random(seed)
        self.rounds = rounds
        self.available = available

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        available = [name for name in self.available if name not in state.agents_already_called]
        if state.current_step >= self.rounds or not available:
            return RoutingDecision([], terminate=True)
        return RoutingDecision(self.rng.sample(available, min(k, len(available))), "parallel").validate(k)


class StaticSupervisorPolicy(Baseline):
    """Frozen prompting-only LLM, never replaced by a rule baseline."""
    name = "static_supervisor"

    def __init__(self, config: dict[str, Any]) -> None:
        if not config.get("name"):
            raise ValueError("static_supervisor unavailable: configure supervisor.name for a frozen HF LLM")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = config.get("device", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config["name"], revision=config.get("revision", "main"))
        self.model = AutoModelForCausalLM.from_pretrained(config["name"], revision=config.get("revision", "main"))
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.max_new_tokens = int(config.get("max_new_tokens", 128))
        self.max_context = int(config.get("max_context", 2048))
        self.rate = float(config.get("cost_per_million_tokens", 0))
        self._torch = torch

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        prompt = (
            f"Select at most {k} specialists from {list(AGENT_NAMES)}. "
            "Return only JSON with selected_agents (array), execution_mode (parallel or sequential), "
            "confidence (0..1), terminate (boolean). Termination selects no agents.\n"
            f"Execution state: {serialize_state(state)}\nDecision: "
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context).to(self.device)
        with self._torch.inference_mode():
            result = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                         pad_token_id=self.tokenizer.eos_token_id)
        self.last_tokens = int(inputs["attention_mask"].sum()) + result.shape[1] - inputs["input_ids"].shape[1]
        self.last_cost_usd = self.last_tokens * self.rate / 1_000_000
        output = self.tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        start, end = output.find("{"), output.rfind("}")
        if start < 0 or end < start:
            raise ValueError("static supervisor produced no valid JSON")
        decision = RoutingDecision(**json.loads(output[start:end + 1]))
        return decision.validate(k)


def build_policy(name: str, config: dict[str, Any], checkpoint: str | None = None) -> Policy:
    normalized = name.lower().replace("-", "_")
    available = tuple(config.get("agents", {}).get("names", AGENT_NAMES))
    if normalized == "all_agent":
        return AllAgentPolicy(available)
    if normalized == "rule_based":
        return RuleBasedPolicy(config.get("routing", {}).get("rules"), available)
    if normalized == "random_top_k":
        return RandomTopKPolicy(int(config.get("seed", 42)), int(config.get("orchestration", {}).get("max_rounds", 3)), available)
    if normalized == "static_supervisor":
        return StaticSupervisorPolicy(config.get("supervisor", {}))
    if normalized in {"base_moe", "conductor_sft", "conductor_preference"}:
        from conductor.controller.factory import build_controller
        if normalized == "base_moe" and checkpoint is not None:
            raise ValueError("base_moe must be evaluated before post-training, without a trained checkpoint")
        if normalized != "base_moe" and checkpoint is None:
            checkpoint = config.get("checkpoints", {}).get(normalized)
            if not checkpoint:
                raise ValueError(f"{normalized} requires a checkpoint")
        policy = build_controller(config, checkpoint)
        expected_stage = {"conductor_sft": "sft", "conductor_preference": "preference"}.get(normalized)
        if expected_stage and getattr(policy, "stage", None) != expected_stage:
            raise ValueError(f"{normalized} requires a {expected_stage} checkpoint; got {getattr(policy, 'stage', None)}")
        policy.name = normalized
        return policy
    raise ValueError(f"unknown policy: {name}")
