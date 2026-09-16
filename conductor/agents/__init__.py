"""Frozen specialist factories; the coordinator is trained independently."""
from typing import Any
from conductor.agents.base import Agent, CAPABILITIES
from conductor.agents.deterministic import DeterministicAgent
from conductor.schema import AGENT_NAMES


def build_agents(config: dict[str, Any] | None = None) -> dict[str, Agent]:
    config = config or {}
    if "agents" in config:
        config = config["agents"]
    names = config.get("names", list(AGENT_NAMES))
    shared: dict[str, Any] = {}
    agents: dict[str, Agent] = {}
    for name in names:
        if name not in CAPABILITIES:
            raise ValueError(f"unknown agent: {name}")
        settings = {**config, **config.get("overrides", {}).get(name, {})}
        backend = settings.get("backend", "deterministic")
        if backend == "deterministic":
            agents[name] = DeterministicAgent(name, settings.get("token_price_per_million", 0.0))
        elif backend == "hf":
            from conductor.agents.backends import HFAgent
            agents[name] = HFAgent(name, settings, shared)
        elif backend == "api":
            from conductor.agents.backends import APIAgent
            agents[name] = APIAgent(name, settings)
        else:
            raise ValueError(f"unsupported agent backend: {backend}")
    return agents

__all__ = ["Agent", "build_agents"]
