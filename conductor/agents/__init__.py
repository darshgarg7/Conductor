"""Frozen specialist factories; the coordinator is trained independently."""
import json
from pathlib import Path
from typing import Any
import asyncio
from conductor.agents.base import Agent, CAPABILITIES
from conductor.agents.deterministic import DeterministicAgent
from conductor.schema import AGENT_NAMES


def build_agents(config: dict[str, Any] | None = None) -> dict[str, Agent]:
    config = config or {}
    if "agents" in config:
        config = config["agents"]
    if config.get("backend") == "workflow":
        from conductor.coordination.specialists import build_workflow_agents
        from conductor.coordination.workloads import PublicStores
        stores = config.get("stores")
        if stores is None and config.get("stores_path"):
            stores = json.loads(Path(config["stores_path"]).read_text())
        if stores is None:
            raise ValueError("workflow agents require agents.stores or agents.stores_path")
        return build_workflow_agents(PublicStores.from_dict(stores))
    names = config.get("names", list(AGENT_NAMES))
    shared: dict[str, Any] = {}
    limits: dict[str, asyncio.Semaphore] = {}
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
    if config.get("limits_enabled", False) or config.get("backend", "deterministic") != "deterministic":
        from conductor.agents.limits import LimitedAgent
        for name, agent in list(agents.items()):
            settings = {**config, **config.get("overrides", {}).get(name, {})}
            key = str((settings.get("backend"), settings.get("model_name"), settings.get("revision"), settings.get("base_url")))
            maximum = int(settings.get("model_max_concurrency", 1))
            if maximum < 1:
                raise ValueError("model_max_concurrency must be positive")
            limits.setdefault(key, asyncio.Semaphore(maximum))
            agents[name] = LimitedAgent(agent, settings, limits[key])
    return agents

__all__ = ["Agent", "build_agents"]
