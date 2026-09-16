"""A constrained, learned distribution over complete routing actions.

External agent top-k is distinct from the backbone's internal expert top-k.
No agent name is parsed from generated language, and no heuristic repairs exist.
"""
from __future__ import annotations

from itertools import combinations, permutations
from typing import Any

import torch

from conductor.schema import AGENT_NAMES, RoutingDecision


class ActionCatalog:
    def __init__(self, max_agents: int = 3, agents: tuple[str, ...] = AGENT_NAMES) -> None:
        if not 1 <= max_agents <= len(agents):
            raise ValueError("max_agents must be within the available agent count")
        self.max_agents = max_agents
        self.agents = agents
        self.actions: list[RoutingDecision] = [RoutingDecision([], terminate=True)]
        for count in range(1, max_agents + 1):
            for subset in combinations(agents, count):
                # A single call has no parallel/sequential distinction.
                self.actions.append(RoutingDecision(list(subset), "parallel"))
                if count > 1:
                    self.actions.extend(RoutingDecision(list(order), "sequential") for order in permutations(subset))
        self._ids = {self.key(action): i for i, action in enumerate(self.actions)}

    @staticmethod
    def key(decision: RoutingDecision) -> tuple[Any, ...]:
        mode = "parallel" if len(decision.selected_agents) <= 1 else decision.execution_mode
        agents = tuple(decision.selected_agents) if mode == "sequential" else tuple(sorted(decision.selected_agents))
        return agents, mode, decision.terminate

    def index(self, decision: RoutingDecision | dict[str, Any]) -> int:
        if isinstance(decision, dict):
            decision = RoutingDecision(**decision)
        decision.validate(self.max_agents, self.agents)
        return self._ids[self.key(decision)]

    def mask(self, k: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if not 1 <= k <= self.max_agents:
            raise ValueError(f"k={k} exceeds trained action catalog [1, {self.max_agents}]")
        return torch.tensor([len(action.selected_agents) <= k for action in self.actions], device=device)

    def decision(self, index: int, confidence: float) -> RoutingDecision:
        action = self.actions[index]
        return RoutingDecision(list(action.selected_agents), action.execution_mode, confidence, action.terminate)

    def __len__(self) -> int:
        return len(self.actions)
