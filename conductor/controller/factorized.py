"""Normalized joint routing likelihoods, not independent agent/BCE scores.

Each complete action has one stop/count/mode/selection path. Parallel subsets
use increasing configured agent order; sequential paths preserve order. Exact
catalog enumeration is the current inference/diagnostic interface, not a claim
of asymptotically cheap decoding. Shared prefixes execute in three batched
conditional steps for max_agents=3, rather than one model pass per action.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from conductor.controller.actions import ActionCatalog
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Unreachable prefixes use a safe dummy support, never NaN all-masked softmax.

    Complete-action legality removes those unreachable branches later. This
    fallback is not a routing-policy repair or an extra probability-bearing path.
    """
    safe = mask.clone()
    safe[..., 0] |= ~safe.any(-1)
    return F.log_softmax(logits.float().masked_fill(~safe, float("-inf")), -1)


class FactorizedHead(nn.Module):
    version = "factorized-v1"
    mask_version = "public-budget-admission-parallel-feasibility-v1"

    def __init__(self, feature_dim: int, max_agents: int = 3,
                 agents: tuple[str, ...] = AGENT_NAMES, hidden_dim: int = 64) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not agents or len(set(agents)) != len(agents) or any(not isinstance(agent, str) for agent in agents):
            raise ValueError("agents must be nonempty unique names")
        if not 1 <= max_agents <= min(3, len(agents)):
            raise ValueError("factorized max_agents must be in [1, min(3, number of agents)]")
        self.in_features = feature_dim
        self.hidden_dim = hidden_dim
        self.catalog = ActionCatalog(max_agents, agents)
        self.out_features = len(self.catalog)
        self.max_agents = max_agents
        self.agents = agents
        self.context = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.stop_head = nn.Linear(hidden_dim, 2)  # index zero=stop, one=continue
        self.count_head = nn.Linear(hidden_dim, max_agents)
        self.count_embedding = nn.Embedding(max_agents, hidden_dim)
        self.mode_embedding = nn.Embedding(2, hidden_dim)  # zero=parallel
        self.mode_head = nn.Linear(hidden_dim, 2)
        self.agent_embedding = nn.Embedding(len(agents), hidden_dim)
        self.prefix_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.agent_head = nn.Linear(hidden_dim, len(agents))
        self._build_paths()

    def _buffer(self, name: str, values: Any, dtype: torch.dtype = torch.long) -> None:
        # Derived catalog tables are reconstructed from versioned head metadata.
        self.register_buffer(name, torch.tensor(values, dtype=dtype), persistent=False)

    def _build_paths(self) -> None:
        ranks = {agent: index for index, agent in enumerate(self.agents)}
        actions = self.catalog.actions[1:]
        counts = [len(action.selected_agents) for action in actions]
        modes = [int(action.execution_mode == "sequential") for action in actions]
        paths = [tuple(ranks[agent] for agent in action.selected_agents) for action in actions]
        self._buffer("action_counts", counts)
        self._buffer("action_modes", modes)
        self._buffer("action_members", [[index in path for index in range(len(self.agents))] for path in paths], torch.bool)
        self._node_counts: list[int] = []
        previous: dict[tuple[int, int, tuple[int, ...]], int] = {}
        for depth in range(self.max_agents):
            nodes: dict[tuple[int, int, tuple[int, ...]], int] = {}
            path_nodes, chosen = [], []
            for count, mode, path in zip(counts, modes, paths):
                if count <= depth:
                    path_nodes.append(0)
                    chosen.append(0)
                    continue
                key = (count, mode, path[:depth])
                nodes.setdefault(key, len(nodes))
                path_nodes.append(nodes[key])
                chosen.append(path[depth])
            keys = list(nodes)
            self._node_counts.append(len(keys))
            self._buffer(f"node_counts_{depth}", [key[0] for key in keys])
            self._buffer(f"node_modes_{depth}", [key[1] for key in keys])
            self._buffer(f"node_last_{depth}", [key[2][-1] if depth else -1 for key in keys])
            self._buffer(f"node_members_{depth}", [[index in key[2] for index in range(len(self.agents))] for key in keys], torch.bool)
            self._buffer(f"path_nodes_{depth}", path_nodes)
            self._buffer(f"path_agents_{depth}", chosen)
            if depth:
                self._buffer(f"node_parents_{depth}", [previous[(c, mode, prefix[:-1])] for c, mode, prefix in keys])
            previous = nodes

    def metadata(self) -> dict[str, Any]:
        order = list(self.agents)
        return {"head_type": "factorized", "head_version": self.version, "mask_version": self.mask_version,
                "agent_order": order, "agent_order_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
                "max_agents": self.max_agents, "in_features": self.in_features, "hidden_dim": self.hidden_dim,
                "action_catalog_size": len(self.catalog), "shared_prefix_nodes_per_depth": list(self._node_counts),
                "factorization": "stop -> count -> mode if count>1 -> agents without replacement",
                "parallel": "increasing agent order with remaining-admissible-capacity masks",
                "sequential": "ordered; within-round duplicates masked; prior-round calls remain admissible",
                "inference": "exact complete-catalog argmax/sampling; not a scalability benchmark"}

    def _constraints(self, features: torch.Tensor, states: Sequence[ExecutionState] | None, k: int | None,
                     call_budgets: torch.Tensor | None, token_budgets: torch.Tensor | None,
                     admissible_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError("features must have shape [batch, in_features]")
        batch = features.shape[0]
        k = self.max_agents if k is None else k
        if type(k) is not int or not 1 <= k <= self.max_agents:
            raise ValueError("k exceeds the configured factorized action catalog")
        if states is not None and len(states) != batch:
            raise ValueError("states must match feature batch length")
        if call_budgets is None:
            call_budgets = ([state.remaining_budget.get("agent_calls", 0) for state in states]
                            if states is not None else [self.max_agents] * batch)
        if token_budgets is None:
            token_budgets = ([state.remaining_budget.get("tokens", 0) for state in states]
                             if states is not None else [1] * batch)
        calls = torch.as_tensor(call_budgets, device=features.device, dtype=torch.float32)
        tokens = torch.as_tensor(token_budgets, device=features.device, dtype=torch.float32)
        if calls.shape != (batch,) or tokens.shape != (batch,):
            raise ValueError("call/token budgets must have shape [batch]")
        if not bool(torch.isfinite(calls).all() & torch.isfinite(tokens).all()) or bool((calls < 0).any() | (tokens < 0).any()):
            raise ValueError("call/token budgets must be finite and nonnegative")
        if admissible_mask is None:
            available = torch.ones(batch, len(self.agents), device=features.device, dtype=torch.bool)
        else:
            available = torch.as_tensor(admissible_mask, device=features.device)
            if available.dtype != torch.bool or available.shape != (batch, len(self.agents)):
                raise ValueError("admissible_mask must be bool [batch, number of agents]")
        capacity = torch.minimum(calls.clamp(max=k).floor().long(), available.sum(-1))
        capacity = capacity.masked_fill(tokens < 1, 0)
        return available, capacity

    def all_log_probabilities(self, features: torch.Tensor, states: Sequence[ExecutionState] | None = None,
                              k: int | None = None, *, call_budgets: torch.Tensor | None = None,
                              token_budgets: torch.Tensor | None = None,
                              admissible_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Exact normalized joint log probabilities; illegal complete actions are -inf.

        Tensor budgets can replace state budget extraction for HF/DDP inputs.
        Admission is public and does not remove agents called in previous rounds.
        Positive tokens permit routing, not a guarantee of future generation cost.
        """
        available, capacity = self._constraints(features, states, k, call_budgets, token_budgets, admissible_mask)
        context = self.context(features)
        batch = len(features)
        stop_mask = torch.stack((torch.ones_like(capacity, dtype=torch.bool), capacity > 0), -1)
        stop = _masked_log_softmax(self.stop_head(context), stop_mask)
        count_mask = torch.arange(1, self.max_agents + 1, device=features.device)[None, :] <= capacity[:, None]
        count = _masked_log_softmax(self.count_head(context), count_mask)
        count_context = context[:, None, :] + self.count_embedding.weight[None, :, :]
        mode_mask = torch.ones(batch, self.max_agents, 2, device=features.device, dtype=torch.bool)
        mode_mask[:, 0, 1] = False
        mode = _masked_log_softmax(self.mode_head(torch.tanh(count_context)), mode_mask)
        action_count = self.action_counts - 1
        joint = stop[:, 1, None] + count[:, action_count] + mode[:, action_count, self.action_modes]
        # Number of available ranks strictly greater than each potential choice.
        after = available.long().flip(-1).cumsum(-1).flip(-1) - available.long()
        ranks = torch.arange(len(self.agents), device=features.device)
        previous_hidden = None
        for depth, nodes in enumerate(self._node_counts):
            node_count = getattr(self, f"node_counts_{depth}")
            node_mode = getattr(self, f"node_modes_{depth}")
            last = getattr(self, f"node_last_{depth}")
            members = getattr(self, f"node_members_{depth}")
            if depth == 0:
                hidden = context[:, None, :] + self.count_embedding(node_count - 1)[None, :, :] + self.mode_embedding(node_mode)[None, :, :]
            else:
                parent = previous_hidden[:, getattr(self, f"node_parents_{depth}"), :]
                embedding = self.agent_embedding(last)[None, :, :].expand(batch, nodes, self.hidden_dim)
                hidden = self.prefix_cell(embedding.reshape(-1, self.hidden_dim), parent.reshape(-1, self.hidden_dim)).reshape(batch, nodes, self.hidden_dim)
            previous_hidden = hidden
            valid_prefix = ~(members[None, :, :] & ~available[:, None, :]).any(-1)
            mask = available[:, None, :] & ~members[None, :, :]
            parallel = ((ranks[None, None, :] > last[None, :, None])
                        & (after[:, None, :] >= (node_count - depth - 1)[None, :, None]))
            mask = mask & ((node_mode == 1)[None, :, None] | parallel)
            mask = mask & valid_prefix[:, :, None] & (node_count[None, :, None] <= capacity[:, None, None])
            conditional = _masked_log_softmax(self.agent_head(torch.tanh(hidden)), mask)
            path = conditional[:, getattr(self, f"path_nodes_{depth}"), getattr(self, f"path_agents_{depth}")]
            joint = joint + torch.where(self.action_counts[None, :] > depth, path, torch.zeros_like(path))
        legal = ((self.action_counts[None, :] <= capacity[:, None])
                 & ~(self.action_members[None, :, :] & ~available[:, None, :]).any(-1))
        return torch.cat((stop[:, :1], joint.masked_fill(~legal, float("-inf"))), -1)

    def forward(self, features: torch.Tensor, states: Sequence[ExecutionState] | None = None,
                k: int | None = None, **constraints: Any) -> torch.Tensor:
        return self.all_log_probabilities(features, states, k, **constraints)

    def log_prob(self, features: torch.Tensor, states: Sequence[ExecutionState] | None,
                 decisions: Sequence[RoutingDecision | dict[str, Any]], k: int | None = None,
                 **constraints: Any) -> torch.Tensor:
        """Teacher-forced complete-action log likelihood, with canonical subsets.

        Currently shares exact catalog computation; no claim of cheaper selected
        likelihood evaluation. Invalid budget/admission targets fail explicitly.
        """
        if len(decisions) != len(features):
            raise ValueError("decisions must match feature batch length")
        indices = torch.tensor([self.catalog.index(decision) for decision in decisions], device=features.device)
        probabilities = self.all_log_probabilities(features, states, k, **constraints)
        result = probabilities.gather(1, indices[:, None]).squeeze(1)
        if bool(torch.isneginf(result).any()):
            raise ValueError("target decision is illegal for k, budget, or public admission")
        return result

    def greedy(self, features: torch.Tensor, states: Sequence[ExecutionState] | None = None,
               k: int | None = None, **constraints: Any) -> list[RoutingDecision]:
        logps = self.all_log_probabilities(features, states, k, **constraints)
        indices = logps.argmax(-1)
        return [self.catalog.decision(int(index), float(row[index].detach().exp())) for row, index in zip(logps, indices)]

    def draw(self, features: torch.Tensor, states: Sequence[ExecutionState] | None = None,
             k: int | None = None, *, generator: torch.Generator | None = None,
             **constraints: Any) -> list[RoutingDecision]:
        logps = self.all_log_probabilities(features, states, k, **constraints)
        indices = torch.multinomial(logps.exp(), 1, generator=generator).squeeze(1)
        return [self.catalog.decision(int(index), float(row[index].detach().exp())) for row, index in zip(logps, indices)]
