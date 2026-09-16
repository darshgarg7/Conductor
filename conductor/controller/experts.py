"""Actual internal-expert statistics; agent activations are measured elsewhere."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch


class ExpertTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.loads: dict[str, torch.Tensor] = {}
        self.task_loads: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self.entropy_sum: dict[str, float] = defaultdict(float)
        self.observations: dict[str, int] = defaultdict(int)

    def add(self, logits: torch.Tensor, top_k: int, task_types: list[str], layer: str = "0",
            attention_mask: torch.Tensor | None = None) -> None:
        """logits shape BxEx or BxLxEx; padding excluded from token-level stats."""
        values = logits.detach().float().cpu()
        if values.ndim == 2:
            values = values[:, None, :]
        if values.shape[0] != len(task_types):
            raise ValueError("expert stats batch/task mismatch")
        masks = (attention_mask.detach().cpu().bool() if attention_mask is not None
                 else torch.ones(values.shape[:2], dtype=torch.bool))
        experts = values.shape[-1]
        self.loads.setdefault(layer, torch.zeros(experts, dtype=torch.long))
        for sample, task_type in enumerate(task_types):
            real = values[sample][masks[sample]]
            if not len(real):
                continue
            probabilities = real.softmax(-1)
            selected = real.topk(min(top_k, experts), dim=-1).indices.flatten()
            counts = torch.bincount(selected, minlength=experts)
            self.loads[layer] += counts
            self.task_loads[task_type].setdefault(layer, torch.zeros(experts, dtype=torch.long))
            self.task_loads[task_type][layer] += counts
            self.entropy_sum[layer] += float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).sum())
            self.observations[layer] += len(real)

    def export(self) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for layer, counts in self.loads.items():
            total = max(int(counts.sum()), 1)
            frequencies = counts.float() / total
            mean = float(counts.float().mean())
            cv = float(counts.float().std(unbiased=False)) / mean if mean else 0.0
            layers[layer] = {
                "activation_counts": counts.tolist(), "activation_frequency": frequencies.tolist(),
                "utilized_experts": int((counts > 0).sum()), "num_experts": len(counts),
                "routing_entropy": self.entropy_sum[layer] / max(self.observations[layer], 1),
                "normalized_routing_entropy": self.entropy_sum[layer] / max(self.observations[layer], 1) / max(math.log(len(counts)), 1e-12),
                "load_coefficient_of_variation": cv, "observations": self.observations[layer],
            }
        return {"layers": layers, "task_type_activation_counts": {
            task: {layer: counts.tolist() for layer, counts in per_layer.items()}
            for task, per_layer in self.task_loads.items()
        }, "scope": "controller internal MoE experts; tiny observes one gate per state; HF per non-padding token per layer"}
