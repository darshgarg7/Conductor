"""Direct Preference Optimization for complete categorical routing actions.

The frozen reference is the exact SFT checkpoint. Its log probabilities are
computed before policy updates and cached, avoiding a second resident 7B model.
This is DPO on discrete actions, not a sequence-language imitation loss.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    advantage = (policy_chosen - policy_rejected) - (reference_chosen.detach() - reference_rejected.detach())
    margin = beta * advantage
    return -F.logsigmoid(margin).mean(), margin.detach()
