from __future__ import annotations

import torch
import pytest

from conductor.controller.actions import ActionCatalog
from conductor.controller.factory import build_controller
from conductor.controller.features import StateFeatures
from conductor.schema import ExecutionState, RoutingDecision


def tiny_config() -> dict:
    return {"seed": 42, "model": {"backend": "tiny", "feature_dim": 128, "hidden_dim": 32,
                                  "num_experts": 4, "expert_top_k": 1, "max_agents": 3, "device": "cpu"}}


def test_catalog_preserves_sequential_order_but_parallel_is_a_set() -> None:
    catalog = ActionCatalog(3)
    a, b = ["math", "coder"], ["coder", "math"]
    assert catalog.index(RoutingDecision(a, "sequential")) != catalog.index(RoutingDecision(b, "sequential"))
    assert catalog.index(RoutingDecision(a, "parallel")) == catalog.index(RoutingDecision(b, "parallel"))
    for k in (1, 2, 3):
        for index, enabled in enumerate(catalog.mask(k)):
            if enabled:
                catalog.decision(index, 0.7).validate(k)


def test_tiny_executes_only_selected_internal_expert() -> None:
    controller = build_controller(tiny_config())
    calls = []
    hooks = [expert.register_forward_hook(lambda module, inputs, outputs, expert_id=i: calls.append(expert_id))
             for i, expert in enumerate(controller.model.experts)]
    controller.route(ExecutionState("Add 2 and 7", "math"), 1).validate(1)
    for hook in hooks:
        hook.remove()
    assert len(calls) == 1
    stats = controller.expert_stats()["layers"]["0"]
    assert sum(stats["activation_counts"]) == 1


def test_entire_state_changes_features_without_grader_labels() -> None:
    features = StateFeatures(128)
    base = ExecutionState("Find something", "retrieval")
    changed = ExecutionState("Find something", "retrieval", tool_results=[{"content": "new tool observation xyz"}])
    assert not torch.equal(features.batch([base]), features.batch([changed]))
    assert "expected_answer" not in base.to_dict()


def test_batch_matches_individual_and_checkpoint_roundtrip(tmp_path) -> None:
    controller = build_controller(tiny_config())
    states = [ExecutionState("Calculate 2+7", "math"), ExecutionState("Write a sort", "code")]
    batched = controller.batch_route(states, 2)
    for state, decision in zip(states, batched):
        assert controller.route(state, 2).selected_agents == decision.selected_agents
    controller.save(tmp_path / "checkpoint", "sft")
    restored = build_controller(tiny_config(), str(tmp_path / "checkpoint"))
    torch.testing.assert_close(controller.forward_states(states), restored.forward_states(states))
    assert restored.stage == "sft"
    with pytest.raises(ValueError, match="catalog"):
        restored.route(states[0], 4)


def test_invalid_logits_fail_closed_explicitly() -> None:
    controller = build_controller(tiny_config())
    with torch.no_grad():
        controller.model.head.weight.fill_(float("nan"))
    decision = controller.route(ExecutionState("task", "math"), 2)
    assert decision.terminate and not decision.selected_agents
    assert controller.last_invalid and controller.invalid_decisions == 1


def test_cpu_bfloat16_autocast_is_real() -> None:
    config = tiny_config()
    config["model"]["dtype"] = "bfloat16"
    controller = build_controller(config)
    assert controller.forward_states([ExecutionState("task", "math")]).dtype == torch.bfloat16
