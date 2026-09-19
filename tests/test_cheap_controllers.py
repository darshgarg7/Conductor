from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from conductor.controller.cheap import CheapController, PublicStateFeatures, serialize_public_state
from conductor.schema import ExecutionState, RoutingDecision


def configuration(architecture: str = "linear", action_head: str = "catalog") -> dict:
    return {"model": {"backend": "cheap", "architecture": architecture, "action_head": action_head,
                      "feature_dim": 128, "hidden_dim": 16, "max_agents": 2, "device": "cpu"}}


def test_public_features_ignore_private_provenance_and_execution_telemetry() -> None:
    base = ExecutionState("Look up the rate and calculate the total", "coordination_workflow")
    base.previous_agent_outputs = [{"agent": "retriever", "content": '{"status":"ok","value":7}',
                                    "latency_seconds": 0.1, "tokens": 1, "cost_usd": 0.01}]
    other = copy.deepcopy(base)
    other.task_type = "oracle_required_math"
    other.previous_agent_outputs[0].update({"latency_seconds": 900, "tokens": 10000, "cost_usd": 99,
                                            "metadata": {"expected_answer": "42", "private_grade": 1,
                                                         "required_agents": ["math"], "family": "secret"}})
    # Empty metadata is still a structural input if it exists in one branch;
    # common frozen agents emit the same structure for both candidate routes.
    base.previous_agent_outputs[0]["metadata"] = {}
    assert serialize_public_state(base) == serialize_public_state(other)
    feature = PublicStateFeatures(128)
    assert torch.equal(feature.encode_text(serialize_public_state(base)),
                       feature.encode_text(serialize_public_state(other)))
    other.previous_agent_outputs[0]["content"] = '{"status":"failed","value":7}'
    assert not torch.equal(feature.encode_text(serialize_public_state(base)),
                           feature.encode_text(serialize_public_state(other)))


def test_embedded_json_telemetry_is_removed_and_request_is_retained() -> None:
    state = ExecutionState("Investigate a failed deployment", "oracle")
    state.tool_results = [{"content": json.dumps({"status": "failed", "expected_answer": "hidden",
                                                   "nested": {"latency_seconds": 30, "error": "unavailable"}})}]
    public = serialize_public_state(state)
    assert "expected_answer" not in public and "latency_seconds" not in public and "oracle" not in public
    assert "failed" in public and "unavailable" in public and "deployment" in public


@pytest.mark.parametrize("architecture", ["linear", "mlp"])
def test_learned_dense_control_and_round_trip(tmp_path: Path, architecture: str) -> None:
    torch.manual_seed(42)
    controller = CheapController(configuration(architecture))
    states = [ExecutionState("Retrieve the source", "coordination_workflow"),
              ExecutionState("Verify the artifact", "coordination_workflow", current_step=2)]
    before = controller.forward_states(states, 2)
    target = torch.tensor([controller.catalog.index(RoutingDecision(["retriever"])), 0])
    controller.enable_training()
    optimizer = torch.optim.Adam(controller.model.parameters(), lr=0.02)
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(controller.forward_states(states, 2), target)
        loss.backward()
        optimizer.step()
    after = controller.forward_states(states, 2)
    assert not torch.equal(before, after)
    checkpoint = tmp_path / "checkpoint"
    controller.save(checkpoint, "sft")
    loaded = CheapController.load(checkpoint, {"model": {"feature_dim": 2048}, "inference": {"sample": False}})
    assert loaded.features.dimension == 128 and loaded.stage == "sft"
    torch.testing.assert_close(after, loaded.forward_states(states, 2))
    assert loaded.expert_stats()["available"] is False


@pytest.mark.parametrize("calls,tokens", [(0, 100), (0.5, 100), (10, 0)])
def test_catalog_control_enforces_budget_support(calls: float, tokens: float) -> None:
    controller = CheapController(configuration())
    state = ExecutionState("More work needed", "coordination_workflow",
                           remaining_budget={"agent_calls": calls, "tokens": tokens})
    logits = controller.forward_states([state], 2)
    assert torch.isfinite(logits[0, 0]) and torch.isneginf(logits[0, 1:]).all()
    assert controller.route(state, 2).terminate


def test_single_remaining_call_masks_multi_agent_actions() -> None:
    controller = CheapController(configuration())
    state = ExecutionState("More work needed", "coordination_workflow",
                           remaining_budget={"agent_calls": 1, "tokens": 10})
    logits = controller.forward_states([state], 2)
    for index, action in enumerate(controller.catalog.actions):
        assert bool(torch.isfinite(logits[0, index])) == (len(action.selected_agents) <= 1)


def test_recall_across_rounds_is_not_admission_blocked() -> None:
    controller = CheapController(configuration())
    state = ExecutionState("Retrieve updated evidence", "coordination_workflow", agents_already_called=["retriever"])
    logits = controller.forward_states([state], 2)
    assert torch.isfinite(logits[0, controller.catalog.index(RoutingDecision(["retriever"]))])


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), True])
def test_invalid_public_budget_rejected(budget: float) -> None:
    controller = CheapController(configuration())
    state = ExecutionState("task", "coordination_workflow", remaining_budget={"agent_calls": budget, "tokens": 10})
    with pytest.raises(ValueError, match="finite nonnegative"):
        controller.encode_states([state])


def test_empty_batch_and_public_feature_cache() -> None:
    config = configuration()
    config["inference"] = {"feature_cache_size": 2}
    controller = CheapController(config)
    state = ExecutionState("task", "coordination_workflow")
    controller.batch_route([state, state], 2)
    assert controller.token_cache_stats()["hits"] == 1
    assert controller.batch_route([], 2) == []
    assert controller.last_tokens == 0 and controller.last_batch_tokens == []
    controller.token_cache_clear()
    assert controller.token_cache_stats()["entries"] == 0


def test_factorized_control_uses_same_public_support() -> None:
    controller = CheapController(configuration(action_head="factorized"))
    state = ExecutionState("task", "coordination_workflow", remaining_budget={"agent_calls": 1, "tokens": 10})
    logp = controller.forward_states([state], 2)
    torch.testing.assert_close(logp.exp().sum(-1), torch.ones(1))
    for index, action in enumerate(controller.catalog.actions):
        assert bool(torch.isfinite(logp[0, index])) == (len(action.selected_agents) <= 1)

