from __future__ import annotations

import copy

import pytest

from conductor.datasets.integrity import seal
from conductor.routing.serialization import serialize_state
from conductor.schema import ExecutionState, RoutingDecision
from conductor.training.curate import curate_preferences


def pair(rejected: str = "planner", rejected_success: bool = True, reward: float = .8) -> dict:
    state = ExecutionState("Compute 2 + 3.", "math")
    return {"task_id": "a", "split": "train", "state": state.to_dict(),
            "state_canonical": serialize_state(state), "chosen": RoutingDecision(["math"]).to_dict(),
            "rejected": RoutingDecision([rejected]).to_dict(),
            "chosen_success": True, "rejected_success": rejected_success,
            "chosen_reward": .9, "rejected_reward": reward,
            "chosen_metrics": {"grader_score": 1.0},
            "rejected_metrics": {"grader_score": float(rejected_success)},
            "comparison": "identical_state_counterfactual_first_action",
            "continuation_configuration_sha256": "shared"}


def test_one_winner_per_state_retains_cost_and_failure_pairs() -> None:
    records = [pair("planner", reward=.85), pair("critic", reward=.84), pair("coder", False, .1)]
    sft, preferences, audit = curate_preferences(records)
    assert len(sft) == 1 and len(preferences) == 2
    assert sft[0]["decision"]["selected_agents"] == ["math"]
    assert [item["rejected"]["selected_agents"] for item in preferences] == [["planner"], ["coder"]]
    assert audit["selection_uses_heldout_labels"] is False
    assert "chosen_final_answer" not in sft[0]["state"]


@pytest.mark.parametrize("field,value", [
    ("split", "eval"), ("chosen_success", False), ("chosen_reward", float("nan")),
    ("comparison", "different_states"), ("state_canonical", "mismatch"),
    ("chosen_reward", .2), ("task_id", ""),
])
def test_invalid_or_leaking_sources_fail(field: str, value: object) -> None:
    record = pair()
    record[field] = value
    with pytest.raises(ValueError):
        curate_preferences([record])


def test_refuses_changed_source_checksum_or_mixed_continuation() -> None:
    record = seal(pair(), "source")
    record["chosen_reward"] = .95
    with pytest.raises(ValueError, match="checksum"):
        curate_preferences([record])
    other = pair("critic")
    other["continuation_configuration_sha256"] = "different"
    with pytest.raises(ValueError, match="continuation"):
        curate_preferences([pair(), other])


def test_conflicting_winners_resolved_by_measured_reward() -> None:
    lower = pair("math", reward=.6)
    lower["chosen"] = RoutingDecision(["tool_executor"]).to_dict()
    lower["chosen_reward"] = .7
    sft, preferences, audit = curate_preferences([lower, pair()])
    assert sft[0]["decision"]["selected_agents"] == ["math"]
    assert len(preferences) == 1
    assert audit["states_with_conflicting_source_winners"] == 1


def test_same_categorical_action_is_not_a_preference() -> None:
    record = pair("math")
    record["rejected"]["execution_mode"] = "sequential"
    with pytest.raises(ValueError, match="categorical actions"):
        curate_preferences([record])


def test_output_does_not_mutate_source() -> None:
    source = pair()
    original = copy.deepcopy(source)
    curate_preferences([source])
    assert source == original


def test_mixed_generation_or_reward_identity_is_rejected() -> None:
    first, second = pair(), pair("critic")
    for record in (first, second):
        for field in ("chosen_metrics", "rejected_metrics"):
            record[field]["source_identity"] = {"configuration_sha256": "one"}
    second["chosen_metrics"]["source_identity"] = {"configuration_sha256": "two"}
    with pytest.raises(ValueError, match="source configuration"):
        curate_preferences([first, second])
    second["rejected_metrics"]["source_identity"] = {"configuration_sha256": "two"}
    with pytest.raises(ValueError, match="source configuration"):
        curate_preferences([first, second])
