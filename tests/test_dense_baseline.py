"""Dense controls remain budgeted and never consult private answer labels."""
from __future__ import annotations

import asyncio
import copy

import pytest

from conductor.agents import build_agents
from conductor.orchestration.runner import run_trajectory
from conductor.routing.policies import AllAgentPolicy, build_policy
from conductor.schema import AGENT_NAMES, ExecutionState, Task


def test_historical_all_agent_default_is_one_parallel_round() -> None:
    policy = build_policy("all_agent", {})
    first = policy.route(ExecutionState("public task", "math"), 1)
    assert first.selected_agents == list(AGENT_NAMES) and first.execution_mode == "parallel"
    assert policy.route(ExecutionState("public task", "math", current_step=1), 1).terminate


def test_two_sequential_dense_rounds_repeat_frozen_specialists() -> None:
    policy = build_policy("all_agent", {"all_agent": {"rounds": 2, "execution_mode": "sequential"}})
    for step in (0, 1):
        state = ExecutionState("public task", "math", current_step=step, agents_already_called=list(AGENT_NAMES))
        decision = policy.route(state, 1)
        assert decision.selected_agents == list(AGENT_NAMES)
        assert decision.execution_mode == "sequential" and not decision.terminate
        decision.validate(len(AGENT_NAMES))
    assert policy.route(ExecutionState("public task", "math", current_step=2), 1).terminate


@pytest.mark.parametrize("rounds", [0, -1, True, 2.0, "2"])
def test_invalid_dense_rounds_rejected(rounds) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AllAgentPolicy(rounds=rounds)


def test_invalid_dense_execution_mode_rejected() -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        build_policy("all_agent", {"all_agent": {"execution_mode": "adaptive"}})


def test_dense_second_pass_handles_dependency_with_same_call_budget() -> None:
    task = Task("dense-dependency", "Compute 12 + 66. then reverse the digits of the result.",
                "composed_math_string", expected_answer="87")
    configured = build_policy("all_agent", {"all_agent": {"rounds": 2, "execution_mode": "sequential"}})
    trajectory = asyncio.run(run_trajectory(task, configured, build_agents(), k=2, max_rounds=3,
                                            token_budget=100000, agent_call_budget=12))
    assert trajectory.task_success and trajectory.final_answer == "87"
    assert trajectory.metadata["agent_activations"] == 12
    assert [len(step.agent_outputs) for step in trajectory.steps] == [8, 4]
    assert trajectory.metadata["budget_clipped_steps"] == [1]
    assert trajectory.steps[1].decision["selected_agents"] == list(AGENT_NAMES[:4])
    assert trajectory.metadata["requested_routing_decisions"][1]["selected_agents"] == list(AGENT_NAMES)
    assert trajectory.metadata["stop_reason"] == "budget_exhausted"
    supplied_to_coder = trajectory.steps[1].state["previous_agent_outputs"]
    assert any(output["agent"] == "math" and output["metadata"]["answer"] == "78" for output in supplied_to_coder)


def test_dense_policy_and_specialists_cannot_see_grader_or_private_key() -> None:
    seen = []
    class RecordingDense(AllAgentPolicy):
        def route(self, state, k):
            seen.append(copy.deepcopy(state.to_dict()))
            return super().route(state, k)
    task = Task("dense-private", "Compute 12 + 66. then reverse the digits of the result.",
                "composed_math_string", expected_answer="PRIVATE WRONG GRADER", metadata={"secret": "never expose"})
    trajectory = asyncio.run(run_trajectory(task, RecordingDense(rounds=2, execution_mode="sequential"),
                                            build_agents(), k=2, max_rounds=3,
                                            token_budget=100000, agent_call_budget=12))
    assert trajectory.final_answer == "87" and not trajectory.task_success
    assert len(seen) == 2
    assert all(set(state) == set(ExecutionState("", "").to_dict()) for state in seen)
    assert all("PRIVATE WRONG GRADER" not in str(state) and "never expose" not in str(state) for state in seen)


def test_fixed_capability_order_handles_heldout_dependencies_without_category_branches():
    from conductor.datasets.tasks import make_tasks
    order = ['planner', 'retriever', 'researcher', 'math', 'coder', 'tool_executor', 'critic', 'verifier']
    policy = build_policy('all_agent', {'all_agent': {'order': order, 'execution_mode': 'sequential'}})
    for task in make_tasks(train_count=0, eval_count=6):
        trajectory = asyncio.run(run_trajectory(task, policy, build_agents(), k=2, agent_call_budget=12))
        assert trajectory.task_success
        assert len(trajectory.steps[0].agent_outputs) == 8
    with pytest.raises(ValueError, match='exactly once'):
        AllAgentPolicy(order=['planner'] * 8)
