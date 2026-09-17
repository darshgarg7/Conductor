"""Synthetic support fixture mechanics; not enterprise or GPU evidence."""
from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import FrozenInstanceError, replace

import pytest

from conductor.demos.support import (RUNBOOKS, SupportRulePolicy, build_support_agents,
                                     check_support_answer, support_tasks)
from conductor.orchestration.runner import initial_state, run_trajectory
from conductor.schema import AGENT_NAMES, RoutingDecision


def _outputs(trajectory):
    return [output for step in trajectory.steps for output in step.agent_outputs]


@pytest.mark.parametrize("k", [1, 2])
def test_support_rule_runs_every_fixed_fixture_with_sparse_dependency_order(k):
    agents = build_support_agents()
    assert tuple(agents) == AGENT_NAMES and all(agent.frozen for agent in agents.values())
    assert all(agent.capability for agent in agents.values())
    with pytest.raises(FrozenInstanceError):
        agents["researcher"].frozen = False
    tasks = support_tasks()
    assert len(tasks) == 12 and all(task.split == "test" for task in tasks)
    for task in tasks:
        trajectory = asyncio.run(run_trajectory(task, SupportRulePolicy(), agents, k=k, max_rounds=3))
        assert trajectory.task_success
        assert [output["agent"] for output in _outputs(trajectory)] == ["retriever", "researcher"]
        assert all(len(step.decision["selected_agents"]) <= k for step in trajectory.steps)
        assert trajectory.steps[-1].decision["terminate"]
        assert check_support_answer(task, trajectory.final_answer, _outputs(trajectory))["contract_valid"]
        assert trajectory.metadata["specialist_audit"]["all_frozen_llm"] is False


def test_research_requires_retrieval_and_parallel_execution_cannot_hide_the_dependency():
    class ParallelFirst:
        name, last_tokens, last_cost_usd = "support_parallel_first", 0, 0.0
        def route(self, state, k):
            return (RoutingDecision(["retriever", "researcher"], "parallel") if not state.current_step
                    else RoutingDecision([], terminate=True))

    task = support_tasks()[0]
    agents = build_support_agents()
    alone = asyncio.run(agents["researcher"].execute(initial_state(task)))
    assert alone.metadata["answer"] is None
    parallel = asyncio.run(run_trajectory(task, ParallelFirst(), agents, k=2))
    assert not parallel.task_success and parallel.final_answer == ""
    sequential = asyncio.run(run_trajectory(task, SupportRulePolicy(), agents, k=2))
    assert sequential.task_success


def test_public_observations_determine_the_answer_independent_of_task_identity_and_private_labels():
    original = support_tasks()[10]  # Historical error keywords with a successful current check.
    hidden = replace(original, id="PRIVATE_TASK_ID", expected_answer="PRIVATE_WRONG_LABEL", metadata={"private": "PRIVATE_METADATA"})
    trajectory = asyncio.run(run_trajectory(hidden, SupportRulePolicy(), build_support_agents()))
    assert trajectory.final_answer == original.expected_answer and not trajectory.task_success
    assert check_support_answer(hidden, trajectory.final_answer, _outputs(trajectory))["contract_valid"]
    for step in trajectory.steps:
        state = json.dumps(step.state)
        assert all(value not in state for value in ("PRIVATE_TASK_ID", "PRIVATE_WRONG_LABEL", "PRIVATE_METADATA"))
        assert "expected_answer" not in step.state
    public = json.loads(original.user_task)
    for observation in public["observations"]:
        if observation["kind"] == "runtime_error":
            observation["value"] = "cuda_out_of_memory"
        if observation["kind"] == "current_validation_success":
            observation["value"] = False
    changed = replace(original, user_task=json.dumps(public))
    changed_trajectory = asyncio.run(run_trajectory(changed, SupportRulePolicy(), build_support_agents()))
    assert json.loads(changed_trajectory.final_answer)["diagnosis"] == "unknown_escalate"
    assert not changed_trajectory.task_success


def test_grounding_contract_rejects_fabricated_evidence_unsupported_diagnosis_and_write_actions():
    task = support_tasks()[0]
    trajectory = asyncio.run(run_trajectory(task, SupportRulePolicy(), build_support_agents()))
    outputs = _outputs(trajectory)
    candidate = json.loads(trajectory.final_answer)
    fabricated = {**candidate, "evidence": candidate["evidence"] + ["invented-gpu-probe"]}
    checked = check_support_answer(task, json.dumps(fabricated), outputs)
    assert checked["schema_valid"] and not checked["evidence_grounded"] and not checked["contract_valid"]
    unsupported = {**candidate, "diagnosis": "driver_below_runtime_minimum"}
    assert not check_support_answer(task, json.dumps(unsupported), outputs)["diagnosis_grounded"]
    unsafe = {**candidate, "next_action": "sudo reboot"}
    assert not check_support_answer(task, json.dumps(unsafe), outputs)["read_only_next_action"]
    executed = [{**output, "metadata": {**output["metadata"], "commands_executed": True}} for output in outputs]
    assert not check_support_answer(task, trajectory.final_answer, executed)["commands_not_executed"]


def test_public_runbooks_cite_primary_sources_and_tools_do_not_execute_commands(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("support tools must never execute external commands")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert all(book.sources and all(url.startswith("https://docs.nvidia.com/") for url in book.sources) for book in RUNBOOKS)
    task = support_tasks()[5]
    for agent in build_support_agents().values():
        output = asyncio.run(agent.execute(initial_state(task)))
        assert output.metadata["commands_executed"] is False
        assert output.metadata["backend"] == "deterministic"
        assert output.metadata["token_accounting"] == "estimated_whitespace"
        assert output.cost_usd == 0.0
