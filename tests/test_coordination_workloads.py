"""Contract tests for controlled dependency workflows."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from conductor.coordination.grading import grade_workflow
from conductor.coordination.evaluate import evaluate
from conductor.coordination.generate import generate
from conductor.coordination.policies import WorkflowRulePolicy
from conductor.coordination.preferences import _tags, generate_preferences
from conductor.coordination.specialists import build_workflow_agents
from conductor.coordination.workloads import FAMILIES, PublicStores, make_final_workload, make_workload, parse_request
from conductor.orchestration.runner import initial_state, run_trajectory
from conductor.training.runner import train
from conductor.schema import RoutingDecision


def test_balanced_group_disjoint_inventory_and_private_boundary():
    tasks, stores = make_workload()
    assert len(tasks) == 108
    assert {task.task_type for task in tasks} == {"coordination_workflow"}
    assert {task.metadata["family"] for task in tasks} == set(FAMILIES)
    assert len({task.metadata["source_group"] for task in tasks}) == len(tasks)
    assert all("expected_answer" not in parse_request(task.user_task) for task in tasks)
    assert stores.corpus_sha256 == PublicStores.from_dict(stores.to_dict()).corpus_sha256


def test_preference_state_tags_cover_handoffs_and_stopping() -> None:
    tasks, _ = make_workload(train_count=6, dev_count=0)
    state = initial_state(tasks[0])
    state.current_step = 1
    handoff = _tags(state, RoutingDecision(["retriever", "math"], "sequential"))
    stopping = _tags(state, RoutingDecision([], terminate=True))
    assert {"intermediate", "handoff"} <= set(handoff)
    assert {"intermediate", "stopping"} <= set(stopping)


def test_final_inventory_has_new_structures_and_separate_groups():
    development, _ = make_workload()
    final, _ = make_final_workload(seed=271828)
    assert len(final) == 120
    assert sum(task.metadata["structure_ood"] for task in final) == 60
    assert not ({task.metadata["source_group"] for task in development}
                & {task.metadata["source_group"] for task in final})


def test_public_rule_solves_every_known_workflow_with_frozen_specialists():
    tasks, stores = make_workload(train_count=12, dev_count=12)
    agents = build_workflow_agents(stores)
    assert all(agent.frozen for agent in agents.values())

    async def execute():
        return [await run_trajectory(task, WorkflowRulePolicy(), agents, k=3, max_rounds=6,
                                     token_budget=16384, agent_call_budget=12) for task in tasks]

    trajectories = asyncio.run(execute())
    assert all(trajectory.task_success for trajectory in trajectories)
    assert any(any(step.decision["execution_mode"] == "parallel" for step in trajectory.steps)
               for trajectory in trajectories)
    assert any(any(any(artifact["kind"] == "tool_failure"
                       for artifact in output["metadata"].get("artifacts", []))
                   for output in step.agent_outputs)
               for trajectory in trajectories for step in trajectory.steps)


def test_missing_and_reversed_prerequisites_block_without_private_answer():
    tasks, stores = make_workload(train_count=6, dev_count=0)
    task = next(task for task in tasks if task.metadata["family"] == "retrieve_then_calculate")
    agents = build_workflow_agents(stores)
    state = initial_state(task, 16384, 12)
    before = asyncio.run(agents["math"].execute(state))
    assert before.metadata["status"] == "blocked"
    assert before.metadata["answer"] is None
    retrieved = asyncio.run(agents["retriever"].execute(state))
    state.previous_agent_outputs.append(asdict(retrieved))
    after = asyncio.run(agents["math"].execute(state))
    assert after.metadata["status"] == "ok"
    assert "expected_answer" not in repr(state.to_dict())


def test_unverified_answer_rejected_when_request_requires_verification():
    tasks, stores = make_workload(train_count=12, dev_count=0)
    task = next(task for task in tasks if parse_request(task.user_task)["verification"])
    agents = build_workflow_agents(stores)

    async def execute():
        # Stop before the rule's requested verification step.
        class EarlyStop(WorkflowRulePolicy):
            def route(self, state, k):
                decision = super().route(state, k)
                if "verifier" in decision.selected_agents:
                    from conductor.schema import RoutingDecision
                    remaining = [name for name in decision.selected_agents if name != "verifier"]
                    if remaining:
                        return RoutingDecision(remaining, decision.execution_mode).validate(k)
                    return RoutingDecision([], terminate=True)
                return decision
        return await run_trajectory(task, EarlyStop(), agents, k=3, max_rounds=6,
                                    token_budget=16384, agent_call_budget=12)

    trajectory = asyncio.run(execute())
    assert not trajectory.task_success
    assert parse_request(task.user_task)["verification"] is True
    assert trajectory.metadata["workflow_grading"]["reason"] == "invalid_final_json"


def test_private_grader_rejects_forged_unsupported_final_text():
    tasks, _ = make_workload(train_count=6, dev_count=0)
    task = tasks[0]
    state = initial_state(task)
    success, score, details = grade_workflow(task, state,
        '{"result":"' + task.expected_answer + '","evidence":[],"verified":false}')
    assert (success, score) == (False, 0.0)
    assert details["reason"] == "no_grounded_candidate"


def test_public_store_and_request_validation_fail_closed():
    with pytest.raises(ValueError):
        PublicStores.from_dict({"numeric_facts": {}, "research_records": {}, "tool_fixtures": {}, "answers": {}})
    with pytest.raises(ValueError):
        parse_request("{}")


def test_small_corpus_has_measured_successes_failures_and_audited_labels(tmp_path):
    data = tmp_path / "data"
    result = asyncio.run(generate({
        "seed": 17, "train_count": 6, "development_count": 6,
        "dataset_output": str(data), "output": str(tmp_path / "run"),
        "routing": {"k": 3},
        "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
    }))
    assert result["coverage_gate_passed"] and result["capability_isolation_passed"]
    assert 0 < result["trajectory_successes"] < result["trajectory_count"]
    records = [json.loads(line) for line in (data / "sft_train.jsonl").read_text().splitlines()]
    assert {len(record["decision"]["selected_agents"]) for record in records} == {0, 1, 2, 3}
    assert all("expected_answer" not in record["state"] for record in records)
    measured = asyncio.run(evaluate({
        "seed": 17, "data": str(data / "tasks.jsonl"), "splits": ["dev"],
        "output": str(tmp_path / "evaluation"),
        "agents": {"backend": "workflow", "stores_path": str(data / "public_stores.json")},
        "routing": {"k": 3},
        "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
        "policies": [{"id": "public_state_rules"}, {"id": "random_top_k"}],
        "latency_repetitions": 1, "bootstrap_samples": 100,
    }))
    rates = {row["policy"]: row["success_rate"] for row in measured["policies"]}
    assert rates["public_state_rules"] == 1.0
    assert measured["breakdowns"]["public_state_rules"]["family"]
    checkpoint = tmp_path / "checkpoint"
    train({
        "seed": 17,
        "model": {"backend": "cheap", "architecture": "linear", "action_head": "catalog",
                  "feature_dim": 128, "hidden_dim": 16, "max_agents": 3, "device": "cpu"},
        "routing": {"k": 3},
        "training": {"stage": "sft", "data": str(data / "sft_train.jsonl"),
                     "validation_data": str(data / "sft_development.jsonl"), "output": str(checkpoint),
                     "epochs": 1, "batch_size": 16, "learning_rate": .01},
    })
    preferences = asyncio.run(generate_preferences({
        "seed": 17, "checkpoint": str(checkpoint), "tasks": str(data / "tasks.jsonl"),
        "stores": str(data / "public_stores.json"), "output": str(tmp_path / "preferences"),
        "run_output": str(tmp_path / "preference-run"),
    }))
    assert preferences["train_pairs"] > 0 and preferences["development_pairs"] > 0
    assert preferences["status"] == "ready" and not preferences["final_partition_used"]
    with pytest.raises(FileExistsError):
        asyncio.run(generate({"dataset_output": str(data), "output": str(tmp_path / "other")}))
