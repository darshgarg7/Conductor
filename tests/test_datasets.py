import asyncio
from dataclasses import asdict
from conductor.agents import build_agents
from conductor.datasets.io import read_jsonl, write_jsonl
from conductor.datasets.tasks import make_tasks
from conductor.orchestration.runner import initial_state
from conductor.generate import preference_records


def test_task_splits_are_reproducible_disjoint_and_ood() -> None:
    first, second = make_tasks(), make_tasks()
    assert [asdict(task) for task in first] == [asdict(task) for task in second]
    train = [task for task in first if task.split == "train"]
    heldout = [task for task in first if task.split == "eval"]
    assert not {task.id for task in train} & {task.id for task in heldout}
    assert not {task.user_task for task in train} & {task.user_task for task in heldout}
    assert {task.task_type for task in heldout} - {task.task_type for task in train}
    for task in first:
        serialized = initial_state(task).to_dict()
        assert "expected_answer" not in serialized and "metadata" not in serialized


def test_jsonl_and_trajectory_roundtrip(tmp_path) -> None:
    from conductor.orchestration.runner import run_trajectory
    from conductor.routing.policies import RuleBasedPolicy
    from conductor.schema import Trajectory
    trajectory = asyncio.run(run_trajectory(make_tasks()[0], RuleBasedPolicy(), build_agents()))
    path = tmp_path / "trajectories.jsonl"
    write_jsonl(path, [trajectory.to_dict()])
    record = read_jsonl(path)[0]
    assert Trajectory.from_dict(record).to_dict() == trajectory.to_dict()


def test_preferences_compare_exact_same_state_and_same_continuation() -> None:
    task = make_tasks(train_count=1, eval_count=1)[0]
    budgets = {"k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}
    records = asyncio.run(preference_records(task, {"preference": {"max_candidates": 10}}, build_agents(), budgets,
                                            {"token_weight": 0.001, "latency_weight": 0.0,
                                             "agent_call_weight": 0.1, "communication_weight": 0.0}))
    assert records
    assert all(record["state"] == initial_state(task).to_dict() for record in records)
    assert all(record["continuation_policy"] == "rule_based" for record in records)
    assert all(record["chosen_reward"] > record["rejected_reward"] for record in records)
    assert all(record["chosen_success"] for record in records)
