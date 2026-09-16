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


def test_journal_recovers_partial_append_and_deduplicates(tmp_path) -> None:
    from conductor.datasets.integrity import Journal, record_checksum, iter_records
    path = tmp_path / "journal.jsonl"
    with Journal(path) as journal:
        assert journal.append({"payload": 1}, "one")
        assert not journal.append({"payload": 1}, "one")
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    with Journal(path) as journal:
        assert journal.recovered_bytes > 0
        assert journal.get("one")["payload"] == 1
        assert journal.append({"payload": 2}, "two")
    records = list(iter_records(path))
    assert len(records) == 2
    assert all(record_checksum(record) == record["metadata"]["record_checksum_sha256"] for record in records)


def test_external_source_rejects_split_leakage(tmp_path) -> None:
    import pytest
    from conductor.datasets.source import external_tasks
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [{"id": "a", "user_task": "same task", "task_type": "real", "split": "train", "expected_answer": "x"},
                       {"id": "b", "user_task": "same task", "task_type": "real", "split": "eval", "expected_answer": "x"}])
    with pytest.raises(ValueError, match="duplicate"):
        list(external_tasks(path))


def test_sharded_generation_resumes_without_duplicate_execution(tmp_path) -> None:
    from conductor.generate import generate
    from conductor.datasets.integrity import shard_for, record_checksum
    config = {"seed": 7, "train_count": 2, "eval_count": 2, "dataset_output": str(tmp_path / "data"),
              "output": str(tmp_path / "run"), "policies": ["all_agent", "rule_based"],
              "jobshards": 2, "shard_index": 0, "preference": {"enabled": False}}
    first = asyncio.run(generate(config))
    second = asyncio.run(generate(config))
    path = tmp_path / "data" / "shards" / "shard-00000-of-00002" / "trajectories.jsonl"
    records = read_jsonl(path)
    assert second["resumed_trajectories"] == first["trajectories"] == len(records)
    assert all(shard_for(record["task"]["id"], 2) == 0 for record in records)
    assert all(record_checksum(record) == record["metadata"]["record_checksum_sha256"] for record in records)


def test_late_state_preference_can_choose_stopping() -> None:
    from conductor.schema import ExecutionState
    from conductor.orchestration.runner import run_trajectory
    from conductor.routing.policies import RuleBasedPolicy
    task = make_tasks(train_count=1, eval_count=1)[0]
    agents = build_agents()
    trajectory = asyncio.run(run_trajectory(task, RuleBasedPolicy(), agents))
    state = ExecutionState(**trajectory.steps[-1].state)
    budgets = {"k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}
    records = asyncio.run(preference_records(task, {"preference": {"max_candidates": 9}}, agents, budgets,
                                            {"latency_weight": 0}, state))
    assert records and all(record["state"] == state.to_dict() for record in records)
    assert all(record["chosen"]["terminate"] for record in records)


def test_journal_constructor_corruption_releases_lock_even_with_retained_exception(tmp_path) -> None:
    import pytest
    from conductor.datasets.integrity import Journal
    path = tmp_path / "corrupt.jsonl"
    with Journal(path) as journal:
        journal.append({"payload": 1}, "one")
    original = path.read_bytes()
    path.write_bytes(b'{"metadata": {"record_id": "one", "record_checksum_sha256": "wrong"}}\n')
    with pytest.raises(ValueError) as retained:
        Journal(path)
    assert retained.value  # Retain the exception/traceback during the next open.
    path.write_bytes(original)
    with Journal(path) as recovered:
        assert recovered.get("one")["payload"] == 1


def test_generation_job_lock_covers_inventory_before_journals_open(tmp_path, monkeypatch) -> None:
    import pytest
    import conductor.generate as module
    from conductor.datasets.integrity import SingleWriterLock
    configuration = {"dataset_output": str(tmp_path / "data"), "output": str(tmp_path / "run")}
    monkeypatch.setattr(module, "make_tasks", lambda *args: pytest.fail("inventory work must not start without job ownership"))
    with SingleWriterLock(tmp_path / "data" / ".generation.lock"):
        with pytest.raises(RuntimeError, match="another generation job"):
            asyncio.run(module.generate(configuration))
    assert not (tmp_path / "data" / "manifest.json").exists()


def test_durable_preference_trials_replay_without_execution_or_remeasuring_latency(tmp_path, monkeypatch) -> None:
    import pytest
    import conductor.generate as module
    from conductor.datasets.integrity import Journal
    task = make_tasks(train_count=1, eval_count=1)[0]
    configuration = {"seed": 42, "preference": {"max_candidates": 9}}
    budgets = {"k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}
    agents = build_agents()
    with Journal(tmp_path / "trials.jsonl") as journal:
        first = asyncio.run(module.preference_records(task, configuration, agents, budgets, {"latency_weight": .02}, trial_journal=journal))
        assert first and len(journal.ids) == 9
        async def forbidden(*args, **kwargs):
            pytest.fail("a durably completed counterfactual must not run twice")
        monkeypatch.setattr(module, "run_trajectory", forbidden)
        second = asyncio.run(module.preference_records(task, configuration, agents, budgets, {"latency_weight": .02}, trial_journal=journal))
    assert first == second


def test_external_source_requires_typed_labels_and_detects_unicode_split_leakage(tmp_path) -> None:
    import pytest
    from conductor.datasets.source import external_tasks
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [{"id": "one", "user_task": "A task", "task_type": "math", "split": "train", "expected_answer": 3}])
    with pytest.raises(ValueError, match="expected_answer"):
        list(external_tasks(path))
    write_jsonl(path, [{"id": "one", "user_task": "Café question", "task_type": "real", "split": "train", "expected_answer": "x"},
                       {"id": "two", "user_task": "Cafe\u0301 question", "task_type": "real", "split": "eval", "expected_answer": "x"}])
    with pytest.raises(ValueError, match="duplicate"):
        list(external_tasks(path))
