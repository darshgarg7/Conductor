import asyncio
import json
from dataclasses import asdict

import pytest

from conductor.datasets.io import write_jsonl
from conductor.datasets.tasks import make_tasks
from conductor.evaluate import checkpoint_policy, evaluate
from conductor.evaluation.tasks import heldout_tasks
from conductor.schema import Task


def test_heldout_checks_ids_and_normalized_prompts():
    train = Task("train", "Compute   1 + 2.", "math")
    leaked = Task("different-id", "compute 1 + 2.", "math", "eval")
    with pytest.raises(ValueError, match="leakage"):
        heldout_tasks([train, leaked])
    with pytest.raises(ValueError, match="leakage"):
        heldout_tasks([train, Task("train", "Different text", "math", "test")])
    with pytest.raises(ValueError, match="No heldout"):
        heldout_tasks([train])


def test_unseen_compositions_are_explicit_heldout():
    tasks, audit = heldout_tasks(make_tasks())
    assert audit["train_overlap_ids"] == audit["train_overlap_texts"] == 0
    assert all(task.split == "eval" for task in tasks)
    assert any(task.metadata["ood"] for task in tasks)


def test_evaluation_uses_identical_tasks_and_marks_supervisor_unavailable(tmp_path):
    data = tmp_path / "tasks.jsonl"
    write_jsonl(data, [asdict(task) for task in make_tasks(train_count=4, eval_count=6)])
    output = tmp_path / "evaluation"
    config = {"data": str(data), "output": str(output), "seed": 42,
              "policies": ["all_agent", "rule_based", "static_supervisor"],
              "agents": {"backend": "deterministic"}, "k": 2, "max_rounds": 3,
              "agent_call_budget": 12, "token_budget": 8192}
    result = asyncio.run(evaluate(config))
    assert next(item for item in result["policy_status"] if item["policy"] == "static_supervisor")["status"] == "unavailable"
    trajectories = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
    ids = {name: {item["task"]["id"] for item in trajectories if item["policy"] == name} for name in ("all_agent", "rule_based")}
    assert ids["all_agent"] == ids["rule_based"]
    assert all(item["task"]["split"] == "eval" for item in trajectories)
    assert (output / "per_category.csv").exists()


def test_checkpoint_override_changes_only_its_saved_stage(tmp_path):
    from conductor.controller.tiny import TinyController
    data = tmp_path / "tasks.jsonl"
    write_jsonl(data, [asdict(task) for task in make_tasks(train_count=2, eval_count=2)])
    config = {"data": str(data), "output": str(tmp_path / "eval"), "policies": ["conductor_sft", "conductor_preference"],
              "agents": {"backend": "deterministic"}, "model": {"backend": "tiny", "device": "cpu"}}
    paths = {name: tmp_path / name for name in ("sft-configured", "sft-override", "preference")}
    for name, path in paths.items():
        TinyController(config).save(path, "preference" if name == "preference" else "sft")
    config["checkpoints"] = {"conductor_sft": str(paths["sft-configured"]), "conductor_preference": str(paths["preference"])}
    result = asyncio.run(evaluate(config, str(paths["sft-override"])))
    statuses = {item["policy"]: item for item in result["policy_status"]}
    assert statuses["conductor_sft"]["checkpoint"] == str(paths["sft-override"])
    assert statuses["conductor_preference"]["checkpoint"] == str(paths["preference"])
    probes = json.loads((tmp_path / "eval" / "initial_state_probe.json").read_text())
    left = [state["state_sha256"] for state in probes["conductor_sft"]["states"]]
    right = [state["state_sha256"] for state in probes["conductor_preference"]["states"]]
    assert left == right


def test_unknown_checkpoint_stage_is_rejected(tmp_path):
    (tmp_path / "controller.json").write_text(json.dumps({"stage": "random_initialization"}))
    with pytest.raises(ValueError, match="stage"):
        checkpoint_policy(str(tmp_path))
