"""Template accounting distinguishes the two code fixtures without input leakage."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict

from conductor.datasets.io import write_jsonl
from conductor.datasets.tasks import TEMPLATE_FAMILIES, make_tasks
from conductor.evaluate import evaluate
from conductor.metrics.aggregate import trajectory_metrics
from conductor.orchestration.runner import initial_state


def test_six_templates_are_balanced_and_existing_prompts_ids_answers_order_are_unchanged() -> None:
    tasks = make_tasks(seed=42, train_count=36, eval_count=48)
    assert Counter(task.metadata["template_family"] for task in tasks if task.split == "eval") == {
        name: 8 for name in TEMPLATE_FAMILIES}
    assert len(TEMPLATE_FAMILIES) == len(set(TEMPLATE_FAMILIES)) == 6
    assert {task.task_type for task in tasks if task.metadata["template_family"] in
            {"reverse_string", "count_vowels"}} == {"code"}
    assert all(task.metadata["family"] == task.task_type for task in tasks)
    assert all(task.metadata["ood"] == (task.metadata["template_family"] in
               {"lookup_multiply", "addition_reverse_digits"}) for task in tasks)
    # Recorded before adding template metadata: captures every original task
    # field and order, without making newly added metadata part of the snapshot.
    records = [{key: getattr(task, key) for key in ("id", "user_task", "task_type", "split", "expected_answer")}
               for task in tasks]
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert fingerprint == "f1232219fa7e27605f51dad6e9d06794c715795cbe6cbef275e424676f5a0521"


def test_template_and_grader_metadata_do_not_enter_public_execution_state() -> None:
    for task in make_tasks(train_count=4, eval_count=6):
        state = initial_state(task).to_dict()
        assert state["user_task"] == task.user_task and state["task_type"] == task.task_type
        assert not {"id", "split", "metadata", "template_family", "expected_answer"}.intersection(state)
        assert "template_family" not in json.dumps(state)
        row = trajectory_metrics({"task": asdict(task), "policy": "fixture", "steps": [],
                                  "task_success": False, "grader_score": 0.0})
        assert row["template_family"] == task.metadata["template_family"]
    legacy = trajectory_metrics({"task": {"id": "legacy", "metadata": {}}, "policy": "fixture", "steps": []})
    assert legacy["template_family"] == "unknown"


def test_actual_rule_evaluation_exports_eight_cases_per_template(tmp_path) -> None:
    tasks = make_tasks(train_count=36, eval_count=48)
    data = tmp_path / "tasks.jsonl"
    write_jsonl(data, [asdict(task) for task in tasks])
    output = tmp_path / "evaluation"
    result = asyncio.run(evaluate({"data": str(data), "output": str(output), "policies": ["rule_based"],
        "agents": {"backend": "deterministic"}, "routing_diagnostics": {"permutation_samples": 0}}))
    assert len(result["templates"]) == 6
    assert {row["template_family"] for row in result["templates"]} == set(TEMPLATE_FAMILIES)
    assert all(row["policy"] == "rule_based" and row["task_count"] == 8 and row["success_rate"] == 1
               for row in result["templates"])
    composed = {"lookup_multiply", "addition_reverse_digits"}
    assert all(row["generalization"] == ("unseen_composition" if row["template_family"] in composed else "seen_family")
               for row in result["templates"])
    with (output / "per_template.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and all(int(row["task_count"]) == 8 for row in rows)
    assert json.loads((output / "metrics.json").read_text())["templates"] == result["templates"]
