"""Exercise the research loop on temporary data without asserting a fake gain."""
import asyncio
import hashlib
import json
from pathlib import Path

from conductor.generate import generate
from conductor.training.runner import train
from conductor.evaluate import evaluate
from conductor.benchmark import benchmark


def test_training_evaluation_and_inference_loop(tmp_path: Path) -> None:
    data = tmp_path / "data"
    base = {"seed": 5, "model": {"backend": "tiny", "feature_dim": 64, "hidden_dim": 16,
            "num_experts": 3, "expert_top_k": 1, "max_agents": 2, "device": "cpu"},
            "routing": {"k": 2}, "agents": {"backend": "deterministic"}}
    generation = asyncio.run(generate({**base, "dataset_output": str(data), "output": str(tmp_path / "generate"),
                              "train_count": 8, "eval_count": 6,
                              "policies": ["all_agent", "rule_based", "random_top_k"]}))
    assert generation["preference_records"] > 0
    sft, dpo = tmp_path / "sft", tmp_path / "dpo"
    common = {"epochs": 2, "batch_size": 16, "validation_fraction": 0.25}
    train({**base, "training": {**common, "stage": "sft", "data": str(data / "sft.jsonl"), "output": str(sft)}})
    before = hashlib.sha256((sft / "model.pt").read_bytes()).hexdigest()
    trained = train({**base, "training": {**common, "stage": "dpo", "data": str(data / "preferences.jsonl"),
                     "output": str(dpo)}}, checkpoint=str(sft))
    assert trained["reference_checkpoint_sha256"]
    assert hashlib.sha256((sft / "model.pt").read_bytes()).hexdigest() == before
    evaluation = asyncio.run(evaluate({**base, "data": str(data / "tasks.jsonl"), "output": str(tmp_path / "eval"),
                        "policies": ["rule_based", "conductor_sft", "conductor_preference"],
                        "checkpoints": {"conductor_sft": str(sft), "conductor_preference": str(dpo)}}))
    assert all(row["task_count"] == 6 for row in evaluation["policies"])
    measured = asyncio.run(benchmark({**base, "checkpoint": str(dpo), "output": str(tmp_path / "bench"),
                         "batch_sizes": [2], "context_sizes": [8], "concurrency": [2], "k_values": [2],
                         "routing_intervals": [1], "strategies": ["batched", "dynamic"],
                         "warmup": 0, "repeats": 1, "requests": 4,
                         "data": str(data / "tasks.jsonl"), "tasks_per_category": 1}))
    assert all(row["request_per_second"] > 0 for row in measured["benchmarks"])
    assert json.loads((tmp_path / "bench" / "run.json").read_text())["runtime_seconds"] > 0
