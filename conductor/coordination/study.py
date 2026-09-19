"""Run the coordination-v2 stages whose gates have already been frozen.

This driver never advances a failed gate.  A blocked stage is a completed
research outcome and is written to the study ledger instead of being repaired
silently or replaced by a selected seed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

from conductor.coordination.evaluate import evaluate
from conductor.coordination.generate import generate
from conductor.datasets.integrity import file_digest
from conductor.training.probe import probe
from conductor.training.runner import train
from conductor.utils.config import load_config
from conductor.utils.runs import write_json


def _completed_checkpoint(path: Path) -> bool:
    if not (path / "controller.json").exists() or not (path / "metrics.json").exists():
        return False
    metrics = json.loads((path / "metrics.json").read_text())
    return bool(metrics.get("completed"))


def _cheap_training_config(config: dict[str, Any], architecture: str, seed: int, output: Path) -> dict[str, Any]:
    study = config["study"]
    model = {
        "backend": "cheap", "architecture": architecture, "action_head": "catalog",
        "feature_dim": int(study.get("cheap_feature_dim", 512)),
        "hidden_dim": int(study.get("cheap_hidden_dim", 128)),
        "max_agents": 3, "device": "cpu", "dtype": "float32",
    }
    return {
        "seed": seed,
        "model": model,
        "routing": {"k": 3},
        "training": {
            "stage": "sft", "data": str(Path(study["dataset"]) / "sft_train.jsonl"),
            "validation_data": str(Path(study["dataset"]) / "sft_development.jsonl"),
            "output": str(output), "epochs": int(study.get("cheap_epochs", 200)),
            "batch_size": int(study.get("cheap_batch_size", 32)),
            "learning_rate": float(study.get("linear_learning_rate", .01)
                                   if architecture == "linear" else study.get("mlp_learning_rate", .003)),
            "weight_decay": float(study.get("cheap_weight_decay", .001)),
            "max_grad_norm": 1.0, "scheduler": "constant", "validation_fraction": 0,
        },
        "inference": {"sample": False, "feature_cache_size": 2048},
        "tracking": {"enabled": False},
    }


def _policy_summary(result: dict[str, Any], identifier: str) -> dict[str, float]:
    overall = next(row for row in result["policies"] if row["policy"] == identifier)
    families = [row for row in result["per_family"] if row["policy"] == identifier]
    stages = [row for row in result["per_dependency_stage"] if row["policy"] == identifier]
    return {
        "success_rate": float(overall["success_rate"]),
        "family_balanced_success": statistics.fmean(float(row["success_rate"]) for row in families),
        "minimum_family_success": min(float(row["success_rate"]) for row in families),
        "minimum_dependency_stage_success": min(float(row["success_rate"]) for row in stages),
        "mean_agent_calls": float(overall["mean_agent_calls"]),
        "mean_wall_clock_seconds": float(overall["mean_wall_clock_seconds"]),
    }


def select_primary_baseline(per_seed: dict[int, dict[str, dict[str, float]]], floor: float) -> dict[str, Any]:
    """Apply the committed mean-quality/calls/latency/identifier ordering."""
    if not per_seed:
        raise ValueError("baseline selection requires measured seeds")
    candidates = sorted(set.intersection(*(set(values) for values in per_seed.values())))
    rows = []
    for identifier in candidates:
        values = [per_seed[seed][identifier] for seed in sorted(per_seed)]
        row = {
            "policy": identifier,
            "seeds": sorted(per_seed),
            "mean_family_balanced_success": statistics.fmean(item["family_balanced_success"] for item in values),
            "mean_agent_calls": statistics.fmean(item["mean_agent_calls"] for item in values),
            "mean_wall_clock_seconds": statistics.fmean(item["mean_wall_clock_seconds"] for item in values),
            "minimum_seed_success": min(item["success_rate"] for item in values),
        }
        row["eligible"] = row["minimum_seed_success"] >= floor
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return {"status": "blocked", "reason": "no cheap candidate met the frozen development floor",
                "floor": floor, "candidates": rows}
    selected = min(eligible, key=lambda row: (-row["mean_family_balanced_success"], row["mean_agent_calls"],
                                              row["mean_wall_clock_seconds"], row["policy"]))
    return {"status": "locked", "floor": floor, "selected": selected["policy"], "candidates": rows,
            "selection_order": ["highest mean family-balanced workflow success", "lowest mean agent calls",
                                "lowest mean total CPU latency", "lexicographic policy identifier"]}


async def run_cheap_stage(config: dict[str, Any]) -> dict[str, Any]:
    study = config["study"]
    root = Path(study.get("output", "outputs/coordination-v2"))
    dataset = Path(study["dataset"])
    required = [dataset / name for name in ("tasks.jsonl", "public_stores.json", "sft_train.jsonl",
                                             "sft_development.jsonl", "coverage_audit.json")]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("coordination-v2 dataset is incomplete; run the data stage first")
    seeds = [int(value) for value in study.get("seeds", [42, 137, 2027])]
    if seeds != [42, 137, 2027] and not study.get("development_override", False):
        raise ValueError("confirmatory cheap stage requires the three predeclared seeds")
    checkpoints: dict[int, dict[str, str]] = {}
    training_metrics: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        checkpoints[seed], training_metrics[seed] = {}, {}
        for architecture in ("linear", "mlp"):
            output = root / "checkpoints" / f"cheap-{architecture}" / f"seed-{seed}"
            settings = _cheap_training_config(config, architecture, seed, output)
            if _completed_checkpoint(output):
                metrics = json.loads((output / "metrics.json").read_text())
                run = json.loads((output / "run.json").read_text())
                expected = file_digest(dataset / "sft_train.jsonl")
                if run.get("data_identity", {}).get("dataset_sha256") != expected:
                    raise ValueError(f"completed {architecture} seed {seed} used a different fitting corpus")
            else:
                metrics = train(settings)
            checkpoints[seed][architecture] = str(output)
            training_metrics[seed][architecture] = metrics
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    evaluations: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        output = root / "development" / f"seed-{seed}"
        if (output / "metrics.json").exists():
            result = json.loads((output / "metrics.json").read_text())
        else:
            evaluation = {
                "seed": seed, "data": str(dataset / "tasks.jsonl"), "splits": ["dev"], "output": str(output),
                "agents": {"backend": "workflow", "stores_path": str(dataset / "public_stores.json")},
                "model": {"backend": "cheap", "device": "cpu", "dtype": "float32"},
                "routing": {"k": 3},
                "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
                "latency_repetitions": int(study.get("development_latency_repetitions", 1)),
                "bootstrap_samples": 1999,
                "policies": [
                    {"id": "public_state_rules"},
                    {"id": "random_top_k"},
                    {"id": "iterative_all_agent_sequential"},
                    {"id": "all_agent_parallel"},
                    {"id": "static_supervisor"},
                    {"id": "sparse_linear_router", "kind": "checkpoint",
                     "checkpoint": checkpoints[seed]["linear"]},
                    {"id": "small_mlp_router", "kind": "checkpoint",
                     "checkpoint": checkpoints[seed]["mlp"]},
                ],
            }
            result = await evaluate(evaluation)
        evaluations[seed] = result
        per_seed[seed] = {identifier: _policy_summary(result, identifier)
                          for identifier in ("public_state_rules", "sparse_linear_router", "small_mlp_router")}
    floor = float(study.get("development_floor", .95))
    gates = {seed: {identifier: {
        "passed": values["success_rate"] >= floor and values["minimum_dependency_stage_success"] >= floor,
        "success_rate": values["success_rate"],
        "minimum_dependency_stage_success": values["minimum_dependency_stage_success"],
        "floor": floor,
    } for identifier, values in policies.items() if identifier != "public_state_rules"}
             for seed, policies in per_seed.items()}
    lock = select_primary_baseline(per_seed, floor)
    lock_path = root / "baseline_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("existing primary baseline lock differs from recomputed measurements")
    else:
        write_json(lock_path, lock)
    summary = {"stage": "cheap_supervised_controls", "seeds": seeds, "per_seed": per_seed,
               "development_gates": gates, "primary_baseline_lock": lock,
               "training_metrics": training_metrics,
               "all_learned_controls_passed": all(cell["passed"] for values in gates.values() for cell in values.values()),
               "next_stage": "cached_pretrained_representation_probe",
               "dataset_hashes": {path.name: file_digest(path) for path in required}}
    write_json(root / "cheap_stage_summary.json", summary)
    return summary


async def run_probe_stage(config: dict[str, Any]) -> dict[str, Any]:
    study = config["study"]
    root = Path(study.get("output", "outputs/coordination-v2"))
    dataset = Path(study["dataset"])
    cheap_summary_path = root / "cheap_stage_summary.json"
    if not cheap_summary_path.exists() or not json.loads(cheap_summary_path.read_text()).get("all_learned_controls_passed"):
        raise ValueError("cached pretrained probe is blocked until the cheap learned controls pass development")
    seeds = [int(value) for value in study.get("seeds", [42, 137, 2027])]
    probe_output = root / "probe"
    model = {
        "backend": "hf", "name": "ibm-granite/granite-3.1-1b-a400m-base",
        "revision": "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29", "pretrained": True,
        "max_agents": 3, "max_length": int(study.get("model_max_length", 384)),
        "device": "cpu", "dtype": "float32", "attention_implementation": "sdpa",
        "local_files_only": True, "trust_remote_code": False, "state_serialization": "priority_v1",
        "head_type": "catalog", "head_hidden_dim": int(study.get("factorized_head_hidden_dim", 64)),
        "lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0,
                 "target_modules": ["q_proj", "v_proj", "router.layer"]},
    }
    settings = {
        "seed": seeds[0], "model": model, "routing": {"k": 3},
        "inference": {"instrument_experts": False, "sample": False, "token_cache_size": 0},
        "probe": {"data": str(dataset / "sft_train.jsonl"),
                  "validation_data": str(dataset / "sft_development.jsonl"),
                  "output": str(probe_output), "batch_size": int(study.get("probe_batch_size", 2)),
                  "epochs": int(study.get("probe_epochs", 200)),
                  "learning_rate": float(study.get("probe_learning_rate", .01)),
                  "head_types": ["catalog", "factorized"], "seeds": seeds},
        "tracking": {"enabled": False},
    }
    if (probe_output / "metrics.json").exists():
        measured_probe = json.loads((probe_output / "metrics.json").read_text())
    else:
        measured_probe = probe(settings)
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        output = root / "development-probe" / f"seed-{seed}"
        if (output / "metrics.json").exists():
            result = json.loads((output / "metrics.json").read_text())
        else:
            checkpoints = measured_probe["checkpoints"][str(seed)]
            result = await evaluate({
                "seed": seed, "data": str(dataset / "tasks.jsonl"), "splits": ["dev"], "output": str(output),
                "agents": {"backend": "workflow", "stores_path": str(dataset / "public_stores.json")},
                "model": model, "routing": {"k": 3},
                "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
                "latency_repetitions": int(study.get("development_latency_repetitions", 1)),
                "bootstrap_samples": 1999,
                "policies": [
                    {"id": "public_state_rules"},
                    {"id": "frozen_catalog_head", "kind": "frozen_backbone_trained_head",
                     "checkpoint": checkpoints["catalog"]},
                    {"id": "frozen_factorized_head", "kind": "frozen_backbone_trained_head",
                     "checkpoint": checkpoints["factorized"]},
                ],
            })
        per_seed[seed] = {identifier: _policy_summary(result, identifier)
                          for identifier in ("frozen_catalog_head", "frozen_factorized_head")}
    floor = float(study.get("development_floor", .95))
    gates = {seed: {identifier: {"passed": values["success_rate"] >= floor
                                           and values["minimum_dependency_stage_success"] >= floor,
                                 "success_rate": values["success_rate"],
                                 "minimum_dependency_stage_success": values["minimum_dependency_stage_success"],
                                 "floor": floor}
                    for identifier, values in policies.items()}
             for seed, policies in per_seed.items()}
    factorized_passed = all(values["frozen_factorized_head"]["passed"] for values in gates.values())
    summary = {"stage": "cached_pretrained_frozen_heads", "seeds": seeds,
               "probe": measured_probe, "per_seed": per_seed, "development_gates": gates,
               "factorized_head_all_seeds_passed": factorized_passed,
               "next_stage": "factorized_lora_sft" if factorized_passed else "blocked_before_lora",
               "model_context_locked": {"state_serialization": "priority_v1",
                                        "max_length": model["max_length"],
                                        "selected_representations": measured_probe["selected_representations"]}}
    write_json(root / "probe_stage_summary.json", summary)
    return summary


async def run(config: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "data":
        return await generate(config["data_generation"])
    if stage == "cheap":
        return await run_cheap_stage(config)
    if stage == "probe":
        return await run_probe_stage(config)
    raise ValueError("stage must be data, cheap, or probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/coordination_v2_study.yaml")
    parser.add_argument("--stage", required=True, choices=("data", "cheap", "probe"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(load_config(args.config), args.stage)), indent=2))


if __name__ == "__main__":
    main()
