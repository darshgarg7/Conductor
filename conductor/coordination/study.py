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
from conductor.coordination.preferences import generate_preferences
from conductor.datasets.integrity import file_digest
from conductor.training.probe import probe
from conductor.training.runner import checkpoint_digest, train
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
    head_identifiers = ("frozen_catalog_head", "frozen_factorized_head")
    eligible = [identifier for identifier in head_identifiers
                if all(values[identifier]["passed"] for values in gates.values())]
    head_rows = []
    for identifier in head_identifiers:
        values = [per_seed[seed][identifier] for seed in seeds]
        head_rows.append({"policy": identifier,
                          "mean_family_balanced_success": statistics.fmean(
                              item["family_balanced_success"] for item in values),
                          "mean_agent_calls": statistics.fmean(item["mean_agent_calls"] for item in values),
                          "mean_wall_clock_seconds": statistics.fmean(
                              item["mean_wall_clock_seconds"] for item in values),
                          "all_seed_gates_passed": identifier in eligible})
    selected = (min((row for row in head_rows if row["all_seed_gates_passed"]),
                    key=lambda row: (-row["mean_family_balanced_success"], row["mean_agent_calls"],
                                     row["mean_wall_clock_seconds"], row["policy"]))["policy"]
                if eligible else None)
    summary = {"stage": "cached_pretrained_frozen_heads", "seeds": seeds,
               "probe": measured_probe, "per_seed": per_seed, "development_gates": gates,
               "head_selection": {"eligible": eligible, "selected": selected, "candidates": head_rows,
                                  "rule": "only all-seed passing heads; then success, calls, CPU latency, identifier"},
               "factorized_head_all_seeds_passed": "frozen_factorized_head" in eligible,
               "next_stage": "catalog_lora_sft" if selected == "frozen_catalog_head" else (
                   "factorized_lora_sft" if selected == "frozen_factorized_head" else "blocked_before_lora"),
               "model_context_locked": {"state_serialization": "priority_v1",
                                        "max_length": model["max_length"],
                                        "selected_representations": measured_probe["selected_representations"]}}
    write_json(root / "probe_stage_summary.json", summary)
    return summary


def _adapter_update_report(before: str | Path, after: str | Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    source = load_file(str(Path(before) / "adapter" / "adapter_model.safetensors"))
    trained = load_file(str(Path(after) / "adapter" / "adapter_model.safetensors"))
    if source.keys() != trained.keys():
        raise ValueError("LoRA adapter tensor inventory changed across SFT")
    changed = {name: float((trained[name].float() - source[name].float()).abs().max())
               for name in source if not torch.equal(source[name], trained[name])}
    nonzero_b = [name for name, value in trained.items() if "lora_B" in name and bool(value.abs().max() > 0)]
    groups = {target: any(target in name for name in changed) for target in ("q_proj", "v_proj", "router")}
    return {"source_adapter": str(before), "trained_adapter": str(after),
            "adapter_tensors": len(source), "changed_tensors": len(changed),
            "maximum_absolute_update": max(changed.values(), default=0.0),
            "nonzero_lora_b_tensors": len(nonzero_b), "target_groups_changed": groups,
            "passed": bool(changed and nonzero_b and all(groups.values()))}


async def run_sft_stage(config: dict[str, Any]) -> dict[str, Any]:
    study = config["study"]
    root = Path(study.get("output", "outputs/coordination-v2"))
    dataset = Path(study["dataset"])
    probe_summary = json.loads((root / "probe_stage_summary.json").read_text())
    selected = probe_summary.get("head_selection", {}).get("selected")
    head_type = {"frozen_catalog_head": "catalog", "frozen_factorized_head": "factorized"}.get(selected)
    if head_type is None:
        raise ValueError("LoRA SFT is blocked because no frozen head passed every development gate")
    seeds = [int(value) for value in study.get("seeds", [42, 137, 2027])]
    checkpoints = probe_summary["probe"]["checkpoints"]
    training_metrics: dict[int, Any] = {}
    adapter_updates: dict[int, Any] = {}
    sft_checkpoints: dict[int, str] = {}
    for seed in seeds:
        source = checkpoints[str(seed)][head_type]
        output = root / "checkpoints" / f"granite-{head_type}-sft" / f"seed-{seed}"
        settings = {
            "seed": seed,
            "model": {"backend": "hf", "device": "cpu", "dtype": "float32",
                      "max_length": int(study.get("model_max_length", 384)),
                      "attention_implementation": "sdpa"},
            "routing": {"k": 3},
            "inference": {"instrument_experts": True, "sample": False,
                          "preserve_checkpoint_context": True, "token_cache_size": 0},
            "training": {"stage": "sft", "data": str(dataset / "sft_train.jsonl"),
                         "validation_data": str(dataset / "sft_development.jsonl"),
                         "checkpoint": source, "output": str(output),
                         "epochs": 2, "batch_size": int(study.get("sft_batch_size", 2)),
                         "gradient_accumulation_steps": int(study.get("sft_gradient_accumulation_steps", 2)),
                         "learning_rate": float(study.get("sft_learning_rate", .0001)),
                         "weight_decay": .01, "max_grad_norm": 1.0,
                         "gradient_checkpointing": False, "scheduler": "constant", "save_every_epochs": 1},
            "tracking": {"enabled": False},
        }
        metrics = json.loads((output / "metrics.json").read_text()) if _completed_checkpoint(output) else train(
            settings, source)
        report = _adapter_update_report(source, output)
        if not report["passed"]:
            raise RuntimeError(f"saved LoRA adapter did not update all declared target groups for seed {seed}")
        training_metrics[seed] = metrics
        adapter_updates[seed] = report
        sft_checkpoints[seed] = str(output)
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        output = root / "development-sft" / f"seed-{seed}"
        if (output / "metrics.json").exists():
            result = json.loads((output / "metrics.json").read_text())
        else:
            result = await evaluate({
                "seed": seed, "data": str(dataset / "tasks.jsonl"), "splits": ["dev"], "output": str(output),
                "agents": {"backend": "workflow", "stores_path": str(dataset / "public_stores.json")},
                "model": {"backend": "hf", "device": "cpu", "dtype": "float32",
                          "max_length": int(study.get("model_max_length", 384))},
                "routing": {"k": 3},
                "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
                "latency_repetitions": int(study.get("development_latency_repetitions", 1)),
                "bootstrap_samples": 1999,
                "policies": [
                    {"id": "public_state_rules"},
                    {"id": "frozen_selected_head", "kind": "frozen_backbone_trained_head",
                     "checkpoint": checkpoints[str(seed)][head_type]},
                    {"id": "conductor_sft", "kind": "conductor_sft",
                     "checkpoint": sft_checkpoints[seed]},
                ],
            })
        per_seed[seed] = {identifier: _policy_summary(result, identifier)
                          for identifier in ("frozen_selected_head", "conductor_sft")}
    floor = float(study.get("development_floor", .95))
    gates = {seed: {"passed": values["conductor_sft"]["success_rate"] >= floor
                              and values["conductor_sft"]["minimum_dependency_stage_success"] >= floor,
                    "success_rate": values["conductor_sft"]["success_rate"],
                    "minimum_dependency_stage_success": values["conductor_sft"]["minimum_dependency_stage_success"],
                    "floor": floor} for seed, values in per_seed.items()}
    passed = all(value["passed"] for value in gates.values())
    summary = {"stage": "lora_sft", "head_type": head_type, "seeds": seeds,
               "checkpoints": sft_checkpoints, "training_metrics": training_metrics,
               "adapter_updates": adapter_updates, "per_seed": per_seed,
               "development_gates": gates, "all_seeds_passed": passed,
               "next_stage": "on_policy_preferences" if passed else "blocked_before_dpo"}
    write_json(root / "sft_stage_summary.json", summary)
    return summary


def _controller_runtime(study: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {"backend": "hf", "device": "cpu", "dtype": "float32",
                  "max_length": int(study.get("model_max_length", 384)),
                  "attention_implementation": "sdpa"},
        "routing": {"k": 3},
        "inference": {"instrument_experts": True, "sample": False,
                      "preserve_checkpoint_context": True, "token_cache_size": 0},
        "tracking": {"enabled": False},
    }


async def run_preference_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Collect exact-state DPO pairs from each eligible SFT actor.

    Development pairs are retained only for diagnostics and validation.  A seed
    with no measured train or development mistakes blocks its dependent DPO run;
    the driver never invents negatives or pools actors across seeds.
    """
    study = config["study"]
    root = Path(study.get("output", "outputs/coordination-v2"))
    dataset = Path(study["dataset"])
    sft_summary_path = root / "sft_stage_summary.json"
    if not sft_summary_path.exists():
        raise FileNotFoundError("on-policy preferences require the completed SFT stage")
    sft_summary = json.loads(sft_summary_path.read_text())
    if not sft_summary.get("all_seeds_passed"):
        raise ValueError("on-policy preferences are blocked because an SFT seed failed development")
    seeds = [int(value) for value in study.get("seeds", [42, 137, 2027])]
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        checkpoint = sft_summary["checkpoints"][str(seed)]
        output = dataset / "preferences" / f"seed-{seed}"
        manifest = output / "manifest.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result.get("tasks_sha256") != file_digest(dataset / "tasks.jsonl"):
                raise ValueError(f"seed {seed} preference task inventory changed")
            if result.get("stores_sha256") != file_digest(dataset / "public_stores.json"):
                raise ValueError(f"seed {seed} preference public stores changed")
            if result.get("actor_checkpoint_sha256") != checkpoint_digest(checkpoint):
                raise ValueError(f"seed {seed} preference actor checkpoint changed")
        else:
            settings = {
                "seed": seed, "checkpoint": checkpoint,
                "tasks": str(dataset / "tasks.jsonl"),
                "stores": str(dataset / "public_stores.json"),
                "output": str(output),
                "run_output": str(root / "preference-generation" / f"seed-{seed}"),
                "minimum_margin": float(study.get("preference_minimum_margin", 1e-6)),
                "reward": {
                    "token_weight": float(study.get("preference_token_weight", .05)),
                    "latency_weight": 0.0,
                    "agent_call_weight": float(study.get("preference_agent_call_weight", .04)),
                    "communication_weight": float(study.get("preference_communication_weight", .01)),
                },
                **_controller_runtime(study),
            }
            result = await generate_preferences(settings, checkpoint)
        per_seed[seed] = result
    ready = all(item.get("status") == "ready" for item in per_seed.values())
    summary = {
        "stage": "on_policy_preferences", "seeds": seeds, "per_seed": per_seed,
        "all_seeds_ready": ready,
        "fit_partition": "train-task actor states only",
        "development_partition": "pair-ranking validation and diagnostics only",
        "final_partition_used": False,
        "next_stage": "dpo" if ready else "blocked_before_dpo",
    }
    write_json(root / "preference_stage_summary.json", summary)
    return summary


async def run_dpo_stage(config: dict[str, Any]) -> dict[str, Any]:
    """Fit one DPO epoch per seed against that seed's exact frozen SFT reference."""
    study = config["study"]
    root = Path(study.get("output", "outputs/coordination-v2"))
    dataset = Path(study["dataset"])
    preference_path = root / "preference_stage_summary.json"
    if not preference_path.exists():
        raise FileNotFoundError("DPO requires the on-policy preference stage")
    preference_summary = json.loads(preference_path.read_text())
    if not preference_summary.get("all_seeds_ready"):
        raise ValueError("DPO is blocked because at least one seed lacks measured train/development pairs")
    sft_summary = json.loads((root / "sft_stage_summary.json").read_text())
    seeds = [int(value) for value in study.get("seeds", [42, 137, 2027])]
    checkpoints: dict[int, str] = {}
    training_metrics: dict[int, Any] = {}
    adapter_updates: dict[int, Any] = {}
    for seed in seeds:
        reference = sft_summary["checkpoints"][str(seed)]
        preferences = dataset / "preferences" / f"seed-{seed}"
        output = root / "checkpoints" / f"granite-{sft_summary['head_type']}-dpo" / f"seed-{seed}"
        settings = {
            "seed": seed,
            **_controller_runtime(study),
            "training": {
                "stage": "dpo", "data": str(preferences / "train.jsonl"),
                "validation_data": str(preferences / "development.jsonl"),
                "checkpoint": reference, "output": str(output),
                "epochs": 1, "batch_size": int(study.get("dpo_batch_size", 2)),
                "gradient_accumulation_steps": int(study.get("dpo_gradient_accumulation_steps", 2)),
                "learning_rate": float(study.get("dpo_learning_rate", .00003)),
                "beta": float(study.get("dpo_beta", .1)), "weight_decay": .01,
                "max_grad_norm": 1.0, "gradient_checkpointing": False,
                "scheduler": "constant", "save_every_epochs": 1,
            },
        }
        if _completed_checkpoint(output):
            metrics = json.loads((output / "metrics.json").read_text())
            if metrics.get("dataset_sha256") != file_digest(preferences / "train.jsonl"):
                raise ValueError(f"seed {seed} completed DPO used a different preference corpus")
            if metrics.get("reference_checkpoint_sha256") != checkpoint_digest(reference):
                raise ValueError(f"seed {seed} completed DPO used a different SFT reference")
        else:
            metrics = train(settings, reference)
        report = _adapter_update_report(reference, output)
        training_metrics[seed] = metrics
        adapter_updates[seed] = report
        checkpoints[seed] = str(output)
    per_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        output = root / "development-dpo" / f"seed-{seed}"
        if (output / "metrics.json").exists():
            result = json.loads((output / "metrics.json").read_text())
        else:
            result = await evaluate({
                "seed": seed, "data": str(dataset / "tasks.jsonl"), "splits": ["dev"],
                "output": str(output),
                "agents": {"backend": "workflow", "stores_path": str(dataset / "public_stores.json")},
                **_controller_runtime(study),
                "orchestration": {"max_rounds": 6, "token_budget": 16384, "agent_call_budget": 12},
                "latency_repetitions": int(study.get("development_latency_repetitions", 1)),
                "bootstrap_samples": 1999,
                "policies": [
                    {"id": "public_state_rules"},
                    {"id": "conductor_sft", "kind": "conductor_sft",
                     "checkpoint": sft_summary["checkpoints"][str(seed)]},
                    {"id": "conductor_preference", "kind": "conductor_preference",
                     "checkpoint": checkpoints[seed]},
                ],
            })
        per_seed[seed] = {identifier: _policy_summary(result, identifier)
                          for identifier in ("conductor_sft", "conductor_preference")}
    # This is a development-only artifact-release decision, frozen before the
    # final inventory exists.  Final evaluation still reports both stages.
    stage_locks: dict[int, dict[str, Any]] = {}
    for seed, values in per_seed.items():
        sft, dpo = values["conductor_sft"], values["conductor_preference"]
        quality_ok = (dpo["success_rate"] >= sft["success_rate"] - .05
                      and dpo["minimum_family_success"] >= .90)
        call_reduction = ((sft["mean_agent_calls"] - dpo["mean_agent_calls"]) / sft["mean_agent_calls"]
                          if sft["mean_agent_calls"] else None)
        dpo_selected = quality_ok and call_reduction is not None and call_reduction >= .10
        stage_locks[seed] = {
            "selected": "conductor_preference" if dpo_selected else "conductor_sft",
            "quality_gate_passed": quality_ok, "actual_call_reduction": call_reduction,
            "dpo_efficiency_gate_passed": bool(dpo_selected),
            "rule": "DPO only if within five quality points, every family >=90%, and calls fall >=10%",
        }
    summary = {
        "stage": "dpo", "seeds": seeds, "checkpoints": checkpoints,
        "training_metrics": training_metrics, "adapter_updates": adapter_updates,
        "per_seed": per_seed, "development_stage_locks": stage_locks,
        "all_seeds_completed": all(value.get("completed") for value in training_metrics.values()),
        "next_stage": "lock_fresh_final_inventory",
    }
    write_json(root / "dpo_stage_summary.json", summary)
    write_json(root / "candidate_stage_lock.json", {
        "scope": "development-only artifact-stage selection before final inventory generation",
        "seeds": stage_locks,
    })
    return summary


async def run(config: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "data":
        return await generate(config["data_generation"])
    if stage == "cheap":
        return await run_cheap_stage(config)
    if stage == "probe":
        return await run_probe_stage(config)
    if stage == "sft":
        return await run_sft_stage(config)
    if stage == "preferences":
        return await run_preference_stage(config)
    if stage == "dpo":
        return await run_dpo_stage(config)
    raise ValueError("stage must be data, cheap, probe, sft, preferences, or dpo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/coordination_v2_study.yaml")
    parser.add_argument("--stage", required=True,
                        choices=("data", "cheap", "probe", "sft", "preferences", "dpo"))
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(load_config(args.config), args.stage)), indent=2))


if __name__ == "__main__":
    main()
