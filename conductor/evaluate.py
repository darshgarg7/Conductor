"""Evaluate identical heldout tasks with fixed agents and aggregate budgets."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from conductor.evaluation.io import write_csv, write_jsonl
from conductor.evaluation.tasks import heldout_tasks
from conductor.evaluation.provenance import canonical_hash, checkpoint_sha256, file_sha256, specialist_identity
from conductor.metrics.aggregate import aggregate_metrics, trajectory_metrics
from conductor.utils.config import load_config
from conductor.utils.runs import Run, log_event, seed_everything, write_json

POLICIES = ("all_agent", "rule_based", "random_top_k", "static_supervisor", "base_moe",
            "conductor_sft", "conductor_preference")


def checkpoint_policy(checkpoint: str) -> str:
    from conductor.controller.artifacts import resolve_checkpoint
    stage = json.loads((resolve_checkpoint(checkpoint) / "controller.json").read_text()).get("stage")
    mapping = {"sft": "conductor_sft", "preference": "conductor_preference", "dpo": "conductor_preference"}
    if stage not in mapping:
        raise ValueError(f"Evaluation checkpoint stage must be sft or preference; got {stage!r}")
    return mapping[stage]


async def evaluate(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    from conductor.agents import build_agents
    from conductor.datasets.io import read_jsonl
    from conductor.orchestration.runner import initial_state, run_trajectory
    from conductor.routing.policies import build_policy
    from conductor.routing.serialization import serialize_state

    seed_everything(config.get("seed", 42))
    override_policy = checkpoint_policy(checkpoint) if checkpoint else None
    run = Run(config.get("output", "outputs/evaluation/dev"), config, checkpoint)
    records = list(read_jsonl(config.get("data", "data/dev/tasks.jsonl")))
    tasks, split_audit = heldout_tasks(records, tuple(config.get("splits", ["eval", "test"])))
    write_json(run.output / "split_audit.json", split_audit)
    agents = build_agents(config)
    budgets = {key: config.get(key, default) for key, default in {
        "k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}.items()}
    strict = bool(config.get("strict_mode", config.get("research", False)))
    selected = list(config.get("policies", POLICIES))
    mandatory = list(config.get("mandatory_policies", POLICIES if strict else []))
    from conductor.controller.artifacts import resolve_checkpoint
    missing = [name for name in mandatory if name not in selected]
    for name in mandatory:
        path = checkpoint if name == override_policy else config.get("checkpoints", {}).get(name)
        if name in {"conductor_sft", "conductor_preference"} and (not path or not (resolve_checkpoint(path) / "controller.json").exists()):
            missing.append(f"{name}: required checkpoint missing")
        if name == "static_supervisor" and not config.get("supervisor", {}).get("name"):
            missing.append("static_supervisor: actual frozen supervisor model missing")
    if missing:
        write_json(run.output / "policy_status.json", {"status": "failed_strict_preflight", "missing": missing})
        raise ValueError("Strict research evaluation cannot omit mandatory policies: " + "; ".join(missing))
    specialists = specialist_identity(agents, config, hash_weights=config.get("hash_specialist_weights", strict))
    from conductor.agents.audit import specialist_audit
    specialist_status = specialist_audit(agents)
    write_json(run.output / "specialist_audit.json", specialist_status)
    if strict and (not specialist_status["all_frozen"] or not specialist_status["all_identities_verifiable"]):
        raise ValueError("Strict research evaluation requires frozen specialists with verifiable model identities")
    provenance = {"data_sha256": file_sha256(config.get("data", "data/dev/tasks.jsonl")),
                  "specialist_sha256": specialists["sha256"], "config_sha256": canonical_hash(config),
                  "experiment_sha256": canonical_hash({"agents": config.get("agents", {}), "budgets": budgets,
                    "seed": config.get("seed", 42), "heldout_task_ids": split_audit["heldout_task_ids"],
                    "inference": config.get("inference", {})}), "strict_mode": strict}
    write_json(run.output / "specialists.json", specialists)
    write_json(run.output / "evaluation_provenance.json", provenance)
    statuses, trajectories, expert_stats, initial_probes = [], [], {}, {}
    for name in selected:
        policy_checkpoint = checkpoint if name == override_policy else None
        policy_checkpoint = policy_checkpoint or config.get("checkpoints", {}).get(name)
        label = name
        if name == "base_moe":
            model_config = config.get("model", {})
            label = "Tiny-MoE-Init" if model_config.get("backend", "tiny") == "tiny" else (
                "Pretrained-MoE-Random-Action-Head" if model_config.get("pretrained", True) else "HF-Random-Smoke")
        try:
            if name in {"conductor_sft", "conductor_preference"} and not policy_checkpoint:
                raise FileNotFoundError("An explicit trained checkpoint is required")
            if policy_checkpoint and not Path(policy_checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint unavailable: {policy_checkpoint}")
            policy = build_policy(name, config, checkpoint=policy_checkpoint)
        except (FileNotFoundError, RuntimeError, ValueError, ImportError) as error:
            statuses.append({"policy": name, "label": label, "status": "unavailable", "reason": str(error),
                             "checkpoint": policy_checkpoint})
            log_event("policy_unavailable", policy=name, reason=str(error))
            if strict and name in mandatory:
                write_json(run.output / "policy_status.json", statuses)
                raise RuntimeError(f"Mandatory research policy {name} is unavailable: {error}") from error
            continue
        policy_digest = checkpoint_sha256(policy_checkpoint)
        actual_model = getattr(policy, "config", config).get("model", {})
        controller_metadata = {"controller_backend": actual_model.get("backend", "tiny") if name in {"base_moe", "conductor_sft", "conductor_preference"} else name,
                               "controller_pretrained": bool(getattr(policy, "pretrained", config.get("model", {}).get("pretrained", False))) if name in {"base_moe", "conductor_sft", "conductor_preference"} else None,
                               "specialist_backend": config.get("agents", {}).get("backend", "deterministic"),
                               "checkpoint_sha256": policy_digest,
                               "controller_token_accounting": getattr(policy, "token_accounting", "none"), **provenance}
        seed_everything(config.get("seed", 42))
        if name in {"base_moe", "conductor_sft", "conductor_preference"}:
            reset = getattr(policy, "reset_expert_stats", None)
            if callable(reset):
                reset()
            states = [initial_state(task, budgets["token_budget"], budgets["agent_call_budget"]) for task in tasks]
            native_batch = getattr(policy, "batch_route", None)
            decisions = []
            probe_batch_size = config.get("probe_batch_size", 4)
            if probe_batch_size < 1:
                raise ValueError("probe_batch_size must be positive")
            for offset in range(0, len(states), probe_batch_size):
                group = states[offset:offset + probe_batch_size]
                decisions.extend(native_batch(group, budgets["k"]) if callable(native_batch) else [policy.route(state, budgets["k"]) for state in group])
            initial_probes[name] = {"states": [
                {"task_id": task.id, "task_type": task.task_type,
                 "state_sha256": hashlib.sha256(serialize_state(state).encode()).hexdigest(),
                 "decision": decision.to_dict()} for task, state, decision in zip(tasks, states, decisions)],
                "expert_stats": policy.expert_stats() if callable(getattr(policy, "expert_stats", None)) else {},
                "scope": "Identical heldout initial execution states; no agent outputs or private grader labels in controller input."}
            if callable(reset):
                reset()
            seed_everything(config.get("seed", 42))
        count = 0
        for task in tasks:
            seed_everything(config.get("seed", 42) + int(hashlib.sha256(task.id.encode()).hexdigest()[:8], 16))
            trajectory = await run_trajectory(task, policy, agents, **budgets)
            item = trajectory.to_dict()
            item["metadata"].update(policy_label=label, evaluation_budgets=budgets, **controller_metadata)
            trajectories.append(item)
            count += 1
        statuses.append({"policy": name, "label": label, "status": "measured", "task_count": count,
                         "checkpoint": policy_checkpoint, **controller_metadata})
        stats = getattr(policy, "expert_stats", None)
        if callable(stats):
            expert_stats[name] = stats()
        log_event("policy_evaluated", policy=name, task_count=count)
        # Release one policy before constructing the next large GPU backbone.
        reset = native_batch = stats = None
        del policy
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    after = specialist_identity(agents, config, hash_weights=config.get("hash_specialist_weights", strict))
    if after["sha256"] != specialists["sha256"]:
        raise RuntimeError("Frozen specialist identity changed during evaluation; research comparisons are invalid")
    rows = [trajectory_metrics(item) for item in trajectories]
    per_policy = aggregate_metrics(rows)
    per_category = aggregate_metrics(rows, ("policy", "category", "split", "generalization"))
    from conductor.metrics.aggregate import paired_differences
    measured_names = [status["policy"] for status in statuses if status["status"] == "measured"]
    comparisons = [paired_differences(rows, baseline, candidate,
                    bootstrap_samples=config.get("bootstrap_samples", 2000), seed=config.get("seed", 42))
                   for baseline in measured_names for candidate in measured_names if baseline != candidate]
    write_json(run.output / "paired_comparisons.json", comparisons)
    write_jsonl(run.output / "trajectories.jsonl", trajectories)
    write_csv(run.output / "task_metrics.csv", rows)
    write_csv(run.output / "per_policy.csv", per_policy)
    write_csv(run.output / "per_category.csv", per_category)
    write_json(run.output / "policy_status.json", statuses)
    write_json(run.output / "expert_stats.json", expert_stats)
    write_json(run.output / "expert_stats_rollout.json", expert_stats)
    write_json(run.output / "initial_state_probe.json", initial_probes)
    result = {"policies": per_policy, "categories": per_category, "policy_status": statuses,
              "fixed_budgets": budgets, "task_count": len(tasks),
              "provenance": provenance, "specialists_unchanged": True,
              "paired_comparisons": comparisons,
              "scope": "Deterministic development agents unless the agent configuration selects actual model backends."}
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/heldout.yaml")
    parser.add_argument("--checkpoint", help="Override checkpoint for trained conductor policies")
    args = parser.parse_args()
    asyncio.run(evaluate(load_config(args.config), args.checkpoint))


if __name__ == "__main__":
    main()
