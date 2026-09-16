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
from conductor.metrics.aggregate import aggregate_metrics, trajectory_metrics
from conductor.utils.config import load_config
from conductor.utils.runs import Run, log_event, seed_everything, write_json

POLICIES = ("all_agent", "rule_based", "random_top_k", "static_supervisor", "base_moe",
            "conductor_sft", "conductor_preference")


def checkpoint_policy(checkpoint: str) -> str:
    stage = json.loads((Path(checkpoint) / "controller.json").read_text()).get("stage")
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
    statuses, trajectories, expert_stats, initial_probes = [], [], {}, {}
    for name in config.get("policies", POLICIES):
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
            continue
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
            trajectory = await run_trajectory(task, policy, agents, **budgets)
            item = trajectory.to_dict()
            item["metadata"].update(policy_label=label, evaluation_budgets=budgets)
            trajectories.append(item)
            count += 1
        statuses.append({"policy": name, "label": label, "status": "measured", "task_count": count,
                         "checkpoint": policy_checkpoint})
        stats = getattr(policy, "expert_stats", None)
        if callable(stats):
            expert_stats[name] = stats()
        log_event("policy_evaluated", policy=name, task_count=count)
    rows = [trajectory_metrics(item) for item in trajectories]
    per_policy = aggregate_metrics(rows)
    per_category = aggregate_metrics(rows, ("policy", "category", "split", "generalization"))
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
              "scope": "Deterministic development agents unless the agent configuration selects actual model backends."}
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/dev.yaml")
    parser.add_argument("--checkpoint", help="Override checkpoint for trained conductor policies")
    args = parser.parse_args()
    asyncio.run(evaluate(load_config(args.config), args.checkpoint))


if __name__ == "__main__":
    main()
