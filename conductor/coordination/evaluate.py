"""Evaluate workflow controllers on one immutable task inventory."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from conductor.agents import build_agents
from conductor.controller.factory import build_controller
from conductor.datasets.integrity import file_digest, iter_records
from conductor.evaluation.io import write_csv, write_jsonl
from conductor.metrics.aggregate import aggregate_metrics, paired_differences, trajectory_metrics
from conductor.orchestration.runner import run_trajectory
from conductor.routing.policies import AllAgentPolicy, RandomTopKPolicy, StaticSupervisorPolicy
from conductor.schema import Task
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json

from .policies import WorkflowRulePolicy
from .workloads import FAMILIES


def _checkpoint_policy(identifier: str, checkpoint: str, config: dict[str, Any]) -> Any:
    policy = build_controller(config, checkpoint)
    policy.name = identifier
    return policy


def build_workflow_policy(specification: dict[str, Any], config: dict[str, Any], seed: int) -> Any:
    identifier = specification["id"]
    kind = specification.get("kind", identifier)
    if kind == "public_state_rules":
        policy = WorkflowRulePolicy()
    elif kind == "random_top_k":
        policy = RandomTopKPolicy(seed, rounds=int(config["orchestration"]["max_rounds"]))
    elif kind == "iterative_all_agent_sequential":
        policy = AllAgentPolicy(rounds=int(specification.get("rounds", 2)), execution_mode="sequential")
    elif kind == "all_agent_parallel":
        policy = AllAgentPolicy(rounds=int(specification.get("rounds", 2)), execution_mode="parallel")
    elif kind == "static_supervisor":
        policy = StaticSupervisorPolicy(config.get("supervisor", {}))
    elif kind in {"checkpoint", "sparse_linear_router", "small_mlp_router", "conductor_sft",
                  "conductor_preference", "frozen_backbone_trained_head"}:
        checkpoint = specification.get("checkpoint")
        if not checkpoint:
            raise FileNotFoundError(f"{identifier} requires an explicit checkpoint")
        policy = _checkpoint_policy(identifier, checkpoint, config)
    elif kind == "pretrained_moe_random_action_head":
        policy = build_controller(config)
    else:
        raise ValueError(f"unknown workflow evaluation policy kind {kind!r}")
    # All-Agent must keep its runner-recognized name for the documented per-step
    # k exception.  The returned trajectory is relabeled after execution.
    if kind not in {"iterative_all_agent_sequential", "all_agent_parallel"}:
        policy.name = identifier
    return policy


def _tasks(path: str | Path, splits: set[str]) -> list[Task]:
    values = [Task(**record) for record in iter_records(path) if record.get("split") in splits]
    if not values:
        raise ValueError(f"no tasks found for explicit splits {sorted(splits)}")
    if len({task.id for task in values}) != len(values):
        raise ValueError("evaluation inventory has duplicate task IDs")
    return values


def _breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "family": defaultdict(list), "dependency_stage": defaultdict(list), "structure": defaultdict(list)}
    for row in rows:
        grouped["family"][str(row["family"])].append(row)
        grouped["dependency_stage"][str(row["dependency_stages"])].append(row)
        grouped["structure"]["ood" if row["structure_ood"] else "seen"].append(row)
    return {dimension: {key: {"tasks": len(group),
                               "success_rate": sum(item["task_success"] for item in group) / len(group),
                               "mean_agent_calls": statistics.fmean(item["agent_calls"] for item in group)}
                         for key, group in sorted(values.items())}
            for dimension, values in grouped.items()}


async def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    seed = int(effective.get("seed", 42))
    seed_everything(seed)
    task_path = effective["data"]
    splits = set(effective.get("splits", ["dev"]))
    tasks = _tasks(task_path, splits)
    agents = build_agents(effective)
    orchestration = effective.get("orchestration", {})
    budgets = {"k": int(effective.get("routing", {}).get("k", 3)),
               "max_rounds": int(orchestration.get("max_rounds", 6)),
               "token_budget": int(orchestration.get("token_budget", 16384)),
               "agent_call_budget": int(orchestration.get("agent_call_budget", 12)),
               "routing_interval": 1}
    repetitions = int(effective.get("latency_repetitions", 1))
    if repetitions < 1:
        raise ValueError("latency_repetitions must be positive")
    run = Run(effective.get("output", "outputs/coordination-v2/evaluation"), effective)
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    expert_stats: dict[str, Any] = {}
    specs = effective.get("policies", [])
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("evaluation needs nonempty uniquely identified policy specifications")
    for spec in specs:
        identifier = spec["id"]
        try:
            policy = build_workflow_policy(spec, effective, seed)
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            statuses.append({"policy": identifier, "status": "unavailable", "reason": str(error)})
            continue
        policy_rows: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            seed_everything(seed)
            if repetition and spec.get("kind", identifier) == "random_top_k":
                policy = build_workflow_policy(spec, effective, seed)
            for task in tasks:
                task_seed = seed + int(hashlib.sha256(task.id.encode()).hexdigest()[:8], 16)
                seed_everything(task_seed)
                trajectory = await run_trajectory(task, policy, agents, **budgets)
                item = trajectory.to_dict()
                item["policy"] = identifier
                item["metadata"].update(evaluation_seed=seed, latency_repetition=repetition,
                                        data_sha256=file_digest(task_path),
                                        controller_token_accounting=getattr(policy, "token_accounting", "none"))
                metric = trajectory_metrics(item)
                metric.update(family=task.metadata["family"],
                              dependency_stages=task.metadata["dependency_stages"],
                              structure_ood=bool(task.metadata.get("structure_ood", False)),
                              repetition=repetition)
                timing.append({key: metric[key] for key in ("task_id", "policy", "repetition",
                              "wall_clock_seconds", "controller_seconds")})
                if repetition == 0:
                    trajectories.append(item)
                    rows.append(metric)
                    policy_rows.append(metric)
        statuses.append({"policy": identifier, "status": "measured", "tasks": len(policy_rows),
                         "checkpoint": spec.get("checkpoint")})
        stats = getattr(policy, "expert_stats", None)
        if callable(stats):
            expert_stats[identifier] = stats()
    summaries = aggregate_metrics(rows)
    per_family = aggregate_metrics(rows, ("policy", "family"))
    per_stage = aggregate_metrics(rows, ("policy", "dependency_stages"))
    measured = [status["policy"] for status in statuses if status["status"] == "measured"]
    comparisons = [paired_differences(rows, baseline, candidate,
                                      bootstrap_samples=int(effective.get("bootstrap_samples", 1999)), seed=seed)
                   for baseline in measured for candidate in measured if baseline != candidate]
    policy_breakdowns = {name: _breakdowns([row for row in rows if row["policy"] == name]) for name in measured}
    timing_summary = {name: {
        "samples": len(values),
        "p50_wall_clock_seconds": float(np.percentile(values, 50)),
        "p95_wall_clock_seconds": float(np.percentile(values, 95)),
        "mean_wall_clock_seconds": statistics.fmean(values),
    } for name in measured if (values := [row["wall_clock_seconds"] for row in timing if row["policy"] == name])}
    write_jsonl(run.output / "trajectories.jsonl", trajectories)
    write_csv(run.output / "task_metrics.csv", rows)
    write_csv(run.output / "timing_repetitions.csv", timing)
    write_csv(run.output / "per_policy.csv", summaries)
    write_csv(run.output / "per_family.csv", per_family)
    write_csv(run.output / "per_dependency_stage.csv", per_stage)
    write_json(run.output / "policy_status.json", statuses)
    write_json(run.output / "paired_comparisons.json", comparisons)
    write_json(run.output / "expert_stats.json", expert_stats)
    result = {"task_count": len(tasks), "splits": sorted(splits), "budgets": budgets,
              "latency_repetitions": repetitions,
              "latency_protocol": "repetition zero cold for process; later repetitions reuse controller caches",
              "policies": summaries, "policy_status": statuses, "per_family": per_family,
              "per_dependency_stage": per_stage, "breakdowns": policy_breakdowns,
              "timing": timing_summary, "paired_comparisons": comparisons,
              "data_sha256": file_digest(task_path), "workflow_families": list(FAMILIES)}
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(load_config(args.config))), indent=2))


if __name__ == "__main__":
    main()
