"""Generate the sealed train/development corpus for coordination-v2.

The rule policy is an offline teacher, not an inference-time oracle.  Raw
trajectories retain successful and unsuccessful behavior from several fixed
policies; supervised labels come only from successful teacher executions.
Private family/source provenance is attached outside the execution state for
auditing and never becomes a controller feature.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from conductor.agents.audit import specialist_audit
from conductor.coordination.coverage import audit_partitions, canonical_sha256, write_immutable_audit
from conductor.coordination.policies import WorkflowRulePolicy
from conductor.coordination.specialists import artifacts, build_workflow_agents
from conductor.coordination.workloads import FAMILIES, make_workload
from conductor.datasets.integrity import digest, file_digest, task_seed
from conductor.evaluation.io import write_jsonl
from conductor.orchestration.runner import run_trajectory
from conductor.routing.policies import AllAgentPolicy, RandomTopKPolicy
from conductor.schema import ExecutionState, RoutingDecision, Task, Trajectory
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json


class StopImmediately:
    """Negative control used only to retain a clear unsuccessful trajectory."""

    name = "stop_immediately"
    last_tokens = 0
    last_cost_usd = 0.0

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        del state, k
        return RoutingDecision([], terminate=True)


def _state_kinds(state: ExecutionState, decision: RoutingDecision) -> list[str]:
    records = artifacts(state)
    kinds = {record.get("kind") for record in records}
    statuses = {output.get("metadata", {}).get("status") for output in state.previous_agent_outputs}
    tags = {"initial" if state.current_step == 0 else "intermediate"}
    if len(decision.selected_agents) > 1 and decision.execution_mode == "sequential":
        tags.add("handoff")
    if "tool_failure" in kinds or statuses & {"blocked", "failed", "error"}:
        tags.add("failure")
    if records and not {"verification", "execution"} & kinds:
        tags.add("partial")
    if decision.terminate:
        tags.add("stopping")
    return sorted(tags)


def _supervision(task: Task, trajectory: Trajectory) -> list[dict[str, Any]]:
    if not trajectory.task_success:
        raise ValueError(f"offline teacher failed task {task.id}; refusing to emit labels")
    records: list[dict[str, Any]] = []
    for step in trajectory.steps:
        state = ExecutionState(**step.state)
        decision = RoutingDecision(**step.decision)
        records.append({
            "task_id": task.id,
            "split": task.split,
            "source_group": task.metadata["source_group"],
            "family": task.metadata["family"],
            "dependency_stages": task.metadata["dependency_stages"],
            "structure_id": task.metadata["structure_id"],
            "state_kinds": _state_kinds(state, decision),
            "state": step.state,
            "decision": step.decision,
            "teacher": "public_state_rules_v1",
            "teacher_trajectory_sha256": canonical_sha256(trajectory.to_dict()),
        })
    return records


def _policy(name: str, seed: int) -> Any:
    if name == "public_state_rules":
        return WorkflowRulePolicy()
    if name == "random_top_k":
        return RandomTopKPolicy(seed=seed, rounds=6)
    if name == "iterative_all_agent_sequential":
        return AllAgentPolicy(rounds=2, execution_mode="sequential")
    if name == "all_agent_parallel":
        return AllAgentPolicy(rounds=2, execution_mode="parallel")
    if name == "stop_immediately":
        return StopImmediately()
    raise ValueError(f"unknown coordination data policy {name!r}")


async def _capability_isolation(tasks: list[Task], agents: dict[str, Any], budgets: dict[str, int]) -> dict[str, Any]:
    """Verify that repeatedly invoking one capability cannot solve a workflow."""
    outcomes: dict[str, dict[str, int]] = {}
    for name in agents:
        class SingleCapability:
            last_tokens = 0
            last_cost_usd = 0.0

            def __init__(self, selected: str) -> None:
                self.name = f"single_capability_{selected}"
                self.selected = selected

            def route(self, state: ExecutionState, k: int) -> RoutingDecision:
                if state.current_step >= budgets["max_rounds"] - 1:
                    return RoutingDecision([], terminate=True)
                return RoutingDecision([self.selected])

        values = [await run_trajectory(task, SingleCapability(name), agents, **budgets) for task in tasks]
        outcomes[name] = {"tasks": len(values), "successes": sum(item.task_success for item in values)}
    if any(value["successes"] for value in outcomes.values()):
        raise RuntimeError("capability isolation failed: one specialist solved a multi-capability workflow")
    return {"scope": "repeat one frozen public capability under the common task budget",
            "per_specialist": outcomes, "all_single_capability_successes_zero": True}


async def generate(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config.get("dataset_output", "data/coordination-v2"))
    run_output = Path(config.get("output", "outputs/coordination-v2/data-generation"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a fresh coordination dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 314159))
    seed_everything(seed)
    train_count = int(config.get("train_count", 72))
    development_count = int(config.get("development_count", 36))
    tasks, stores = make_workload(seed=seed, train_count=train_count, dev_count=development_count)
    agents = build_workflow_agents(stores)
    budgets = {"k": int(config.get("routing", {}).get("k", 3)),
               "max_rounds": int(config.get("orchestration", {}).get("max_rounds", 6)),
               "token_budget": int(config.get("orchestration", {}).get("token_budget", 16384)),
               "agent_call_budget": int(config.get("orchestration", {}).get("agent_call_budget", 12)),
               "routing_interval": 1}
    if budgets != {"k": 3, "max_rounds": 6, "token_budget": 16384,
                   "agent_call_budget": 12, "routing_interval": 1}:
        raise ValueError("coordination-v2 data generation uses the protocol's frozen budgets")
    policy_names = list(config.get("policies", ["public_state_rules", "random_top_k",
        "iterative_all_agent_sequential", "all_agent_parallel", "stop_immediately"]))
    if "public_state_rules" not in policy_names or len(set(policy_names)) < 3:
        raise ValueError("generation requires the teacher and at least two additional policies")
    run = Run(run_output, config)
    trajectories: list[dict[str, Any]] = []
    supervision: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for task in tasks:
        for name in policy_names:
            isolated_seed = task_seed(seed, task.id, name)
            seed_everything(isolated_seed)
            trajectory = await run_trajectory(task, _policy(name, isolated_seed), agents, **budgets)
            trajectory.metadata.update(generation_seed=isolated_seed, generation_policy=name,
                                       workflow_family=task.metadata["family"],
                                       structure_id=task.metadata["structure_id"])
            trajectories.append(trajectory.to_dict())
            if name == "public_state_rules":
                supervision[task.split].extend(_supervision(task, trajectory))
    required_cells = ["state:initial", "state:intermediate", "state:handoff", "state:failure",
                      "state:stopping", "count:0", "count:1", "count:2", "count:3",
                      "mode:parallel", "mode:sequential"]
    coverage = audit_partitions(supervision["train"], supervision["dev"], k=budgets["k"],
                                required_families=FAMILIES,
                                minimum_dev_groups=int(config.get(
                                    "minimum_development_groups_per_family", development_count // len(FAMILIES))),
                                required_cells=required_cells)
    isolation = await _capability_isolation(tasks, agents, budgets)
    write_jsonl(output / "tasks.jsonl", [asdict(task) for task in tasks])
    write_jsonl(output / "trajectories.jsonl", trajectories)
    write_jsonl(output / "sft_train.jsonl", supervision["train"])
    write_jsonl(output / "sft_development.jsonl", supervision["dev"])
    write_json(output / "public_stores.json", stores.to_dict())
    write_json(output / "capability_isolation.json", isolation)
    coverage_sha = write_immutable_audit(output / "coverage_audit.json", coverage)
    files = {path.name: file_digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    manifest = {
        "schema_version": "coordination-v2-data-v1",
        "seed": seed,
        "budgets": budgets,
        "task_count": len(tasks),
        "train_tasks": sum(task.split == "train" for task in tasks),
        "development_tasks": sum(task.split == "dev" for task in tasks),
        "trajectory_count": len(trajectories),
        "trajectory_successes": sum(item["task_success"] for item in trajectories),
        "sft_train_records": len(supervision["train"]),
        "sft_development_records": len(supervision["dev"]),
        "policy_counts": {name: sum(item["metadata"]["generation_policy"] == name for item in trajectories)
                          for name in policy_names},
        "public_store_sha256": stores.corpus_sha256,
        "specialist_audit": specialist_audit(agents),
        "coverage_audit_sha256": coverage_sha,
        "files": files,
        "task_inventory_sha256": digest([asdict(task) for task in tasks]),
        "controller_state_excludes_private_task_metadata": True,
        "supervision_source": "successful public-state rule teacher trajectories",
    }
    write_json(output / "manifest.json", manifest)
    result = {**manifest, "dataset_output": str(output), "coverage_gate_passed": True,
              "capability_isolation_passed": True}
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = asyncio.run(generate(load_config(args.config)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
