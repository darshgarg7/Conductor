"""Construct DPO pairs from measured mistakes made by a saved SFT policy."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

from conductor.agents.audit import specialist_audit
from conductor.controller.actions import ActionCatalog
from conductor.controller.factory import build_controller
from conductor.coordination.coverage import (audit_preferences, canonical_sha256, preference_pair_sha256,
                                             write_immutable_audit)
from conductor.coordination.policies import WorkflowRulePolicy
from conductor.coordination.specialists import artifacts, build_workflow_agents
from conductor.coordination.workloads import PublicStores
from conductor.datasets.integrity import digest, file_digest, iter_records, task_seed
from conductor.evaluation.io import write_jsonl
from conductor.generate import FirstDecisionPolicy
from conductor.orchestration.runner import run_trajectory
from conductor.preference.reward import reward
from conductor.schema import ExecutionState, RoutingDecision, Task
from conductor.training.runner import checkpoint_digest
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json


def _tags(state: ExecutionState, decision: RoutingDecision) -> list[str]:
    records = artifacts(state)
    kinds = {record.get("kind") for record in records}
    statuses = {output.get("metadata", {}).get("status") for output in state.previous_agent_outputs}
    result = {"initial" if state.current_step == 0 else "intermediate"}
    if "tool_failure" in kinds or statuses & {"blocked", "failed", "error"}:
        result.add("failure")
    if records and not {"verification", "execution"} & kinds:
        result.add("partial")
    if len(decision.selected_agents) > 1 and decision.execution_mode == "sequential":
        result.add("handoff")
    if decision.terminate:
        result.add("stopping")
    return sorted(result)


def _branch_metrics(trajectory: Any) -> dict[str, Any]:
    return {"task_success": trajectory.task_success, "grader_score": trajectory.grader_score,
            "agent_calls": trajectory.metadata["agent_activations"],
            "total_tokens": trajectory.metadata["total_tokens"],
            "communication_edges": len(trajectory.communication_graph),
            "wall_clock_latency": trajectory.wall_clock_latency,
            "stop_reason": trajectory.metadata["stop_reason"]}


async def generate_preferences(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    checkpoint = checkpoint or config.get("checkpoint")
    if not checkpoint:
        raise ValueError("on-policy preference generation requires an SFT checkpoint")
    actor = build_controller(config, checkpoint)
    if actor.stage != "sft":
        raise ValueError("on-policy preferences require a saved SFT controller")
    output = Path(config.get("output", "data/coordination-v2/preferences"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"use a fresh on-policy preference directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    tasks = [Task(**record) for record in iter_records(config["tasks"])
             if record.get("split") in {"train", "dev"}]
    stores = PublicStores.from_dict(json.loads(Path(config["stores"]).read_text()))
    agents = build_workflow_agents(stores)
    budgets = {"k": 3, "max_rounds": 6, "token_budget": 16384,
               "agent_call_budget": 12, "routing_interval": 1}
    weights = {"token_weight": .05, "latency_weight": 0.0, "agent_call_weight": .04,
               "communication_weight": .01, "token_scale": 1000.0, "latency_scale": 1.0,
               "agent_call_scale": 12.0, "communication_scale": 20.0,
               **config.get("reward", {})}
    continuation_identity = digest({
        "policy": "public_state_rules_v1", "implementation": inspect.getsource(WorkflowRulePolicy),
        "specialists": specialist_audit(agents)["fingerprint"], "budgets": budgets, "reward": weights})
    actor_digest = checkpoint_digest(checkpoint)
    catalog = ActionCatalog(3)
    run = Run(config.get("run_output", str(output) + "-run"), config, checkpoint)
    pairs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    rollouts: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    seen_pairs: set[str] = set()
    margin = float(config.get("minimum_margin", 1e-6))
    for task in tasks:
        isolated_seed = task_seed(seed, task.id, actor_digest)
        seed_everything(isolated_seed)
        actor_trajectory = await run_trajectory(task, actor, agents, **budgets)
        actor_item = actor_trajectory.to_dict()
        actor_item["metadata"].update(preference_actor_sha256=actor_digest, preference_rollout_seed=isolated_seed)
        rollouts.append(actor_item)
        requested = actor_trajectory.metadata.get("requested_routing_decisions", [])
        clipped = set(actor_trajectory.metadata.get("budget_clipped_steps", []))
        if len(requested) != len(actor_trajectory.steps):
            raise RuntimeError("actor rollout did not retain one requested decision per executed step")
        for step_index, step in enumerate(actor_trajectory.steps):
            state = ExecutionState(**step.state)
            if state.current_step in clipped:
                exclusions["budget_clipped_actor_action"] += 1
                continue
            rejected = RoutingDecision(**requested[step_index])
            chosen = WorkflowRulePolicy().route(state, budgets["k"])
            if catalog.index(chosen) == catalog.index(rejected):
                exclusions["actor_matches_rescue"] += 1
                continue
            remaining_rounds = budgets["max_rounds"] - state.current_step
            if remaining_rounds < 1:
                exclusions["no_round_budget_for_counterfactual"] += 1
                continue
            branch_budgets = {**budgets, "max_rounds": remaining_rounds}
            branches = []
            for label, first in (("chosen", chosen), ("rejected", rejected)):
                branch_seed = task_seed(seed, task.id, canonical_sha256({"state": step.state, "first": first.to_dict()}))
                seed_everything(branch_seed)
                trajectory = await run_trajectory(
                    task, FirstDecisionPolicy(first, WorkflowRulePolicy()), agents,
                    initial_execution_state=state, **branch_budgets)
                value = reward(trajectory, weights)
                branches.append((label, first, trajectory, value, branch_seed))
                trials.append({"task_id": task.id, "split": task.split, "source_group": task.metadata["source_group"],
                               "actor_checkpoint_sha256": actor_digest, "continuation_sha256": continuation_identity,
                               "branch": label, "first_decision": first.to_dict(), "reward": value,
                               "seed": branch_seed, "trajectory": trajectory.to_dict()})
            _, chosen_action, chosen_trajectory, chosen_reward, chosen_seed = branches[0]
            _, rejected_action, rejected_trajectory, rejected_reward, rejected_seed = branches[1]
            if not chosen_trajectory.task_success:
                exclusions["rescue_did_not_succeed"] += 1
                continue
            if rejected_trajectory.grader_score > chosen_trajectory.grader_score:
                exclusions["rescue_reduced_quality"] += 1
                continue
            if chosen_reward - rejected_reward < margin:
                exclusions["insufficient_reward_margin"] += 1
                continue
            record = {
                "task_id": task.id, "split": task.split, "source_group": task.metadata["source_group"],
                "family": task.metadata["family"], "dependency_stages": task.metadata["dependency_stages"],
                "structure_id": task.metadata["structure_id"], "state_kinds": _tags(state, rejected),
                "state": step.state, "chosen": chosen_action.to_dict(), "rejected": rejected_action.to_dict(),
                "chosen_reward": chosen_reward, "rejected_reward": rejected_reward,
                "chosen_metrics": _branch_metrics(chosen_trajectory),
                "rejected_metrics": _branch_metrics(rejected_trajectory),
                "preference_source": "on_policy_mistake", "actor_checkpoint_sha256": actor_digest,
                "actor_policy": actor.name, "actor_rollout_seed": isolated_seed,
                "chosen_branch_seed": chosen_seed, "rejected_branch_seed": rejected_seed,
                "continuation_sha256": continuation_identity,
                "chosen_continuation_sha256": continuation_identity,
                "rejected_continuation_sha256": continuation_identity,
                "comparison": "identical_public_state_counterfactual_first_action_shared_rule_continuation",
                "reward_scope": "incremental continuation after common execution prefix",
            }
            pair_id = preference_pair_sha256(record["state"], chosen_action, rejected_action,
                                              continuation_identity, budgets["k"])
            if pair_id in seen_pairs:
                exclusions["duplicate_pair"] += 1
                continue
            seen_pairs.add(pair_id)
            record["pair_sha256"] = pair_id
            pairs[task.split].append(record)
    train_audit = audit_preferences(pairs["train"])
    development_audit = audit_preferences(pairs["dev"])
    train_tasks = {record["task_id"] for record in pairs["train"]}
    development_tasks = {record["task_id"] for record in pairs["dev"]}
    train_groups = {record["source_group"] for record in pairs["train"]}
    development_groups = {record["source_group"] for record in pairs["dev"]}
    if train_tasks & development_tasks or train_groups & development_groups:
        raise ValueError("preference fitting and development partitions overlap")
    write_jsonl(output / "train.jsonl", pairs["train"])
    write_jsonl(output / "development.jsonl", pairs["dev"])
    write_jsonl(output / "actor_rollouts.jsonl", rollouts)
    write_jsonl(output / "counterfactual_trials.jsonl", trials)
    audit = {"train": train_audit, "development": development_audit,
             "exclusions": dict(sorted(exclusions.items())),
             "partition_audit": {"task_disjoint": True, "source_group_disjoint": True},
             "state_kind_counts": {split: dict(Counter(tag for record in values
                                                        for tag in record["state_kinds"]))
                                   for split, values in pairs.items()}}
    audit_sha = write_immutable_audit(output / "preference_audit.json", audit)
    status = "ready" if pairs["train"] and pairs["dev"] else "blocked_insufficient_pairs"
    result = {"status": status, "train_pairs": len(pairs["train"]),
              "development_pairs": len(pairs["dev"]), "actor_checkpoint_sha256": actor_digest,
              "continuation_sha256": continuation_identity, "preference_audit_sha256": audit_sha,
              "tasks_sha256": file_digest(config["tasks"]), "stores_sha256": file_digest(config["stores"]),
              "latency_weight": weights["latency_weight"], "exclusions": dict(sorted(exclusions.items())),
              "fit_partition": "train only", "development_partition": "diagnostic and validation only",
              "final_partition_used": False}
    write_json(output / "manifest.json", result)
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(generate_preferences(load_config(args.config), args.checkpoint)), indent=2))


if __name__ == "__main__":
    main()
