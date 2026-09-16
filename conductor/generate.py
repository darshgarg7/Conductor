"""Offline trajectories, successful routing supervision, and exact-state preferences."""
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from conductor.agents import build_agents
from conductor.datasets.io import write_jsonl
from conductor.datasets.tasks import make_tasks
from conductor.orchestration.runner import initial_state, run_trajectory
from conductor.routing.policies import build_policy
from conductor.routing.serialization import serialize_state
from conductor.schema import ExecutionState, Policy, RoutingDecision, Task, Trajectory
from conductor.utils.config import load_config
from conductor.utils.runs import Run, log_event, seed_everything


class FirstDecisionPolicy:
    """Counterfactual first action followed by the same frozen continuation policy."""
    name = "counterfactual_rule_continuation"
    last_tokens = 0
    last_cost_usd = 0.0

    def __init__(self, first: RoutingDecision, continuation: Policy) -> None:
        self.first, self.continuation = first, continuation
        self.called = False

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        if not self.called:
            self.called = True
            return copy.deepcopy(self.first).validate(k)
        decision = self.continuation.route(state, k)
        self.last_tokens = int(getattr(self.continuation, "last_tokens", 0))
        self.last_cost_usd = float(getattr(self.continuation, "last_cost_usd", 0))
        return decision


def candidate_decisions(names: list[str], k: int) -> list[RoutingDecision]:
    # Parallel subsets are unordered; sequential order changes dependency visibility.
    candidates = [RoutingDecision([name], "sequential") for name in names]
    for size in range(2, min(k, len(names)) + 1):
        candidates.extend(RoutingDecision(list(group), "parallel")
                          for group in itertools.combinations(names, size))
        candidates.extend(RoutingDecision(list(group), "sequential")
                          for group in itertools.permutations(names, size))
    candidates.append(RoutingDecision([], terminate=True))
    return candidates


async def preference_records(task: Task, config: dict[str, Any], agents: dict[str, Any],
                             budgets: dict[str, int], weights: dict[str, float]) -> list[dict[str, Any]]:
    from conductor.preference.reward import reward
    # Preference credit is attached to one initial action; subsequent rounds always
    # invoke the common continuation rather than reusing that initial action.
    budgets = {**budgets, "routing_interval": 1}
    trials: list[tuple[RoutingDecision, Trajectory, float]] = []
    candidates = candidate_decisions(list(agents), budgets["k"])
    limit = max(len(agents) + 1, int(config.get("preference", {}).get("max_candidates", 16)))
    # Preserve singles/termination; sample larger decisions without task/label hints.
    if limit < len(candidates):
        randomizer = random.Random(int(config.get("seed", 42)))
        larger = randomizer.sample(candidates[len(agents):-1], limit - len(agents) - 1)
        candidates = candidates[:len(agents)] + larger + [candidates[-1]]
    canonical = serialize_state(initial_state(task, budgets["token_budget"], budgets["agent_call_budget"]))
    for first in candidates:
        continuation = build_policy("rule_based", config)
        policy = FirstDecisionPolicy(first, continuation)
        trajectory = await run_trajectory(task, policy, agents, **budgets)
        if not trajectory.steps or serialize_state(ExecutionState(**trajectory.steps[0].state)) != canonical:
            raise RuntimeError("counterfactual preferences require exactly identical initial states")
        if not trajectory.metadata.get("token_usage_known", True) or not trajectory.metadata.get("inference_cost_known", True):
            continue  # an unknown failed-backend bill must not masquerade as a cheaper route
        trials.append((first, trajectory, reward(trajectory, weights)))
    successful = [trial for trial in trials if trial[1].task_success]
    if not successful:
        return []
    chosen = max(successful, key=lambda trial: trial[2])
    records = []
    margin = float(config.get("preference", {}).get("minimum_margin", 1e-6))
    for rejected in trials:
        if chosen[0].to_dict() == rejected[0].to_dict() or chosen[2] - rejected[2] < margin:
            continue
        # Cost-only pairs cannot sacrifice quality; failures may be rejected independently.
        if rejected[1].grader_score > chosen[1].grader_score:
            continue
        records.append({
            "task_id": task.id, "split": task.split, "state": chosen[1].steps[0].state,
            "chosen": chosen[0].to_dict(), "rejected": rejected[0].to_dict(),
            "chosen_reward": chosen[2], "rejected_reward": rejected[2],
            "chosen_metrics": {**chosen[1].metadata, "wall_clock_latency": chosen[1].wall_clock_latency,
                               "estimated_inference_cost": chosen[1].estimated_inference_cost,
                               "grader_score": chosen[1].grader_score},
            "rejected_metrics": {**rejected[1].metadata, "wall_clock_latency": rejected[1].wall_clock_latency,
                                 "estimated_inference_cost": rejected[1].estimated_inference_cost,
                                 "grader_score": rejected[1].grader_score},
            "chosen_final_answer": chosen[1].final_answer, "rejected_final_answer": rejected[1].final_answer,
            "chosen_continuation_decisions": [step.decision for step in chosen[1].steps[1:]],
            "rejected_continuation_decisions": [step.decision for step in rejected[1].steps[1:]],
            "chosen_success": chosen[1].task_success, "rejected_success": rejected[1].task_success,
            "continuation_policy": "rule_based", "comparison": "identical_state_counterfactual_first_action",
            "state_canonical": canonical,
        })
    return records


async def generate(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    run = Run(config.get("output", "outputs/generation/dev"), config)
    data_config = config.get("tasks", config.get("data", {}))
    if not isinstance(data_config, dict):
        data_config = {}
    tasks = make_tasks(seed, int(config.get("train_count", data_config.get("train_count", 24))),
                       int(config.get("eval_count", data_config.get("eval_count", 12))))
    directory = Path(config.get("dataset_output", "data/dev"))
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "tasks.jsonl", (asdict(task) for task in tasks))
    agents = build_agents(config)
    orchestration = config.get("orchestration", {})
    budgets = {key: int(config.get(key, orchestration.get(key, value))) for key, value in {
        "k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}.items()}
    weights = config.get("reward", config.get("preference", {}).get("reward", {}))
    policies: list[Policy] = []
    unavailable: list[dict[str, str]] = []
    for name in config.get("policies", ["all_agent", "rule_based", "random_top_k"]):
        try:
            policies.append(build_policy(name, config))
        except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
            unavailable.append({"policy": name, "reason": str(error)})
            log_event("generation_policy_unavailable", policy=name, reason=str(error))
    if len(policies) < 2:
        raise ValueError("trajectory data requires at least two distinct generation strategies")
    counters: dict[str, Any] = {"tasks": len(tasks), "train_tasks": sum(task.split == "train" for task in tasks),
                               "eval_tasks": sum(task.split == "eval" for task in tasks), "trajectories": 0,
                               "successes": 0, "failures": 0, "sft_records": 0, "preference_records": 0,
                               "sft_excluded_non_sparse": 0, "unavailable_policies": unavailable}
    # Only a single task's records are retained in memory, allowing offline scaling.
    with (directory / "trajectories.jsonl").open("w") as trajectory_file, \
         (directory / "sft.jsonl").open("w") as sft_file, \
         (directory / "preferences.jsonl").open("w") as preference_file:
        for task in tasks:
            for policy in policies:
                trajectory = await run_trajectory(task, policy, agents, **budgets)
                trajectory_file.write(json.dumps(trajectory.to_dict(), allow_nan=False) + "\n")
                counters["trajectories"] += 1
                counters["successes" if trajectory.task_success else "failures"] += 1
                if task.split == "train" and trajectory.task_success:
                    for step in trajectory.steps:
                        decision = RoutingDecision(**step.decision)
                        if len(decision.selected_agents) > budgets["k"]:
                            counters["sft_excluded_non_sparse"] += 1
                            continue
                        if not decision.terminate and not step.agent_outputs:
                            continue
                        decision.validate(budgets["k"], tuple(agents))
                        record = {"task_id": task.id, "split": task.split, "state": step.state,
                                  "decision": step.decision, "source_policy": policy.name,
                                  "trajectory_success": True}
                        sft_file.write(json.dumps(record, allow_nan=False) + "\n")
                        counters["sft_records"] += 1
            if task.split == "train" and config.get("preference", {}).get("enabled", True):
                records = await preference_records(task, config, agents, budgets, weights)
                for record in records:
                    preference_file.write(json.dumps(record, allow_nan=False) + "\n")
                counters["preference_records"] += len(records)
            log_event("task_generated", task_id=task.id, split=task.split)
    counters.update(dataset_directory=str(directory),
                    scope="Deterministic development agents validate mechanics; no pretrained MoE research claim.",
                    preference_protocol="exact-state counterfactual first action, identical rule-based continuation",
                    training_split="train only; heldout task IDs and texts are disjoint")
    run.finish(counters)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generation.yaml")
    args = parser.parse_args()
    asyncio.run(generate(load_config(args.config)))


if __name__ == "__main__":
    main()
