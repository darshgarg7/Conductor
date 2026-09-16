"""Offline trajectories, successful routing supervision, and exact-state preferences."""
from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import random
from dataclasses import asdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from conductor.agents import build_agents
from conductor.datasets.integrity import (Journal, SingleWriterLock, digest, durable_json, durable_jsonl,
                                         file_digest, fsync_directory, iter_records, shard_for, task_seed)
from conductor.datasets.source import external_tasks
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
                             budgets: dict[str, int], weights: dict[str, float],
                             execution_state: ExecutionState | None = None, trial_journal: Journal | None = None,
                             source_identity: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    state = copy.deepcopy(execution_state) if execution_state is not None else initial_state(task, budgets["token_budget"], budgets["agent_call_budget"])
    canonical = serialize_state(state)
    for first in candidates:
        trial_id = digest({"task": asdict(task), "state": state.to_dict(), "first": first.to_dict(),
                           "configuration": digest(config), "source_identity": source_identity,
                           "continuation": {"policy": "rule_based", "budgets": budgets}})
        if trial_journal is not None and trial_id in trial_journal.ids:
            trajectory = Trajectory.from_dict(trial_journal.get(trial_id))
            # Storage checksums/IDs are not measured execution metadata. Keep
            # replayed preference records byte-equivalent to their first run.
            trajectory.metadata.pop("record_id", None)
            trajectory.metadata.pop("record_checksum_sha256", None)
        else:
            isolated_seed = task_seed(int(config.get("seed", 42)), task.id, trial_id)
            seed_everything(isolated_seed)
            continuation = build_policy("rule_based", config)
            policy = FirstDecisionPolicy(first, continuation)
            trajectory = await run_trajectory(task, policy, agents, **budgets, initial_execution_state=state)
            trajectory.metadata.update(preference_trial_seed=isolated_seed, preference_trial_id=trial_id,
                                       source_identity=source_identity)
            if trial_journal is not None:
                trial_journal.append(trajectory.to_dict(), trial_id)
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
            "counterfactual_step": state.current_step, "reward_scope": "incremental continuation; common prefix cancels",
            "continuation_configuration_sha256": digest({"policy": "rule_based", "routing": config.get("routing", {}), "budgets": budgets}),
            "state_canonical": canonical,
        })
    return records


async def generate(config: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    shards, index = int(effective.get("jobshards", 1)), int(effective.get("shard_index", 0))
    if not 0 <= index < shards:
        raise ValueError("shard_index must be in [0, jobshards)")
    directory = Path(effective.get("dataset_output", "data/dev"))
    if shards > 1:
        directory = directory / "shards" / f"shard-{index:05d}-of-{shards:05d}"
    with SingleWriterLock(directory / ".generation.lock"):
        return await _generate_locked(effective)


async def _generate_locked(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    shards, index = int(config.get("jobshards", 1)), int(config.get("shard_index", 0))
    if not 0 <= index < shards:
        raise ValueError("shard_index must be in [0, jobshards)")
    suffix = f"shard-{index:05d}-of-{shards:05d}"
    directory = Path(config.get("dataset_output", "data/dev"))
    output = Path(config.get("output", "outputs/generation/dev"))
    if shards > 1:
        directory, output = directory / "shards" / suffix, output / suffix
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    source = config.get("task_source", {})
    if isinstance(source, str):
        source = {"path": source}
    data_config = config.get("tasks", {})
    if source.get("path"):
        source_path = Path(source["path"])
        source_hash = file_digest(source_path)
        task_factory = lambda: external_tasks(source_path)
        source_kind = source.get("kind", "external_unverified")
    else:
        generated = make_tasks(seed, int(config.get("train_count", data_config.get("train_count", 24))),
                               int(config.get("eval_count", data_config.get("eval_count", 12))))
        source_hash = digest([asdict(task) for task in generated])
        task_factory = lambda: iter(generated)
        source_kind = "synthetic_development_templates"
    identity = {"configuration_sha256": digest(config), "source_sha256": source_hash,
                "source_kind": source_kind, "jobshards": shards, "shard_index": index}
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("resume configuration/source mismatch; use a new output directory")
        if previous.get("task_inventory_sha256") and file_digest(directory / "tasks.jsonl") != previous["task_inventory_sha256"]:
            raise ValueError("resume task inventory checksum mismatch")
        if not config.get("resume", True):
            raise ValueError("existing dataset requires resume=true or a new directory")
    elif (directory / "trajectories.jsonl").exists() and (directory / "trajectories.jsonl").stat().st_size:
        raise ValueError("refusing to overwrite a historical dataset without a recovery manifest")
    # Validate ALL source tasks, including other shards and held-out splits,
    # exactly while constructing a durable inventory. Execute only this snapshot
    # afterward, rather than rereading a mutable external source for each task.
    task_count = 0
    def inventory_records() -> Any:
        nonlocal task_count
        for task in task_factory():
            task_count += 1
            if shard_for(task.id, shards) == index:
                yield asdict(task)
    candidate = directory / ".inventory-candidate.jsonl"
    try:
        durable_jsonl(candidate, inventory_records())
        if source.get("path") and file_digest(source_path) != source_hash:
            raise ValueError("external source changed while validating the task snapshot")
        inventory = directory / "tasks.jsonl"
        if inventory.exists():
            if file_digest(candidate) != file_digest(inventory):
                raise ValueError("durable task inventory does not match the validated task source")
        else:
            candidate.replace(inventory)
            fsync_directory(directory)
    finally:
        candidate.unlink(missing_ok=True)
    assigned = lambda: (Task(**record) for record in iter_records(directory / "tasks.jsonl"))
    if not manifest_path.exists():
        durable_json(manifest_path, {**identity, "completed_task_ids": [],
                     "task_inventory_sha256": file_digest(directory / "tasks.jsonl")})
    run = Run(output, config)
    agents = build_agents(config)
    orchestration = config.get("orchestration", {})
    budgets = {key: int(config.get(key, orchestration.get(key, value))) for key, value in {
        "k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}.items()}
    if orchestration.get("hard_token_budget", False):
        budgets["hard_token_budget"] = True
    if orchestration.get("agent_timeout_seconds") is not None:
        budgets["agent_timeout_seconds"] = float(orchestration["agent_timeout_seconds"])
    weights = config.get("reward", config.get("preference", {}).get("reward", {}))
    policies, unavailable = [], []
    for name in config.get("policies", ["all_agent", "rule_based", "random_top_k"]):
        try:
            policies.append(build_policy(name, config))
        except (ValueError, ImportError, FileNotFoundError, RuntimeError) as error:
            unavailable.append({"policy": name, "reason": str(error)})
    if len({policy.name for policy in policies}) < 2:
        raise ValueError("trajectory data requires at least two distinct generation strategies")
    from conductor.agents.audit import specialist_audit
    fingerprint = specialist_audit(agents)["fingerprint"]
    effective_policies = sorted(policy.name for policy in policies)
    if previous.get("specialist_fingerprint") not in {None, fingerprint} or previous.get("effective_policies", effective_policies) != effective_policies:
        raise ValueError("resume frozen specialist identities/effective policies changed; use a new dataset directory")
    completed = set(previous.get("completed_task_ids", [])) if manifest_path.exists() else set()
    counters: dict[str, Any] = {"tasks": 0, "source_tasks": task_count, "train_tasks": 0, "eval_tasks": 0,
                               "trajectories": 0, "successes": 0, "failures": 0, "sft_records": 0,
                               "preference_records": 0, "sft_excluded_non_sparse": 0,
                               "unavailable_policies": unavailable, "resumed_trajectories": 0}
    with ExitStack() as stack:
        trajectories = stack.enter_context(Journal(directory / "trajectories.jsonl"))
        sft = stack.enter_context(Journal(directory / "sft.jsonl"))
        preferences = stack.enter_context(Journal(directory / "preferences.jsonl"))
        trial_journal = stack.enter_context(Journal(directory / "preference_trials.jsonl"))
        for task in assigned():
            counters["tasks"] += 1
            counters["train_tasks" if task.split == "train" else "eval_tasks"] += 1
            states = {}
            for policy in policies:
                record_id = digest({"task_id": task.id, "policy": policy.name, "source": source_hash,
                                    "configuration": identity["configuration_sha256"]})
                if record_id in trajectories.ids:
                    trajectory = Trajectory.from_dict(trajectories.get(record_id))
                    if trajectory.metadata.get("specialist_fingerprint") != fingerprint:
                        raise ValueError("recovered trajectory has a different frozen specialist fingerprint")
                    counters["resumed_trajectories"] += 1
                else:
                    isolated_seed = task_seed(seed, task.id, policy.name)
                    seed_everything(isolated_seed)
                    if hasattr(policy, "rng"):
                        policy.rng.seed(isolated_seed)
                    trajectory = await run_trajectory(task, policy, agents, **budgets)
                    trajectory.metadata.update(source_sha256=source_hash, source_kind=source_kind,
                                               generation_seed=isolated_seed, task_sha256=digest(asdict(task)),
                                               generation_configuration_sha256=identity["configuration_sha256"],
                                               jobshards=shards, shard_index=index, generation_git_commit=run.record["git_commit"])
                    trajectories.append(trajectory.to_dict(), record_id)
                counters["trajectories"] += 1
                counters["successes" if trajectory.task_success else "failures"] += 1
                for step in trajectory.steps:
                    states.setdefault(digest(step.state), step.state)
                if task.split == "train" and trajectory.task_success:
                    for step_index, step in enumerate(trajectory.steps):
                        decision = RoutingDecision(**step.decision)
                        if len(decision.selected_agents) > budgets["k"]:
                            counters["sft_excluded_non_sparse"] += 1
                            continue
                        if not decision.terminate and not step.agent_outputs:
                            continue
                        decision.validate(budgets["k"], tuple(agents))
                        sft.append({"task_id": task.id, "split": task.split, "state": step.state,
                                    "decision": step.decision, "source_policy": policy.name,
                                    "trajectory_success": True}, digest({"trajectory": record_id, "step": step_index}))
            if task.split == "train" and config.get("preference", {}).get("enabled", True) and task.id not in completed:
                max_states = int(config.get("preference", {}).get("max_states_per_task", 1))
                ordered = sorted(states.values(), key=lambda state: state["current_step"])
                if max_states > 1 and len(ordered) > max_states:
                    ordered = [ordered[0]] + ordered[-(max_states - 1):]
                for state_dict in ordered[:max_states]:
                    state = ExecutionState(**state_dict)
                    if state.remaining_budget["tokens"] < 1 or state.remaining_budget["agent_calls"] < 1:
                        continue
                    seed_everything(task_seed(seed, task.id, digest(state_dict)))
                    pairs = await preference_records(task, config, agents, budgets, weights, state,
                                                     trial_journal=trial_journal, source_identity=identity)
                    for pair in pairs:
                        pair.setdefault("metadata", {}).update(source_sha256=source_hash, source_kind=source_kind)
                        pair_id = digest({"task": task.id, "state": pair["state"],
                                          "chosen": pair["chosen"], "rejected": pair["rejected"]})
                        # Keep the first durable measurement when recovering a
                        # partially completed task; reruns change measured latency.
                        if pair_id not in preferences.ids:
                            preferences.append(pair, pair_id)
            completed.add(task.id)
            manifest = {**identity, "completed_task_ids": sorted(completed),
                        "journals": {"trajectories": len(trajectories.ids), "sft": len(sft.ids), "preferences": len(preferences.ids)},
                        "recovered_partial_bytes": trajectories.recovered_bytes + sft.recovered_bytes + preferences.recovered_bytes,
                        "specialist_fingerprint": fingerprint, "effective_policies": effective_policies,
                        "durable_preference_trials": len(trial_journal.ids),
                        "git_commit": run.record["git_commit"], "task_inventory_sha256": file_digest(directory / "tasks.jsonl")}
            durable_json(manifest_path, manifest)
            log_event("task_generated", task_id=task.id, split=task.split)
        counters["sft_records"], counters["preference_records"] = len(sft.ids), len(preferences.ids)
    counters.update(dataset_directory=str(directory), source_sha256=source_hash, source_kind=source_kind,
                    durable_preference_trials=len(trial_journal.ids),
                    recovery_scope="completed trajectories and counterfactual trials replay without execution; interrupted uncommitted invocations may repeat",
                    scope="Actual executions; synthetic development fixtures do not establish real language-task quality.",
                    preference_protocol="exact-state counterfactual action at configurable initial/late states, fixed continuation",
                    training_split="train only; source task IDs/public texts validated disjoint")
    run.finish(counters)
    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--jobshards", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.shard_index is not None:
        config["shard_index"] = args.shard_index
    if args.jobshards is not None:
        config["jobshards"] = args.jobshards
    asyncio.run(generate(config))


if __name__ == "__main__":
    main()
