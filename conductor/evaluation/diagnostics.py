"""Task-weighted descriptive diagnostics for external coordination actions.

Repeated runs of a task are averaged before aggregation. Unique IDs are the
analysis units; they do not establish independence between template siblings.
Concentrated routing can be appropriate, and is never an automatic failure gate.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from conductor.schema import AGENT_NAMES, RoutingDecision

NO_ROUTE = "no_observed_routing_decision"


def _record(value: Any) -> dict[str, Any]:
    record = asdict(value) if is_dataclass(value) else value
    if not isinstance(record, dict):
        raise ValueError("trajectory and step records must be mappings")
    return record


def _entropy(mass: dict[Any, float]) -> float:
    total = sum(mass.values())
    return -sum((value / total) * math.log2(value / total) for value in mass.values() if value > 0) if total else 0.0


def _mutual_information(joint: dict[tuple[str, str], float]) -> float:
    total = sum(joint.values())
    left: Counter = Counter()
    right: Counter = Counter()
    for (category, action), value in joint.items():
        left[category] += value
        right[action] += value
    information = sum((value / total) * math.log2(value * total / (left[category] * right[action]))
                      for (category, action), value in joint.items() if value > 0) if total else 0.0
    return max(0.0, information)  # protect the zero case from floating-point cancellation


def _permutation_null(task_actions: list[tuple[str, Counter]], observed: float | None,
                      samples: int, seed: int) -> dict[str, Any]:
    scope = ("Shuffles task-type labels across unique task IDs, keeping each task's repeated-run action distribution together. "
             "Exchangeability is an assumption; shared-template task dependence is not corrected. "
             "P-values are unadjusted for multiple policy/metric comparisons. "
             "Association does not establish learning, specialization or causation.")
    if observed is None or samples == 0 or len(task_actions) < 2:
        return {"status": "unavailable", "reason": "Missing routing coverage, fewer than two tasks, or permutations disabled",
                "scope": scope}
    generator = random.Random(seed)
    labels = [category for category, _ in task_actions]
    null_values = []
    for _ in range(samples):
        generator.shuffle(labels)
        joint: Counter = Counter()
        for label, (_, actions) in zip(labels, task_actions):
            for action, mass in actions.items():
                joint[(label, action)] += mass
        null_values.append(_mutual_information(joint))
    mean = sum(null_values) / samples
    return {"status": "measured", "independent_task_count": len(task_actions), "permutation_samples": samples,
            "seed": seed, "observed_mutual_information_bits": observed, "mean_permuted_mutual_information_bits": mean,
            "excess_over_permuted_mean_bits": observed - mean,
            "p_value_ge_observed": (1 + sum(value >= observed - 1e-12 for value in null_values)) / (samples + 1),
            "scope": scope}


def _action(value: Any) -> tuple[str, dict[str, Any]]:
    decision = RoutingDecision(**_record(value)).validate(len(AGENT_NAMES))
    agents = list(decision.selected_agents)
    mode = "parallel" if len(agents) <= 1 else decision.execution_mode
    if mode == "parallel":
        agents.sort()
    action = {"selected_agents": agents, "execution_mode": mode, "terminate": decision.terminate}
    return json.dumps(action, sort_keys=True, separators=(",", ":")), action


def _normalize(item: Any) -> dict[str, Any]:
    raw = _record(item)
    task = _record(raw["task"])
    for field in ("id", "user_task", "task_type"):
        if not isinstance(task.get(field), str) or not task[field]:
            raise ValueError(f"task {field} must be a nonempty string")
    if not isinstance(raw.get("policy"), str) or not raw["policy"]:
        raise ValueError("policy must be a nonempty string")
    if type(raw.get("task_success")) is not bool:
        raise ValueError("task_success must be a recorded boolean")
    expected = task.get("expected_answer")
    final = raw.get("final_answer")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("expected_answer must be a string when present")
    exact = final.strip() == expected.strip() if isinstance(final, str) and isinstance(expected, str) else None
    steps = raw.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps must be an ordered list")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("trajectory metadata must be a mapping")
    requested = metadata.get("requested_routing_decisions")
    if requested is not None and (not isinstance(requested, list) or len(requested) != len(steps)):
        raise ValueError("requested routing decisions must align with the recorded steps")
    events, activations = [], Counter()
    last_step = -1
    for index, value in enumerate(steps):
        step = _record(value)
        state = step.get("state", {})
        if not isinstance(state, dict):
            raise ValueError("execution state must be a mapping")
        number = state.get("current_step", index)
        if type(number) is not int or number < 0 or number <= last_step:
            raise ValueError("coordination steps must be strictly increasing nonnegative integers")
        last_step = number
        reused = step.get("routing_reused", False)
        if type(reused) is not bool:
            raise ValueError("routing_reused must be a boolean")
        actual_key, actual = _action(step["decision"])
        key, action = _action(requested[index]) if requested is not None else (actual_key, actual)
        outputs = step.get("agent_outputs", [])
        if not isinstance(outputs, list):
            raise ValueError("agent outputs must be a list")
        for output in outputs:
            agent = _record(output).get("agent")
            if agent not in AGENT_NAMES:
                raise ValueError("agent output uses an unknown specialist")
            activations[agent] += 1
        events.append({"step": number, "key": key, "action": action, "actual": actual, "reused": reused})
    routes = [event for event in events if not event["reused"]]
    return {"task": task, "policy": raw["policy"], "recorded_success": raw["task_success"], "exact_success": exact,
            "events": events, "routes": routes, "activations": activations,
            "first_key": routes[0]["key"] if routes else NO_ROUTE,
            "requested_source": requested is not None,
            "conditions": {key: metadata[key] for key in ("checkpoint_sha256", "specialist_fingerprint", "evaluation_budgets")
                           if metadata.get(key) is not None}}


def _distribution(mass: Counter, actions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(mass.values())
    return [{"action": actions.get(key), "missing_routing_decision": key == NO_ROUTE,
             "task_weighted_mass": value, "probability": value / total if total else None}
            for key, value in sorted(mass.items(), key=lambda pair: (-pair[1], pair[0]))]


def routing_diagnostics(trajectories: Iterable[Any], *, permutation_samples: int = 199, seed: int = 42) -> dict[str, Any]:
    """Analyze raw trajectories; never interpret low diversity as a quality verdict.

    Each policy's unique task gets one unit. Repeated executions receive 1/R
    of that task's weight. Within a run, action-frequency entropy further gives
    each fresh routing event 1/S of the run's weight. Raw counts remain descriptive.
    First-action MI is unavailable when any run lacks a fresh routing decision.
    """
    if type(permutation_samples) is not int or permutation_samples < 0 or type(seed) is not int or seed < 0:
        raise ValueError("permutation count and seed must be nonnegative integers")
    normalized = [_normalize(item) for item in trajectories]
    identities: dict[str, tuple[Any, ...]] = {}
    prompts: dict[str, str] = {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for run in normalized:
        task = run["task"]
        identity = (task["user_task"], task["task_type"], task.get("split"), task.get("expected_answer"))
        if task["id"] in identities and identities[task["id"]] != identity:
            raise ValueError("repeated task ID has inconsistent public identity or grading label")
        if task["user_task"] in prompts and prompts[task["user_task"]] != task["id"]:
            raise ValueError("identical public task text cannot create extra independent task units")
        identities[task["id"]] = identity
        prompts[task["user_task"]] = task["id"]
        grouped[run["policy"]][task["id"]].append(run)
    policies = {}
    for policy, tasks in sorted(grouped.items()):
        actions: dict[str, dict[str, Any]] = {}
        first_mass: Counter = Counter()
        route_mass: Counter = Counter()
        selection_mass: Counter = Counter()
        activation_mass: Counter = Counter()
        called_mass: Counter = Counter()
        joint: Counter = Counter()
        category_mass: Counter = Counter()
        step_mass: dict[int, Counter] = defaultdict(Counter)
        categories: dict[str, dict[str, Any]] = {}
        first_step_mass: Counter = Counter()
        task_actions: list[tuple[str, Counter]] = []
        all_conditions: dict[str, set[str]] = defaultdict(set)
        raw_routes = raw_reused = disagreements = 0
        recorded_success = exact_success = exact_known = 0.0
        for task_id, runs in sorted(tasks.items()):
            category = runs[0]["task"]["task_type"]
            category_mass[category] += 1
            category_record = categories.setdefault(category, {"independent_task_count": 0, "trajectory_count": 0,
                "recorded_success_mass": 0.0, "exact_success_mass": 0.0, "exact_grading_mass": 0.0,
                "actual_activation_mass": 0.0})
            category_record["independent_task_count"] += 1
            category_record["trajectory_count"] += len(runs)
            weight = 1 / len(runs)
            task_first_actions: Counter = Counter()
            for run in runs:
                first_mass[run["first_key"]] += weight
                task_first_actions[run["first_key"]] += weight
                joint[(category, run["first_key"])] += weight
                success = weight * run["recorded_success"]
                recorded_success += success
                category_record["recorded_success_mass"] += success
                if run["exact_success"] is not None:
                    exact_known += weight
                    exact_success += weight * run["exact_success"]
                    category_record["exact_grading_mass"] += weight
                    category_record["exact_success_mass"] += weight * run["exact_success"]
                    disagreements += run["exact_success"] != run["recorded_success"]
                category_record["actual_activation_mass"] += weight * sum(run["activations"].values())
                for agent, count in run["activations"].items():
                    activation_mass[agent] += weight * count
                    called_mass[agent] += weight
                for key, condition in run["conditions"].items():
                    all_conditions[key].add(json.dumps(condition, sort_keys=True, allow_nan=False))
                if run["routes"]:
                    first_step_mass[run["routes"][0]["step"]] += weight
                    event_weight = weight / len(run["routes"])
                    for event in run["routes"]:
                        actions[event["key"]] = event["action"]
                        route_mass[event["key"]] += event_weight
                        for agent in event["action"]["selected_agents"]:
                            selection_mass[agent] += event_weight
                else:
                    route_mass[NO_ROUTE] += weight
                for event in run["events"]:
                    mass = step_mass[event["step"]]
                    mass["visited"] += weight
                    mass["execution_stops"] += weight * event["actual"]["terminate"]
                    if not event["reused"]:
                        mass["fresh_route"] += weight
                        mass["requested_stops"] += weight * event["action"]["terminate"]
                raw_routes += len(run["routes"])
                raw_reused += len(run["events"]) - len(run["routes"])
            task_actions.append((category, task_first_actions))
        count = len(tasks)
        missing = first_mass.get(NO_ROUTE, 0.0)
        first_keys = set(first_mass) - {NO_ROUTE}
        constant = len(first_keys) == 1 and missing == 0
        information = _mutual_information(joint) if missing == 0 else None
        first_entropy = _entropy(first_mass)
        type_entropy = _entropy(category_mass)
        selected_total = sum(selection_mass.values())
        warnings = []
        if any(mass < 2 for mass in category_mass.values()):
            warnings.append({"code": "sparse_task_type_cells", "severity": "scope",
                "message": "Some task types have fewer than two task units. Plug-in mutual information can be maximal "
                           "for random unique actions; inspect the permutation null, not MI alone."})
        if constant:
            warnings.append({"code": "constant_first_action", "severity": "descriptive",
                "message": "Every observed first action is identical. This can be a justified shared route; "
                           "inspect task success and required capabilities before calling it harmful collapse."})
        if len(category_mass) < 2:
            warnings.append({"code": "single_task_type", "severity": "scope",
                             "message": "One task type cannot establish task-type-dependent routing."})
        if missing:
            warnings.append({"code": "missing_first_routes", "severity": "coverage",
                             "message": "First-action MI is unavailable because routing coverage is incomplete."})
        if disagreements:
            warnings.append({"code": "success_record_disagreement", "severity": "data_integrity",
                             "message": "Recorded success differs from independent exact answer comparison."})
        if not math.isclose(exact_known, count):
            warnings.append({"code": "exact_grading_incomplete", "severity": "coverage",
                             "message": "Exact success is unavailable unless all runs retain final answers and private grading labels."})
        mixed = any(len(values) > 1 for values in all_conditions.values())
        if mixed:
            warnings.append({"code": "mixed_conditions", "severity": "scope",
                             "message": "This policy pools different checkpoints, specialists or budgets; separate them for controlled comparisons."})
        if first_step_mass and set(first_step_mass) != {0}:
            warnings.append({"code": "noninitial_first_routes", "severity": "scope",
                             "message": "First observed routing includes resumed/noninitial states; it is not a uniform initial-state probe."})
        for category, row in categories.items():
            denominator = row["independent_task_count"]
            row["recorded_success_rate"] = row["recorded_success_mass"] / denominator
            row["exact_grading_coverage"] = row["exact_grading_mass"] / denominator
            row["exact_success_rate"] = row["exact_success_mass"] / denominator if math.isclose(row["exact_grading_mass"], denominator) else None
            row["mean_actual_agent_activations"] = row["actual_activation_mass"] / denominator
        policies[policy] = {"independent_task_count": count, "task_ids": sorted(tasks),
            "trajectory_count": sum(len(runs) for runs in tasks.values()),
            "runs_per_task": {task_id: len(runs) for task_id, runs in sorted(tasks.items())},
            "splits": sorted({str(run["task"].get("split", "unspecified")) for runs in tasks.values() for run in runs}),
            "recorded_success_rate": recorded_success / count,
            "exact_grading_coverage": exact_known / count,
            "exact_success_rate": exact_success / count if math.isclose(exact_known, count) else None,
            "reported_success_disagreements": disagreements,
            "raw_fresh_routing_decisions": raw_routes, "raw_reused_decisions": raw_reused,
            "unique_routing_actions": len(set(route_mass) - {NO_ROUTE}),
            "unique_first_actions": len(first_keys), "constant_first_action": constant,
            "first_action_distribution": _distribution(first_mass, actions),
            "action_distribution": _distribution(route_mass, actions),
            "first_action_entropy_bits": first_entropy, "action_entropy_bits": _entropy(route_mass),
            "task_type_entropy_bits": type_entropy, "task_type_first_action_mutual_information_bits": information,
            "normalized_task_type_first_action_mutual_information": information / min(first_entropy, type_entropy)
                if information is not None and min(first_entropy, type_entropy) > 0 else None,
            "first_action_association_permutation_null": _permutation_null(task_actions, information, permutation_samples, seed),
            "missing_first_action_task_fraction": missing / count,
            "first_route_step_distribution": [{"step": step, "task_weighted_mass": mass}
                                               for step, mass in sorted(first_step_mass.items())],
            "agent_selection_distribution": {agent: {"selection_share": selection_mass[agent] / selected_total if selected_total else None,
                "task_normalized_route_selection_probability": selection_mass[agent] / count,
                "mean_actual_activations_per_task": activation_mass[agent] / count,
                "task_called_probability": called_mass[agent] / count} for agent in AGENT_NAMES},
            "agent_selection_entropy_bits": _entropy(selection_mass),
            "mean_actual_agent_activations": sum(activation_mass.values()) / count,
            "termination_by_step": [{"step": step, "visiting_task_mass": mass["visited"],
                "visitation_rate": mass["visited"] / count, "fresh_routing_task_mass": mass["fresh_route"],
                "requested_termination_frequency_given_fresh_route": mass["requested_stops"] / mass["fresh_route"] if mass["fresh_route"] else None,
                "execution_termination_frequency_given_visit": mass["execution_stops"] / mass["visited"],
                "requested_termination_task_probability": mass["requested_stops"] / count}
                for step, mass in sorted(step_mass.items())],
            "categories": dict(sorted(categories.items())),
            "condition_inventory": {key: [json.loads(value) for value in sorted(values)] for key, values in sorted(all_conditions.items())},
            "mixed_conditions": mixed,
            "requested_action_run_coverage": sum(run["requested_source"] for runs in tasks.values() for run in runs)
                / sum(len(runs) for runs in tasks.values()),
            "warnings": warnings}
    return {"schema_version": 1, "policies": policies,
        "analysis_unit": "One equally weighted unique task ID per policy; executions averaged within task. "
                         "Unique IDs do not establish sampling independence between shared-template tasks.",
        "action_frequency_unit": "Each task splits its unit across runs, then across fresh routing events; "
                                 "long trajectories do not gain extra statistical weight. Raw counts are descriptive.",
        "action_source": "Aligned requested decisions when available; otherwise recorded execution decisions. "
                         "Reused decisions excluded from fresh routing distributions; actual outputs determine activations.",
        "association_unit": "Task-type versus first observed fresh external action, using equal task weights. "
                            "Unavailable for incomplete routing coverage; bits are descriptive plug-in estimates, "
                            "with a task-unit-preserving permutation null to expose high-cardinality/small-corpus bias.",
        "interpretation": "Routing concentration is descriptive, not an automatic model failure or causal specialization claim. "
                          "These are external specialist actions, not internal MoE expert activations. "
                          "Exact grading reads private labels only in offline analysis; no held-out filtering is performed here."}
