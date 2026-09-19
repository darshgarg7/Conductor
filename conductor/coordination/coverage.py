"""Offline supervision audits, separate from all controller and agent inputs.

Audit labels can use private workflow provenance.  None of the returned records
is a state feature, a teacher at inference, or a completion oracle.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from conductor.controller.actions import ActionCatalog
from conductor.controller.cheap import serialize_public_state
from conductor.schema import ExecutionState, RoutingDecision


AUDIT_VERSION = "coordination_coverage_v1"
STATE_KINDS = ("initial", "intermediate", "handoff", "partial", "failure", "stopping")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                   allow_nan=False).encode()).hexdigest()


def preference_pair_sha256(state: dict[str, Any], chosen: RoutingDecision,
                           rejected: RoutingDecision, continuation: str, k: int = 3) -> str:
    """Return the one canonical identity used by generation and audit."""
    catalog = ActionCatalog(k)
    public_sha = hashlib.sha256(serialize_public_state(ExecutionState(**state)).encode()).hexdigest()
    return canonical_sha256({"state": public_sha, "chosen": catalog.key(chosen),
                             "rejected": catalog.key(rejected), "continuation": continuation})


def _field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    return record.get(key, record.get("metadata", {}).get(key, default))


def _identity(record: dict[str, Any]) -> tuple[str, str, str]:
    task = _field(record, "task_id")
    group = _field(record, "source_group")
    family = _field(record, "family", _field(record, "workflow_family"))
    if not all(isinstance(item, str) and item for item in (task, group, family)):
        raise ValueError("coverage records require nonempty offline task_id, source_group and family")
    return task, group, family


def _tags(record: dict[str, Any], state: ExecutionState, decision: RoutingDecision) -> list[str]:
    tags = _field(record, "state_kinds", _field(record, "state_kind", []))
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list) or any(tag not in STATE_KINDS for tag in tags):
        raise ValueError(f"state coverage tags must be drawn from {STATE_KINDS}")
    result = set(tags)
    result.add("initial" if state.current_step == 0 else "intermediate")
    if decision.terminate:
        result.add("stopping")
    return sorted(result)


def _valid_decision(record: dict[str, Any], key: str, catalog: ActionCatalog, k: int) -> RoutingDecision:
    decision = RoutingDecision(**record[key]).validate(k, catalog.agents)
    catalog.index(decision)
    state = ExecutionState(**record["state"])
    calls = state.remaining_budget.get("agent_calls", 0)
    tokens = state.remaining_budget.get("tokens", 0)
    if isinstance(calls, bool) or isinstance(tokens, bool) or not isinstance(calls, (int, float)) or not isinstance(tokens, (int, float)):
        raise ValueError("supervised states require numeric public budgets")
    if not 0 <= calls < float("inf") or not 0 <= tokens < float("inf"):
        raise ValueError("supervised states require finite nonnegative public budgets")
    if len(decision.selected_agents) > int(calls) or (tokens <= 0 and not decision.terminate):
        raise ValueError("supervised action violates public state budget support")
    return decision


def audit_sft_records(records: Iterable[dict[str, Any]], k: int = 3) -> dict[str, Any]:
    values = list(records)
    catalog = ActionCatalog(k)
    counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tasks: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, set[str]] = defaultdict(set)
    steps: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    orders: Counter[str] = Counter()
    handoffs: Counter[str] = Counter()
    canonical_labels: dict[str, set[str]] = defaultdict(set)
    label_occurrences: Counter[str] = Counter()
    for record in values:
        task, group, family = _identity(record)
        decision = _valid_decision(record, "decision", catalog, k)
        state = ExecutionState(**record["state"])
        tasks[family].add(task)
        groups[family].add(group)
        steps[str(state.current_step)] += 1
        tags = _tags(record, state, decision)
        label_cells = [f"state:{tag}" for tag in tags]
        label_cells += [f"count:{len(decision.selected_agents)}"]
        if len(decision.selected_agents) > 1:
            label_cells += [f"mode:{decision.execution_mode}"]
            if decision.execution_mode == "sequential":
                orders[">".join(decision.selected_agents)] += 1
        label_cells += [f"split:{_field(record, 'split', 'train')}"]
        for cell in label_cells:
            counts[cell] += 1
            family_counts[family][cell] += 1
        for agent in decision.selected_agents:
            agents[agent] += 1
        for predecessor, consumer in zip(decision.selected_agents, decision.selected_agents[1:]):
            if decision.execution_mode == "sequential":
                handoffs[f"{predecessor}>{consumer}"] += 1
        state_sha = hashlib.sha256(serialize_public_state(state).encode()).hexdigest()
        action_sha = canonical_sha256(catalog.key(decision))
        canonical_labels[state_sha].add(action_sha)
        label_occurrences[state_sha] += 1
    return {"audit_version": AUDIT_VERSION, "scope": "chosen supervised labels, not candidate action availability",
            "records": len(values), "records_sha256": canonical_sha256(values), "k": k,
            "counts": dict(sorted(counts.items())), "steps": dict(sorted(steps.items())),
            "agent_labels": dict(sorted(agents.items())), "sequential_orders": dict(sorted(orders.items())),
            "within_action_handoffs": dict(sorted(handoffs.items())),
            "families": {family: {"tasks": len(tasks[family]), "source_groups": len(groups[family]),
                                     "counts": dict(sorted(family_counts[family].items()))}
                         for family in sorted(family_counts)},
            "distinct_public_states": len(canonical_labels),
            "duplicate_public_state_records": sum(count - 1 for count in label_occurrences.values()),
            "contradictory_public_state_labels": sum(len(labels) > 1 for labels in canonical_labels.values()),
            "state_tag_source": "offline provenance except initial/intermediate and chosen stop"}


def audit_preferences(records: Iterable[dict[str, Any]], k: int = 3) -> dict[str, Any]:
    values = list(records)
    catalog = ActionCatalog(k)
    task_ids: set[str] = set()
    groups: set[str] = set()
    pairs: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    families: Counter[str] = Counter()
    dependency_stages: Counter[str] = Counter()
    state_kinds: Counter[str] = Counter()
    chosen_counts: Counter[str] = Counter()
    rejected_counts: Counter[str] = Counter()
    chosen_modes: Counter[str] = Counter()
    rejected_modes: Counter[str] = Counter()
    actors: Counter[str] = Counter()
    for record in values:
        task, group, family = _identity(record)
        task_ids.add(task)
        groups.add(group)
        good = _valid_decision(record, "chosen", catalog, k)
        bad = _valid_decision(record, "rejected", catalog, k)
        if catalog.key(good) == catalog.key(bad):
            raise ValueError("preference chosen/rejected collapse to the same canonical complete action")
        state = ExecutionState(**record["state"])
        public_sha = hashlib.sha256(serialize_public_state(state).encode()).hexdigest()
        for key in ("chosen_state", "rejected_state"):
            if key in record:
                branch_sha = hashlib.sha256(serialize_public_state(ExecutionState(**record[key])).encode()).hexdigest()
                if public_sha != branch_sha:
                    raise ValueError("preference branches do not share the same public state")
        continuation = _field(record, "continuation_sha256")
        if not isinstance(continuation, str) or not re_full_hash(continuation):
            raise ValueError("measured preference requires an identified common continuation_sha256")
        for key in ("chosen_continuation_sha256", "rejected_continuation_sha256"):
            if key in record and record[key] != continuation:
                raise ValueError("preference branches use different continuation policies")
        pair_sha = preference_pair_sha256(record["state"], good, bad, continuation, k)
        stored_pair = record.get("pair_sha256")
        if stored_pair is not None and stored_pair != pair_sha:
            raise ValueError("stored preference pair identity does not match canonical public-state identity")
        pairs[pair_sha] += 1
        sources[str(_field(record, "preference_source", "unspecified"))] += 1
        families[family] += 1
        dependency_stages[str(_field(record, "dependency_stages", "unknown"))] += 1
        for tag in _field(record, "state_kinds", []):
            state_kinds[str(tag)] += 1
        chosen_counts[str(len(good.selected_agents))] += 1
        rejected_counts[str(len(bad.selected_agents))] += 1
        if len(good.selected_agents) > 1:
            chosen_modes[good.execution_mode] += 1
        if len(bad.selected_agents) > 1:
            rejected_modes[bad.execution_mode] += 1
        actors[str(_field(record, "actor_checkpoint_sha256", "missing"))] += 1
    return {"audit_version": AUDIT_VERSION, "records": len(values), "records_sha256": canonical_sha256(values),
            "tasks": len(task_ids), "source_groups": len(groups), "unique_pairs": len(pairs),
            "duplicate_pairs": sum(count - 1 for count in pairs.values()), "sources": dict(sorted(sources.items())),
            "families": dict(sorted(families.items())),
            "dependency_stages": dict(sorted(dependency_stages.items())),
            "state_kinds": dict(sorted(state_kinds.items())),
            "chosen_agent_counts": dict(sorted(chosen_counts.items())),
            "rejected_agent_counts": dict(sorted(rejected_counts.items())),
            "chosen_multiagent_modes": dict(sorted(chosen_modes.items())),
            "rejected_multiagent_modes": dict(sorted(rejected_modes.items())),
            "actor_checkpoint_sha256_counts": dict(sorted(actors.items())),
            "shared_public_state_and_continuation_validated": True, "canonical_distinct_actions_validated": True}


def re_full_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def audit_partitions(train: Iterable[dict[str, Any]], development: Iterable[dict[str, Any]], *,
                     k: int = 3, required_families: Iterable[str] = (), minimum_dev_groups: int = 1,
                     required_cells: Iterable[str] = (), require_distinct_labels: bool = True) -> dict[str, Any]:
    """Reject group/task leakage and empty requested chosen-label cells.

    Split grouping is fixed upstream.  This audit never repairs a partition or
    moves a task after seeing a score.  Completeness gates refer to chosen
    labels across each partition unless a caller intentionally requests none.
    """
    if minimum_dev_groups < 1:
        raise ValueError("minimum_dev_groups must be positive")
    fitting, selection = list(train), list(development)
    if not fitting or not selection:
        raise ValueError("training and development supervision must both be nonempty")
    identities = [[_identity(record) for record in records] for records in (fitting, selection)]
    train_tasks = {task for task, _, _ in identities[0]}
    dev_tasks = {task for task, _, _ in identities[1]}
    train_groups = {group for _, group, _ in identities[0]}
    dev_groups = {group for _, group, _ in identities[1]}
    if train_tasks & dev_tasks:
        raise ValueError("training/development task overlap")
    if train_groups & dev_groups:
        raise ValueError("training/development underlying source-group overlap")
    reports = {"train": audit_sft_records(fitting, k), "development": audit_sft_records(selection, k)}
    required = set(required_families) or {family for _, _, family in identities[0]}
    for split, report in reports.items():
        absent = required - report["families"].keys()
        if absent:
            raise ValueError(f"{split} missing required families: {sorted(absent)}")
        missing = [cell for cell in required_cells if not report["counts"].get(cell, 0)]
        if missing:
            raise ValueError(f"{split} missing chosen-label coverage cells: {sorted(missing)}")
        if require_distinct_labels and report["contradictory_public_state_labels"]:
            raise ValueError(f"{split} contains contradictory canonical labels for identical public states")
    for family in required:
        if reports["development"]["families"][family]["source_groups"] < minimum_dev_groups:
            raise ValueError(f"development family {family} has too few distinct source groups")
    return {"audit_version": AUDIT_VERSION, "partitions": reports,
            "task_disjoint": True, "source_group_disjoint": True,
            "required_families": sorted(required), "required_cells": sorted(required_cells),
            "minimum_development_source_groups_per_family": minimum_dev_groups,
            "coverage_gate_passed": True}


def write_immutable_audit(path: str | Path, report: dict[str, Any]) -> str:
    """Publish once and return the byte digest; never overwrite an audit seal."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with destination.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()
