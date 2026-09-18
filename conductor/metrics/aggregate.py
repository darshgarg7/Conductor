"""Metrics computed from observed trajectories; missing measurements stay missing."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

import numpy as np

from conductor.schema import AGENT_NAMES


def record(value: Any) -> dict[str, Any]:
    return asdict(value) if is_dataclass(value) else value


def percentile(values: Iterable[float], q: float) -> float | None:
    data = list(values)
    return float(np.percentile(data, q)) if data else None


def trajectory_metrics(trajectory: Any) -> dict[str, Any]:
    item = record(trajectory)
    steps = [record(step) for step in item.get("steps", [])]
    outputs = [output for step in steps for output in step.get("agent_outputs", [])]
    controller_tokens = sum(step.get("controller_tokens", 0) for step in steps)
    downstream_tokens = sum(output.get("tokens", 0) for output in outputs)
    controller_seconds = sum(step.get("controller_latency_seconds", 0.0) for step in steps)
    controller_cost = sum(step.get("controller_cost_usd", 0.0) for step in steps)
    downstream_cost = sum(output.get("cost_usd", 0.0) for output in outputs)
    active_rounds = [step for step in steps if step.get("agent_outputs")]
    task = item["task"]
    metadata = task.get("metadata", {})
    wall = item.get("wall_clock_latency", 0.0)
    activations = sum(len(step.get("agent_outputs", [])) for step in active_rounds)
    possible = len(AGENT_NAMES) * len(active_rounds)
    graph = item.get("communication_graph", [])
    physical_messages = [edge for edge in graph if edge.get("source") == "controller" or edge.get("target") == "controller"]
    trajectory_metadata = item.get("metadata", {})
    accounts = [output.get("metadata", {}).get("token_accounting", "unknown") for output in outputs]
    controller_account = trajectory_metadata.get("controller_token_accounting", "unknown")
    billable_accounts = {"model_tokenizer", "provider_usage", "hf_tokenizer"}
    actual_controller = controller_tokens == 0 or "HF tokenizer" in controller_account
    usage_known = (trajectory_metadata.get("token_usage_known", True)
                   and trajectory_metadata.get("inference_cost_known", True)
                   and not any(output.get("metadata", {}).get("token_usage_unknown", False)
                               or output.get("metadata", {}).get("cost_usage_unknown", False) for output in outputs))
    cost_valid = bool(usage_known and all(account in billable_accounts for account in accounts) and actual_controller)
    unique_edges = {(str(edge.get("source", edge.get("from", ""))),
                     str(edge.get("target", edge.get("to", "")))) for edge in graph}
    # Forwarded state may include self messages; the controller is also a graph node.
    possible_edges = (len(AGENT_NAMES) + 1) ** 2
    density = len(unique_edges) / possible_edges
    return {
        "task_id": task["id"], "policy": item["policy"],
        "policy_label": trajectory_metadata.get("policy_label", item["policy"]),
        "data_sha256": trajectory_metadata.get("data_sha256"),
        "specialist_sha256": trajectory_metadata.get("specialist_sha256"),
        "experiment_sha256": trajectory_metadata.get("experiment_sha256"),
        "checkpoint_sha256": trajectory_metadata.get("checkpoint_sha256"),
        "config_sha256": trajectory_metadata.get("config_sha256"),
        "cost_comparison_valid": cost_valid,
        "token_billing_status": "actual_backend_tokens" if cost_valid else "unknown_or_proxy_tokens",
        "category": task.get("task_type", "unknown"), "split": task.get("split", "unknown"),
        "template_family": metadata.get("template_family", "unknown"),
        "generalization": metadata.get("generalization", metadata.get("distribution",
                                        "unseen_composition" if metadata.get("ood") else "seen_family")),
        "task_success": bool(item.get("task_success", False)), "grader_score": item.get("grader_score", 0.0),
        "controller_tokens": controller_tokens, "downstream_tokens": downstream_tokens,
        "total_tokens": controller_tokens + downstream_tokens, "agent_calls": len(outputs),
        "rounds": len(active_rounds), "routing_calls": sum(not step.get("routing_reused", False) for step in steps),
        "wall_clock_seconds": wall, "controller_seconds": controller_seconds,
        "controller_overhead_fraction": controller_seconds / wall if wall > 0 else None,
        "controller_token_fraction": controller_tokens / (controller_tokens + downstream_tokens) if controller_tokens + downstream_tokens else None,
        "controller_cost_usd": controller_cost, "downstream_cost_usd": downstream_cost,
        "cost_usd": controller_cost + downstream_cost,
        "controller_cost_fraction": controller_cost / (controller_cost + downstream_cost) if controller_cost + downstream_cost > 0 else None,
        "communication_edges": len(graph), "unique_communication_edges": len(unique_edges),
        "physical_controller_agent_messages": trajectory_metadata.get("controller_agent_messages", len(physical_messages)),
        "logical_inter_agent_edges": trajectory_metadata.get("logical_inter_agent_edges", len(graph) - len(physical_messages)),
        "communication_density": density, "communication_sparsity": 1 - density,
        "communication_message_bytes": trajectory_metadata.get("serialized_state_and_output_bytes",
                                                                 sum(edge.get("message_bytes", 0) for edge in physical_messages)),
        "communication_estimated_tokens": sum(edge.get("estimated_message_tokens", 0) for edge in physical_messages),
        "activation_sparsity": 1.0 - activations / possible if possible else None,
    }


def aggregate_metrics(rows: list[dict[str, Any]], group_by: tuple[str, ...] = ("policy",)) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_by)].append(row)
    summaries = []
    fields = ("grader_score", "controller_tokens", "downstream_tokens", "total_tokens", "agent_calls", "rounds",
              "routing_calls", "wall_clock_seconds", "controller_seconds", "controller_overhead_fraction",
              "controller_token_fraction", "controller_cost_fraction",
              "controller_cost_usd", "downstream_cost_usd", "cost_usd", "communication_edges",
              "unique_communication_edges", "communication_density", "communication_sparsity",
              "physical_controller_agent_messages", "logical_inter_agent_edges",
              "communication_message_bytes", "communication_estimated_tokens", "activation_sparsity")
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        summary = dict(zip(group_by, key))
        summary.update(task_count=len(group), success_rate=sum(row["task_success"] for row in group) / len(group))
        for field in fields:
            values = [row[field] for row in group if row.get(field) is not None]
            summary[f"mean_{field}"] = float(np.mean(values)) if values else None
        for field in ("wall_clock_seconds", "controller_seconds", "cost_usd"):
            summary[f"p50_{field}"] = percentile((row[field] for row in group), 50)
            summary[f"p95_{field}"] = percentile((row[field] for row in group), 95)
        summaries.append(summary)
    return summaries


def paired_differences(rows: list[dict[str, Any]], baseline: str, candidate: str,
                       bootstrap_samples: int = 2000, seed: int = 42) -> dict[str, Any]:
    """Only compare the intersection of identical task IDs, and disclose coverage."""
    def unique(policy: str) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["policy"] == policy:
                if row["task_id"] in selected:
                    raise ValueError(f"Duplicate paired task ID for {policy}: {row['task_id']}; aggregate repeated trials explicitly")
                selected[row["task_id"]] = row
        return selected
    left, right = unique(baseline), unique(candidate)
    common = sorted(left.keys() & right.keys())
    result: dict[str, Any] = {"baseline": baseline, "candidate": candidate, "paired_tasks": len(common),
                              "baseline_only_tasks": len(left.keys() - right.keys()),
                              "candidate_only_tasks": len(right.keys() - left.keys()),
                              "status": "measured" if common else "unanswered", "categories": {}}
    if not common:
        result["reason"] = "No paired task trajectories are available."
        return result
    fields = ("task_success", "grader_score", "total_tokens", "agent_calls", "cost_usd", "wall_clock_seconds")
    from conductor.metrics.statistics import conservative_claim_gate, paired_bootstrap
    interval_names = {"task_success": "success", "total_tokens": "token", "agent_calls": "agent_calls",
                      "cost_usd": "cost", "wall_clock_seconds": "latency"}
    for field in fields:
        result[f"mean_delta_{field}"] = float(np.mean([right[i][field] - left[i][field] for i in common]))
        if field in interval_names:
            estimate = paired_bootstrap([float(left[i][field]) for i in common], [float(right[i][field]) for i in common],
                                        seed=seed, samples=bootstrap_samples)
            name = interval_names[field]
            result[f"{name}_delta_ci95"] = estimate["delta_ci95"]
            result[f"{name}_ratio_ci95"] = estimate["ratio_ci95"]
    fingerprints = {}
    for field in ("data_sha256", "specialist_sha256", "experiment_sha256", "checkpoint_sha256", "config_sha256"):
        fingerprints[field] = {"baseline": sorted({str(left[i].get(field)) for i in common}),
                               "candidate": sorted({str(right[i].get(field)) for i in common})}
    result["fingerprints"] = fingerprints
    result["comparable_fingerprints"] = all(
        len({left[i].get(field) for i in common} | {right[i].get(field) for i in common}) == 1
        and all(left[i].get(field) and right[i].get(field) for i in common)
        for field in ("data_sha256", "specialist_sha256", "experiment_sha256"))
    result["unknown_cost_exclusions"] = sum(not left[i].get("cost_comparison_valid", False) or
                                            not right[i].get("cost_comparison_valid", False) for i in common)
    result["bootstrap"] = {"unit": "unique_task_id", "samples": bootstrap_samples, "seed": seed}
    result["claim_gate"] = conservative_claim_gate(result)
    for category in sorted({left[i]["category"] for i in common}):
        ids = [i for i in common if left[i]["category"] == category]
        result["categories"][category] = {"paired_tasks": len(ids), **{
            f"mean_delta_{field}": float(np.mean([right[i][field] - left[i][field] for i in ids])) for field in fields}}
    return result
