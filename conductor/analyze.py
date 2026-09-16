"""Paired quality/cost analysis and exportable plots from measured artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from conductor.evaluation.io import write_csv
from conductor.metrics.aggregate import aggregate_metrics, paired_differences, trajectory_metrics
from conductor.utils.runs import write_json


def summarize_experts(stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive load/specialization measures, without semantic or causal claims."""
    result = []
    for policy, item in stats.items():
        for layer, observed in item.get("layers", {}).items():
            counts = observed.get("activation_counts", [])
            if not counts or not sum(counts):
                continue
            overall = [count / sum(counts) for count in counts]
            divergences = {}
            for task_type, layers in item.get("task_type_activation_counts", {}).items():
                task_counts = layers.get(layer, [])
                if not task_counts or not sum(task_counts):
                    continue
                frequencies = [count / sum(task_counts) for count in task_counts]
                midpoint = [(left + right) / 2 for left, right in zip(overall, frequencies)]
                def kl(distribution):
                    return sum(value * math.log(value / center, 2) for value, center in zip(distribution, midpoint) if value > 0)
                divergences[task_type] = (kl(overall) + kl(frequencies)) / 2
            result.append({"policy": policy, "layer": layer, "num_experts": len(counts),
                           "utilized_experts": sum(count > 0 for count in counts),
                           "utilized_fraction": sum(count > 0 for count in counts) / len(counts),
                           "activation_frequency": overall,
                           "normalized_routing_entropy": observed.get("normalized_routing_entropy"),
                           "load_coefficient_of_variation": observed.get("load_coefficient_of_variation"),
                           "task_distribution_js_divergence": divergences,
                           "observations": observed.get("observations"),
                           "interpretation": "Task/load correlation is descriptive; it does not establish semantic expert specialization."})
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "No measured rows are available."
    def format_value(value):
        if value is None:
            return "unavailable"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    return "\n".join(["| " + " | ".join(label for _, label in columns) + " |",
                      "| " + " | ".join("---" for _ in columns) + " |",
                      *["| " + " | ".join(format_value(row.get(key)) for key, _ in columns) + " |" for row in rows]])


def plot_results(rows: list[dict[str, Any]], output: Path, benchmarks: dict[str, Any] | None = None,
                 probes: list[dict[str, Any]] | None = None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    summaries = aggregate_metrics(rows)
    if summaries:
        names = [row["policy"] for row in summaries]
        labels = {row["policy"]: row.get("policy_label", row["policy"]) for row in rows}
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        for axis, field, label in zip(axes, ("success_rate", "mean_total_tokens", "mean_agent_calls"),
                                     ("Task success rate", "Mean observed tokens", "Mean downstream calls")):
            axis.bar([labels[name] for name in names], [row[field] for row in summaries], color="#3e7cb1")
            axis.set_ylabel(label)
            axis.tick_params(axis="x", rotation=55)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylim(0, 1.05)
        fig.suptitle("Heldout development tasks: identical task set and aggregate budgets")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = output / f"quality_cost.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            artifacts.append(str(path))
        plt.close(fig)
        categories = aggregate_metrics(rows, ("policy", "category"))
        fig, axis = plt.subplots(figsize=(10, 5))
        for name in names:
            selected = [row for row in categories if row["policy"] == name]
            axis.plot([row["category"] for row in selected], [row["success_rate"] for row in selected], marker="o", label=labels[name])
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Task success rate")
        axis.set_title("Heldout task categories (composition categories are separately labeled)")
        axis.tick_params(axis="x", rotation=25)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = output / "categories.png"
        fig.savefig(path, dpi=180)
        artifacts.append(str(path))
        plt.close(fig)
    if probes:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        policy_names = sorted({row["policy"] for row in probes})
        for name in policy_names:
            values = [row for row in probes if row["policy"] == name]
            label = next((row.get("policy_label", name) for row in rows if row["policy"] == name), name)
            axes[0].plot([row["layer"] for row in values], [row["utilized_fraction"] for row in values], marker="o", label=label)
            axes[1].plot([row["layer"] for row in values], [row["load_coefficient_of_variation"] for row in values], marker="o", label=label)
        axes[0].set_ylabel("Fraction of experts activated")
        axes[0].set_ylim(-0.05, 1.05)
        axes[1].set_ylabel("Load coefficient of variation")
        for axis in axes:
            axis.set_xlabel("Controller layer")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.25)
        fig.suptitle("Internal experts on identical heldout initial states")
        fig.tight_layout()
        path = output / "expert_utilization.png"
        fig.savefig(path, dpi=180)
        artifacts.append(str(path))
        plt.close(fig)
        first_layer = sorted({row["layer"] for row in probes})[0]
        selected = [row for row in probes if row["layer"] == first_layer]
        if selected and len({len(row["activation_frequency"]) for row in selected}) == 1:
            fig, axis = plt.subplots(figsize=(10, 3 + 0.5 * len(selected)))
            im = axis.imshow([row["activation_frequency"] for row in selected], cmap="Blues", aspect="auto", vmin=0)
            axis.set_yticks(range(len(selected)), [row["policy"] for row in selected])
            axis.set_xlabel("Internal expert index")
            axis.set_title(f"Identical-state expert activation frequency, layer {first_layer}")
            fig.colorbar(im, ax=axis, label="Fraction of observed activations")
            fig.tight_layout()
            path = output / "expert_frequency.png"
            fig.savefig(path, dpi=180)
            artifacts.append(str(path))
            plt.close(fig)
    benchmark_rows = (benchmarks or {}).get("benchmarks", [])
    if benchmark_rows:
        fig, axis = plt.subplots(figsize=(10, 5))
        # A single reference slice prevents unequal workload dimensions from masquerading as paired optimization gains.
        reference = benchmark_rows[0]
        fields = ("context_size_requested_words", "concurrency", "k", "routing_interval")
        matching = [row for row in benchmark_rows if all(row.get(field) == reference.get(field) for field in fields)]
        for strategy in sorted({row["strategy"] for row in matching}):
            selected = sorted((row for row in matching if row["strategy"] == strategy), key=lambda row: row["batch_size"])
            axis.plot([row["batch_size"] for row in selected], [row["latency_p95_seconds"] * 1000 for row in selected],
                      marker="o", label=strategy)
        axis.set_xlabel("Maximum/offline batch size")
        axis.set_ylabel("Queue-inclusive request p95 (ms)")
        axis.set_title("Measured controller timing at one fixed workload slice")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = output / "controller_latency.png"
        fig.savefig(path, dpi=180)
        artifacts.append(str(path))
        plt.close(fig)
    return artifacts


def analyze(evaluation: str | Path, output: str | Path | None = None,
            benchmark_path: str | Path | None = None) -> dict[str, Any]:
    source = Path(evaluation)
    target = Path(output) if output else source / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    trajectory_path = source / "trajectories.jsonl"
    trajectories = [json.loads(line) for line in trajectory_path.read_text().splitlines() if line.strip()] if trajectory_path.exists() else []
    rows = [trajectory_metrics(item) for item in trajectories]
    policies = sorted({row["policy"] for row in rows})
    paired = [paired_differences(rows, baseline, candidate)
              for baseline in ("all_agent", "rule_based", "random_top_k", "static_supervisor", "base_moe", "conductor_sft")
              for candidate in ("conductor_sft", "conductor_preference") if baseline != candidate]
    categories = aggregate_metrics(rows, ("policy", "category", "split", "generalization"))
    stats_path = source / "expert_stats.json"
    expert_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    probe_path = source / "initial_state_probe.json"
    initial_probes = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    initial_expert_summary = summarize_experts({policy: item.get("expert_stats", {}) for policy, item in initial_probes.items()})
    rollout_expert_summary = summarize_experts(expert_stats)
    probe_hashes = {policy: [state["state_sha256"] for state in item.get("states", [])] for policy, item in initial_probes.items()}
    identical_probe_states = bool(probe_hashes) and len({tuple(values) for values in probe_hashes.values()}) == 1
    generalization = aggregate_metrics(rows, ("policy", "generalization", "split"))
    policy_table = aggregate_metrics(rows)
    labels = {row["policy"]: row.get("policy_label", row["policy"]) for row in rows}
    for item in policy_table:
        item["label"] = labels[item["policy"]]
    benchmark_result = None
    if benchmark_path:
        path = Path(benchmark_path)
        path = path / "metrics.json" if path.is_dir() else path
        if path.exists():
            benchmark_result = json.loads(path.read_text())
    top_k: dict[str, Any] = {}
    for item in trajectories:
        value = item.get("metadata", {}).get("evaluation_budgets", {}).get("k")
        if value is not None:
            top_k.setdefault(item["policy"], set()).add(value)
    top_k = {name: sorted(values) for name, values in top_k.items()}
    orchestration_tasks = (benchmark_result or {}).get("orchestration_tasks", [])
    k_values = sorted({row["k"] for row in orchestration_tasks})
    intervals = sorted({row["routing_interval"] for row in orchestration_tasks})
    ablation_rows = [{**row, "policy": f"k{row['k']}_interval{row['routing_interval']}"} for row in orchestration_tasks]
    top_k_comparisons = [paired_differences(ablation_rows, f"k{k_values[0]}_interval{interval}", f"k{k}_interval{interval}")
                         for interval in intervals for k in k_values[1:]] if k_values else []
    interval_comparisons = [paired_differences(ablation_rows, f"k{k}_interval{intervals[0]}", f"k{k}_interval{interval}")
                            for k in k_values for interval in intervals[1:]] if intervals else []
    conclusions = []
    for comparison in paired:
        if comparison["status"] != "measured":
            conclusions.append(f"{comparison['candidate']} versus {comparison['baseline']}: unanswered (no paired trajectories).")
            continue
        quality = comparison["mean_delta_task_success"]
        cost = comparison["mean_delta_total_tokens"]
        conclusions.append(f"{comparison['candidate']} versus {comparison['baseline']}: {comparison['paired_tasks']} paired tasks; "
                           f"success-rate change {quality:+.3f}; mean token change {cost:+.1f}. "
                           "These measurements describe this task set and configured backends only.")
    if not trajectories:
        conclusions.append("Quality/cost questions are unanswered because measured trajectories are absent.")
    if not expert_stats:
        conclusions.append("Expert specialization/utilization is unanswered because expert statistics are absent.")
    if not identical_probe_states:
        conclusions.append("Before/after internal expert routing on identical states is unanswered: matched initial-state probes are absent.")
    elif initial_expert_summary:
        conclusions.append("Internal expert comparisons use identical heldout initial states; task-load correlations remain descriptive and do not establish semantic specialization.")
    if not benchmark_result:
        conclusions.append("Inference optimization gains and latency effects are unanswered because measured benchmarks are absent.")
    if len({k for values in top_k.values() for k in values}) < 2 and not top_k_comparisons:
        conclusions.append("The quality effect of changing top-k is unanswered: evaluation contains fewer than two k settings.")
    for comparison in top_k_comparisons + interval_comparisons:
        if comparison["status"] == "measured":
            conclusions.append(f"Routing ablation {comparison['candidate']} versus {comparison['baseline']}: "
                               f"{comparison['paired_tasks']} paired actual heldout runner tasks; "
                               f"success-rate change {comparison['mean_delta_task_success']:+.3f}, "
                               f"mean call change {comparison['mean_delta_agent_calls']:+.2f}, "
                               f"mean wall-time change {comparison['mean_delta_wall_clock_seconds']:+.6f} seconds.")
    result = {"policies": policies, "paired_comparisons": paired, "categories": categories,
              "absolute_policy_metrics": policy_table, "generalization": generalization,
              "top_k_settings": top_k, "top_k_comparisons": top_k_comparisons,
              "routing_interval_comparisons": interval_comparisons,
              "paired_inference_optimizations": (benchmark_result or {}).get("paired_optimizations", []),
              "expert_stats": expert_stats, "initial_state_expert_summary": initial_expert_summary,
              "rollout_expert_summary": rollout_expert_summary, "identical_probe_states": identical_probe_states,
              "conclusions": conclusions,
              "limitations": ["Development deterministic agents do not establish real language-model coordination quality.",
                              "Random-initialized tiny models are not pretrained MoE baselines.",
                              "No GPU cluster performance claim follows from CPU/local measurements."]}
    result["plots"] = plot_results(rows, target / "plots", benchmark_result, initial_expert_summary)
    write_json(target / "analysis.json", result)
    write_csv(target / "paired_comparisons.csv", [{key: value for key, value in item.items() if key != "categories"} for item in paired])
    write_csv(target / "categories.csv", categories)
    write_csv(target / "generalization.csv", generalization)
    write_csv(target / "absolute_policy_metrics.csv", policy_table)
    write_csv(target / "initial_state_experts.csv", initial_expert_summary)
    write_csv(target / "rollout_experts.csv", rollout_expert_summary)
    write_csv(target / "routing_ablations.csv", [{key: value for key, value in item.items() if key != "categories"}
                                               for item in top_k_comparisons + interval_comparisons])
    report = ["# Measured experiment analysis", "", "## Absolute heldout quality and cost", "",
              markdown_table(policy_table, [("label", "Policy"), ("success_rate", "Success"), ("mean_agent_calls", "Calls"),
                                            ("mean_total_tokens", "Tokens"), ("mean_wall_clock_seconds", "Wall s"),
                                            ("p95_wall_clock_seconds", "p95 s"), ("mean_controller_token_fraction", "Controller token share"),
                                            ("mean_controller_overhead_fraction", "Controller latency share"),
                                            ("mean_cost_usd", "Configured cost USD")]),
              "Tokens use backend accounting; tiny controller counts are lexical proxies. Monetary cost uses configured rates; zero rates do not establish free inference. Controller cost share is unavailable at zero total cost.",
              "## Paired differences (candidate minus baseline)", "", *conclusions,
              "## Task categories", "", markdown_table(categories, [("policy", "Policy"), ("category", "Category"),
                                                                       ("split", "Split"), ("generalization", "Generalization"),
                                                                       ("success_rate", "Success"), ("mean_agent_calls", "Calls")]),
              "## Seen families and unseen compositions", "", markdown_table(generalization, [("policy", "Policy"),
                    ("generalization", "Group"), ("split", "Split"), ("task_count", "Tasks"), ("success_rate", "Success"), ("mean_agent_calls", "Calls")]),
              "Unseen compositions are separately heldout task families; results do not establish broader language-model generalization.",
              "## Internal experts on fixed initial states", "", markdown_table(initial_expert_summary, [
                  ("policy", "Policy"), ("layer", "Layer"), ("utilized_fraction", "Utilized fraction"),
                  ("normalized_routing_entropy", "Normalized entropy"), ("load_coefficient_of_variation", "Load CV")]),
              "Initial-state probe hashes match across stages: " + str(identical_probe_states) + ". Rollout statistics reflect different visited states and are not a controlled before/after comparison.",
              "## Scope and missing evidence", "", *result["limitations"],
              "Expert task-load divergence and routing ablations: analysis.json. Top-k settings: " + json.dumps(top_k) + "."]
    (target / "report.md").write_text("\n\n".join(report) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="outputs/evaluation/dev")
    parser.add_argument("--output")
    parser.add_argument("--benchmark", default="outputs/benchmarks/dev")
    args = parser.parse_args()
    analyze(args.evaluation, args.output, args.benchmark)


if __name__ == "__main__":
    main()
