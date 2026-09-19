"""Publish the recorded coordination-v2 development-gate study.

This report intentionally stops where the frozen protocol stopped.  It refuses
different inputs rather than carrying the recorded interpretation onto a new
experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from conductor.datasets.integrity import file_digest
from conductor.utils.runs import write_json


RECORDED_INPUT_SHA256 = "7e3e944d25a4741580e2c5345379cb233f14644ec9c7859d0c376c9cdc59a600"
SUMMARY_NAMES = ("cheap", "probe", "sft", "preference", "dpo")
SEEDS = (42, 137, 2027)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _recorded_fingerprint(data: Path, output: Path) -> tuple[str, dict[str, str]]:
    paths = [data / "manifest.json", *(output / f"{name}_stage_summary.json" for name in SUMMARY_NAMES)]
    hashes = {str(path): file_digest(path) for path in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return fingerprint, hashes


def _policy_rows(output: Path, stage: str, policy: str) -> list[dict[str, float]]:
    values = []
    for seed in SEEDS:
        path = output / stage / f"seed-{seed}" / "per_policy.csv"
        row = next(item for item in csv.DictReader(path.open()) if item["policy"] == policy)
        values.append({key: float(row[key]) for key in (
            "success_rate", "mean_agent_calls", "mean_total_tokens", "mean_controller_tokens",
            "mean_downstream_tokens", "mean_wall_clock_seconds", "mean_controller_seconds")})
    return values


def _aggregate(output: Path, stage: str, policy: str, label: str) -> dict[str, Any]:
    values = _policy_rows(output, stage, policy)
    return {
        "policy": policy, "label": label, "seeds": list(SEEDS),
        "success_by_seed": [item["success_rate"] for item in values],
        **{f"mean_{key}": statistics.fmean(item[key] for item in values) for key in values[0]},
    }


def _copy_measurements(data: Path, output: Path, archive: Path) -> None:
    target = archive / "measurements"
    target.mkdir(parents=True, exist_ok=True)
    sources = {
        data / "manifest.json": target / "data_manifest.json",
        data / "coverage_audit.json": target / "coverage_audit.json",
        data / "capability_isolation.json": target / "capability_isolation.json",
        **{output / f"{name}_stage_summary.json": target / f"{name}_stage_summary.json"
           for name in SUMMARY_NAMES},
    }
    for seed in SEEDS:
        sources[data / "preferences" / f"seed-{seed}" / "manifest.json"] = (
            target / "preferences" / f"seed-{seed}" / "manifest.json")
        sources[data / "preferences" / f"seed-{seed}" / "preference_audit.json"] = (
            target / "preferences" / f"seed-{seed}" / "preference_audit.json")
        for stage in ("development", "development-probe", "development-sft"):
            for name in ("task_metrics.csv", "per_policy.csv", "per_family.csv",
                         "per_dependency_stage.csv", "policy_status.json"):
                sources[output / stage / f"seed-{seed}" / name] = (
                    target / stage / f"seed-{seed}" / name)
    for source, destination in sources.items():
        if not source.exists():
            raise FileNotFoundError(f"recorded coordination-v2 artifact is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _provenance(output: Path) -> dict[str, Any]:
    paths = [output / "data-generation" / "run.json", output / "probe" / "run.json"]
    for seed in SEEDS:
        paths.extend([
            output / "checkpoints" / "granite-catalog-sft" / f"seed-{seed}" / "run.json",
            output / "preference-generation" / f"seed-{seed}" / "run.json",
        ])
    records = []
    for path in paths:
        run = _load(path)
        records.append({
            "path": str(path), "git_commit": run.get("git_commit"), "git_dirty": run.get("git_dirty"),
            "seed": run.get("seed"), "runtime_seconds": run.get("runtime_seconds"),
            "hardware": run.get("hardware"), "library_versions": run.get("library_versions"),
        })
    return {
        "runs": records,
        "source_commits": sorted({record["git_commit"] for record in records if record["git_commit"]}),
        "dirty_run_note": "Generated working data was untracked during early phases; source commit IDs remain recorded. "
                          "The public archive contains byte hashes for every retained measurement.",
    }


def _plot(rows: list[dict[str, Any]], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [row["label"] for row in rows]
    colors = ["#0f766e", "#2563eb", "#dc2626", "#7c3aed"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for index, seed in enumerate(SEEDS):
        positions = [position + (index - 1) * .22 for position in range(len(rows))]
        axes[0].bar(positions, [row["success_by_seed"][index] for row in rows], width=.21,
                    label=f"seed {seed}", alpha=.85)
    axes[0].set(title="Online development success", ylabel="Success rate", ylim=(0, 1.08),
                xticks=range(len(rows)), xticklabels=labels)
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [row["mean_mean_agent_calls"] for row in rows], color=colors)
    axes[1].set(title="Mean specialist activations", ylabel="Calls per task")
    axes[2].bar(labels, [row["mean_mean_wall_clock_seconds"] for row in rows], color=colors)
    axes[2].set(title="Descriptive CPU wall time", ylabel="Seconds per task", yscale="log")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Coordination-v2 development gates (36 tasks per seed)", fontweight="bold")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Policy | Success by seed | Mean calls | Mean CPU wall time |",
             "| --- | ---: | ---: | ---: |"]
    for row in rows:
        successes = ", ".join(f"{round(value * 36)}/36" for value in row["success_by_seed"])
        lines.append(f"| {row['label']} | {successes} | {row['mean_mean_agent_calls']:.2f} | "
                     f"{row['mean_mean_wall_clock_seconds']:.3f} s |")
    return "\n".join(lines)


def publish(data: Path, output: Path, archive: Path) -> None:
    fingerprint, input_hashes = _recorded_fingerprint(data, output)
    if fingerprint != RECORDED_INPUT_SHA256:
        raise ValueError("recorded coordination-v2 inputs changed; write a new interpretation for a new experiment")
    archive.mkdir(parents=True, exist_ok=True)
    _copy_measurements(data, output, archive)
    data_manifest = _load(data / "manifest.json")
    sft = _load(output / "sft_stage_summary.json")
    preferences = _load(output / "preference_stage_summary.json")
    dpo = _load(output / "dpo_stage_summary.json")
    rows = [
        _aggregate(output, "development-sft", "public_state_rules", "Public-state rules"),
        _aggregate(output, "development-probe", "frozen_catalog_head", "Frozen catalog head"),
        _aggregate(output, "development-probe", "frozen_factorized_head", "Factorized head"),
        _aggregate(output, "development-sft", "conductor_sft", "LoRA SFT"),
    ]
    summary = {
        "scope": "CPU development-gate study on controlled synthetic workflows; no final test or GPU run",
        "recorded_input_sha256": fingerprint, "input_hashes": input_hashes,
        "tasks": {key: data_manifest[key] for key in ("task_count", "train_tasks", "development_tasks",
                                                        "trajectory_count", "trajectory_successes",
                                                        "sft_train_records", "sft_development_records")},
        "development": rows,
        "sft": {"all_seeds_passed": sft["all_seeds_passed"],
                "trainable_parameters": sft["training_metrics"]["42"]["trainable_parameters"],
                "total_controller_parameters": sft["training_metrics"]["42"]["total_controller_parameters"],
                "adapter_update_checks": {seed: sft["adapter_updates"][seed]["passed"] for seed in sft["adapter_updates"]}},
        "preferences": preferences["per_seed"],
        "dpo": dpo,
        "final_inventory_generated": False,
        "study_outcome": "blocked_before_dpo_and_final_under_predeclared_all_seed_gate",
    }
    write_json(archive / "measurement_summary.json", summary)
    write_json(archive / "provenance.json", _provenance(output))
    write_json(archive / "omitted_artifacts.json", {
        "model_checkpoints": {seed: {"path": path,
                                      "actor_checkpoint_sha256": preferences["per_seed"][seed]["actor_checkpoint_sha256"]}
                              for seed, path in sft["checkpoints"].items()},
        "omitted": [
            {"kind": "checkpoint_weights_and_optimizer_state", "reason": "large reproducible working artifacts"},
            {"kind": "full_training_and_development_trajectories", "reason": "compact archive retains task metrics and audits"},
            {"kind": "final_inventory", "reason": "not generated after the DPO eligibility gate failed"},
        ],
    })
    _plot(rows, archive / "development.png")
    report = f"""# Coordination-v2 development-gate study

This run tested whether a pinned pretrained Granite sparse MoE could learn a
multi-step coordination policy after the original controller collapsed on
unseen compositions. The redesign fixed the state boundary, added genuine
dependency workflows, audited complete routing labels, compared cheap learned
controls, and repeated model fitting across seeds 42, 137, and 2027.

The run reached LoRA SFT and then stopped at the predeclared preference-data
gate. **No DPO checkpoint or fresh final-test result exists for coordination-v2.**
That is the intended fail-closed behavior, not a missing metric.

## Workload and training

The controlled inventory contains 72 fitting tasks and 36 development tasks,
balanced across six workflow families. Five fixed policies produced
{data_manifest['trajectory_count']} trajectories, including
{data_manifest['trajectory_successes']} successes and
{data_manifest['trajectory_count'] - data_manifest['trajectory_successes']} failures.
Successful public-state-rule trajectories yielded {data_manifest['sft_train_records']}
fitting states and {data_manifest['sft_development_records']} group-disjoint
development states. Coverage includes initial, intermediate, handoff, failure,
stopping, one/two/three-agent, parallel, and sequential labels. Repeated
single-specialist ablations solved zero tasks.

The coordinator is pinned `ibm-granite/granite-3.1-1b-a400m-base`. Rank-four
LoRA plus the action head trained {sft['training_metrics']['42']['trainable_parameters']:,}
of {sft['training_metrics']['42']['total_controller_parameters']:,} controller
parameters (0.071%). Saved-tensor checks observed updates in all 144 adapter
tensors for every seed, including attention and expert-router targets. The eight
deterministic specialists remained outside the optimizer.

## Online development gates

{_table(rows)}

![Coordination-v2 development results](development.png)

Public-state rules, sparse linear routing, the small MLP, the fitted catalog
head, and LoRA SFT each solve 36/36 tasks for all three seeds. The factorized
head is unstable: 25/36, 27/36, and 6/36. Under the frozen rule, only the
all-seed-passing catalog head advanced to LoRA.

LoRA SFT also solves 36/36 for every seed, but does not reduce activation count:
it averages {rows[3]['mean_mean_agent_calls']:.2f} calls per task versus
{rows[0]['mean_mean_agent_calls']:.2f} for rules. Its descriptive one-repetition
CPU wall time averages {rows[3]['mean_mean_wall_clock_seconds']:.3f} seconds per
task versus {rows[0]['mean_mean_wall_clock_seconds']:.3f} seconds for rules.
Model construction is outside trajectory latency. These development timings
selected and gated models; they are not a final inference benchmark or a GPU claim.

## Why DPO and final evaluation stopped

On-policy collection compared each SFT request with a measured public-rule
rescue from the identical public state under a shared continuation. Seed 42
produced 0 fitting and 0 development pairs; seed 137 produced 0 and 1; seed
2027 produced 2 and 3. The other actions matched the rescue policy (279, 279,
and 274 exclusions respectively). Seeds 42 and 137 therefore had no legal DPO
fitting corpus. The protocol forbids pooling seeds, fitting on development
pairs, or manufacturing negative examples.

The all-seed DPO requirement failed, so DPO did not start and the 120-task final
inventory was intentionally never generated. Development success cannot be
reported as held-out generalization: these families and representation choices
were used for selection. Coordination-v2 supports a multi-seed pretrained-MoE
SFT implementation claim and documents a failed factorized-head/DPO hypothesis;
it does not support preference-optimization, quality-improvement, cost-saving,
NVIDIA-performance, or production-readiness claims.

## Interpretation

The redesigned SFT controller can reproduce a deterministic workflow teacher
across the known development families. A 1.3B-parameter coordinator is still
the wrong operational choice for this fixture because the public rule policy
matches its quality and calls with much lower CPU overhead. DPO could not earn
its place: the greedy SFT actors supplied too few mistakes to optimize under
the committed on-policy rule.

A new confirmatory cycle should predeclare harder and independently held
workflows plus a sampling or exploration policy for preference collection.
Those changes require a new protocol and a new unopened final suite; they cannot
be retrofitted into this result.

## Evidence

[Measurement summary](measurement_summary.json), [coverage audit](measurements/coverage_audit.json),
[SFT stage](measurements/sft_stage_summary.json), [preference stage](measurements/preference_stage_summary.json),
[DPO gate](measurements/dpo_stage_summary.json), [provenance](provenance.json),
and per-seed task/family tables under `measurements/` are retained. Large
checkpoint and trajectory files are described in [omitted artifacts](omitted_artifacts.json).
All published files are covered by [checksums](checksums.json).
"""
    (archive / "report.md").write_text(report)
    write_json(archive / "checksums.json", {
        str(path.relative_to(archive)): file_digest(path)
        for path in sorted(archive.rglob("*")) if path.is_file() and path.name != "checksums.json"
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/coordination-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/coordination-v2"))
    parser.add_argument("--archive", type=Path, default=Path("results/coordination-v2"))
    args = parser.parse_args()
    publish(args.data, args.output, args.archive)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    print(json.dumps({"archive": str(args.archive), "recorded_input_sha256": RECORDED_INPUT_SHA256,
                      "source_head": commit}, indent=2))


if __name__ == "__main__":
    main()
