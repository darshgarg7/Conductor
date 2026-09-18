"""Archive and summarize the locked routing-repair experiment without weights."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

from conductor.analyze import markdown_table
from conductor.datasets.integrity import file_digest
from conductor.evaluation.diagnostics import routing_diagnostics
from conductor.metrics.aggregate import paired_differences, trajectory_metrics
from conductor.utils.runs import write_json


PHASES = ("generation", "probe", "sft", "preference", "reference", "frozen", "evaluation", "dense-sequential")


def archive(source: Path, data: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"use a fresh archive: {output}")
    provenance = {}
    for phase in PHASES:
        run = json.loads((source / phase / "run.json").read_text())
        if run.get("git_dirty") is not False or run.get("runtime_seconds") is None:
            raise ValueError(f"phase must complete from a clean source tree: {phase}")
        provenance[phase] = {key: run[key] for key in ("git_commit", "git_dirty", "seed", "runtime_seconds")}
    for phase in ("sft", "preference"):
        metrics = json.loads((source / phase / "metrics.json").read_text())
        if not metrics.get("completed") or not metrics.get("pretrained") or metrics.get("specialists_updated") is not False:
            raise ValueError(f"incomplete pretrained coordinator-only training: {phase}")
    heldout_ids = None
    merged, display_rows, omissions, compressed = [], [], [], []
    labels = {"reference": {"conductor_sft": "Original SFT", "conductor_preference": "Original DPO"},
              "frozen": {"conductor_sft": "Frozen-backbone head"},
              "evaluation": {"conductor_sft": "Repaired LoRA SFT", "conductor_preference": "Repaired DPO",
                             "all_agent": "All-Agent parallel", "rule_based": "Rules",
                             "random_top_k": "Random top-k", "base_moe": "Pretrained + random head"},
              "dense-sequential": {"all_agent": "All-Agent sequential"}}
    for phase in PHASES:
        directory = source / phase
        if phase in labels:
            audit = json.loads((directory / "split_audit.json").read_text())
            ids = set(audit["heldout_task_ids"])
            if heldout_ids is None:
                heldout_ids = ids
            if ids != heldout_ids or len(ids) != 48:
                raise ValueError("all comparisons must cover the same locked 48 held-out IDs")
            metrics = json.loads((directory / "metrics.json").read_text())
            if any(status["status"] != "measured" for status in metrics["policy_status"]):
                raise ValueError("comparison has an unavailable policy")
            for row in metrics["policies"]:
                label = labels[phase][row["policy"]]
                display_rows.append({**row, "policy": label,
                                     "success": f"{round(row['success_rate'] * row['task_count'])}/{row['task_count']}"})
            for line in (directory / "trajectories.jsonl").read_text().splitlines():
                item = json.loads(line)
                item["policy"] = labels[phase][item["policy"]]
                merged.append(item)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            if (len(relative.parts) == 1 and path.suffix in {".json", ".jsonl", ".csv"}
                    and path.name not in {"checkpoint_pointer.json", "latest_resume.json"}):
                target = output / "phases" / phase / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            elif path.name in {"controller.json", "adapter_config.json"} and relative.parts[0] in {"selected", "adapter"}:
                target = output / "phases" / phase / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            elif len(relative.parts) == 1:
                omissions.append({"phase": phase, "path": str(relative), "sha256": file_digest(path),
                                  "reason": "Weights, features, optimizer state, local pointers and logs remain local."})
    for path in sorted(data.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        target = output / "data" / path.relative_to(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "preference_trials.jsonl":
            with target.with_suffix(".jsonl.gz").open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as packed:
                    with path.open("rb") as handle:
                        shutil.copyfileobj(handle, packed)
            compressed.append({"archive": str(target.with_suffix(".jsonl.gz").relative_to(output)),
                               "uncompressed_sha256": file_digest(path), "uncompressed_bytes": path.stat().st_size})
        else:
            shutil.copyfile(path, target)
    write_json(output / "provenance.json", provenance)
    write_json(output / "omitted_artifacts.json", omissions)
    write_json(output / "compressed_artifacts.json", compressed)
    rows = [trajectory_metrics(item) for item in merged]
    paired = [paired_differences(rows, baseline, candidate, bootstrap_samples=2000, seed=42)
              for baseline in ("Original SFT", "Original DPO", "Frozen-backbone head", "Rules", "All-Agent sequential")
              for candidate in ("Repaired LoRA SFT", "Repaired DPO") if baseline != candidate]
    write_json(output / "paired_repair_comparisons.json", paired)
    routing = routing_diagnostics(merged, seed=42)
    write_json(output / "routing_diagnostics.json", routing)
    probe = json.loads((output / "phases/probe/metrics.json").read_text())
    curator = json.loads((output / "data/curated/metrics.json").read_text())
    sft = json.loads((output / "phases/sft/metrics.json").read_text())
    dpo = json.loads((output / "phases/preference/metrics.json").read_text())
    ablations = [{"representation": name, **result} for name, result in probe["ablations"].items()]
    text = f"""# Routing repair: measured CPU results

The original Granite pilot collapsed to coder → stop. This follow-up repairs
state truncation and teacher labels, then evaluates all candidates on a newly
locked 48-task inventory. It tests correctness on synthetic templates, with
the same eight frozen deterministic specialists and execution budgets.

## Training and selection

Generation uses 36 training tasks and 48 held-out tasks, seed 314159. Evaluation
has eight examples per concrete template. Two composition categories are absent
from training. Inputs, labels, source commits and hardware remain in the raw
phase records; [the committed protocol](../../docs/routing_repair_protocol.md)
describes the selection rule and limitations.

The curator retains {curator['sft_examples']} distinct-state SFT labels and
{curator['preference_examples']} preference pairs. It selects successful measured
counterfactual winners under a shared rule continuation. That is useful offline
supervision, not evidence of a globally optimal controller.

All four probes share frozen pretrained weights and a 256-token priority input.
The selected representation is **{probe['selected_representation']}**. Epoch and
representation selection use internal validation only; no held-out score
selects the model.

{markdown_table(ablations, [('representation','Head input'),('epoch','Selected epoch'),('validation_accuracy','Global action accuracy'),('validation_loss','Cross entropy'),('feature_rms','Feature RMS'),('initial_logit_std','Initial logit standard deviation')])}

The fitted head warms up {sft['epochs']}-epoch attention/router LoRA SFT, then
{dpo['epochs']}-epoch categorical DPO against the exact frozen new SFT checkpoint.
SFT fits {sft['train_examples']} labels and reserves {sft['validation_examples']};
DPO fits {dpo['train_examples']} pairs and reserves {dpo['validation_examples']}.
The frozen-head candidate is a distinct ablation and does not update the backbone.
DPO validation pair-ranking accuracy is {dpo['final_validation']['accuracy']:.1%};
it is neither full-catalog argmax accuracy nor online task success.

## Online execution

{markdown_table(display_rows, [('policy','Policy'),('success','Exact success'),('mean_agent_calls','Mean agent calls'),('mean_controller_tokens','Controller tokens'),('mean_downstream_tokens','Tool token estimates'),('p95_wall_clock_seconds','System p95 seconds')])}

[Paired differences](paired_repair_comparisons.json) compare identical task IDs.
The old checkpoints keep their original 128-token format. The repaired pipeline
bundles input format, 256-token cap, teacher curation, more data and head warmup;
old/new differences cannot identify a single causal change. Probe comparisons
isolate pooling and normalization within the new representation.

Per-template/category summaries, exact output traces, requested stopping actions,
expert probes and task-weighted routing diagnostics are archived. The permutation
null guards against interpreting a unique random route per task as specialization;
association remains descriptive and does not demonstrate semantic MoE experts.

## What this does and does not establish

This is a single-seed repair experiment on related arithmetic, lookup and string
templates. The held-out composition labels support a narrow transfer test. IDs
are distinct, but template siblings are correlated; bootstrap intervals describe
this inventory rather than broad language-task populations. Head selection uses
several internal-validation comparisons and can overfit that partition.

The rule baseline already understands the public task taxonomy, so this corpus
does not establish a need for a 1.3B MoE or superiority over smaller routers.
Dense sequential execution is reported alongside the weaker parallel control.
Fewer calls are not monetary savings: controller tokens are real HF tokenizer
counts, tool tokens are estimates, and configured zero prices leave costs unknown.
CPU timings do not establish NVIDIA performance or an optimization speedup.
CUDA/NCCL, GPU profiling, SLURM and the NVIDIA container remain unvalidated.

Both favorable and unfavorable results remain published. Deployment and support
ticket acceptance require their own gates; this experiment does not validate
the unrelated support fixture or make the learned controller production ready.
"""
    (output / "report.md").write_text(text)
    plot(display_rows, output)
    write_json(output / "checksums.json", {str(path.relative_to(output)): file_digest(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"})


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["policy"] in {"Original SFT", "Frozen-backbone head", "Repaired LoRA SFT", "Repaired DPO", "Rules", "All-Agent sequential"}]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    names = [row["policy"].replace(" ", "\n", 1) for row in selected]
    for axis, field, label in zip(axes, ("success_rate", "mean_agent_calls"), ("Exact task success fraction (48 tasks)", "Mean downstream agent calls")):
        axis.bar(names, [row[field] for row in selected], color="#475569")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("Routing repair: CPU + frozen deterministic specialists; one seed")
    fig.tight_layout()
    fig.savefig(output / "repair.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/research/routing-repair"))
    parser.add_argument("--data", type=Path, default=Path("data/generated/routing-repair"))
    parser.add_argument("--output", type=Path, default=Path("results/routing-repair"))
    args = parser.parse_args()
    archive(args.source, args.data, args.output)


if __name__ == "__main__":
    main()
