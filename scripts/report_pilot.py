"""Render a portable pilot report and training plot from archived measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from conductor.analyze import markdown_table
from conductor.datasets.integrity import file_digest
from conductor.utils.runs import write_json


# The narrative interprets this recorded pilot. New experiments use conductor.analyze
# and need their own interpretation rather than inheriting these observations.
RECORDED_METRICS_SHA256 = "9f6ace1b5601ce7b507eaad36937857bcbab6d7df9e92f402b2748b5458b0e7d"


def report(root: Path) -> None:
    phases = ("generation", "sft", "preference", "evaluation", "inference", "dense-sequential-evaluation",
              "strong-dense-evaluation", "serving", "export-verification")
    inputs = {phase: file_digest(root / "phases" / phase / "metrics.json") for phase in phases}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    if fingerprint != RECORDED_METRICS_SHA256:
        raise ValueError("This report interprets the recorded Granite pilot only. Use conductor.analyze for new measurements.")
    def load(phase: str, name: str = "metrics.json") -> dict[str, Any]:
        return json.loads((root / "phases" / phase / name).read_text())

    sft, dpo, evaluation, inference = [load(phase) for phase in ("sft", "preference", "evaluation", "inference")]
    if not all(item.get("completed") and item.get("pretrained") for item in (sft, dpo)):
        raise ValueError("A completed pretrained SFT and DPO run is required.")
    if len(evaluation["policy_status"]) != 7 or any(row["status"] != "measured" for row in evaluation["policy_status"]):
        raise ValueError("All seven baseline policies must have completed evaluation.")
    model = load("sft", "controller.json")["configuration"]["model"]
    labels = {"all_agent": "All-Agent", "base_moe": "Base MoE (random head)", "conductor_sft": "Conductor-SFT",
              "conductor_preference": "Conductor-Preference", "rule_based": "Rule-Based",
              "random_top_k": "Random Top-K", "static_supervisor": "Static Supervisor"}
    training = []
    for phase, metrics in (("sft", sft), ("preference", dpo)):
        training.append({"stage": "SFT" if phase == "sft" else "DPO", "fitting": metrics["train_examples"], "validation": metrics["validation_examples"],
                         "windows": metrics["optimizer_windows"], "initial": metrics["initial_train"]["loss"],
                         "final": metrics["final_train"]["loss"], "validation_loss": metrics["final_validation"]["loss"]})
    policies = [{**row, "label": labels[row["policy"]],
                 "success": f"{round(row['task_count'] * row['success_rate'])}/{row['task_count']}"}
                for row in evaluation["policies"]]
    dense = []
    for phase, label in (("dense-sequential-evaluation", "One sequential round"), ("strong-dense-evaluation", "Two sequential rounds")):
        item = load(phase)
        if any(row["status"] != "measured" for row in item["policy_status"]):
            raise ValueError(f"Incomplete dense control: {phase}")
        dense += [{**row, "control": label, "success": f"{round(row['task_count'] * row['success_rate'])}/{row['task_count']}"}
                  for row in item["policies"] if row["policy"] == "all_agent"]
    serving, export = load("serving"), load("export-verification")
    if serving.get("status") != "measured" or export.get("status") != "measured":
        raise ValueError("Trained-model serving and export verification must be complete.")
    text = f"""# Granite CPU pilot

Conductor asks whether coordinator-only post-training can improve the quality/cost
tradeoff of sparse multi-agent execution. This first experiment establishes that
the training and inference pipeline runs on a pinned pretrained MoE. It does
**not** establish an accuracy improvement or a batching speedup.

## Setup

The coordinator is `{model['name']}`, revision `{model['revision']}`, with
{sft['total_controller_parameters']:,} parameters including its action head.
Rank-4 LoRA adapts attention and expert-router projections; {sft['trainable_parameters']:,}
parameters are trainable (0.07%). The base weights stay frozen. The 24-layer
backbone selects eight of 32 neural experts per token; this is independent of
the external specialist cap k=2.

Execution uses CPU float32 with SDPA and a 128-token input cap on a Mac. The
recorded environment is Python 3.12.14, PyTorch 2.14.0, Transformers 4.57.6 and
PEFT 0.21.0. CUDA, NCCL, SLURM execution and the NVIDIA container remain unvalidated.
[Run records](provenance.json) retain each phase's actual source commit and seed.

Eight fixed deterministic specialists make development inexpensive and reproducible.
They cover planning, retrieval, research, coding, tools, criticism, verification
and arithmetic. They are outside the coordinator optimizer. These fixtures are
not a production LLM workload.

## Data and post-training

Eighteen synthetic tasks are split into 12 training and six held-out tasks.
All-Agent, Rule-Based and Random Top-K generate 54 reference trajectories: 36
training-task trajectories and 18 held-out trajectories. Generation records 51
successes and three failures. Those are data-generation outcomes, not trained-model
accuracy.

The training inventory has 72 SFT labels and 216 exact-state preference pairs,
including initial and later stopping decisions. A task-level internal split
reserves three training task IDs for validation, leaving nine for fitting.
Twelve dense steps exceed the sparse action budget and are excluded. Preference
replays use a fixed continuation, with 240 durably saved counterfactual trials.
The reward penalizes token units, activations and communication; its latency
weight is zero to avoid learning from noisy development timings.

{markdown_table(training, [('stage','Stage'),('fitting','Fitting examples'),('validation','Validation examples'),('windows','Optimizer windows'),('initial','Initial train loss'),('final','Final train loss'),('validation_loss','Final validation loss')])}

SFT uses categorical cross-entropy; DPO uses chosen/rejected action log probabilities
relative to the exact frozen SFT checkpoint. Loss magnitudes are different
objectives. DPO's final validation pair-ranking accuracy is
{dpo['final_validation']['accuracy']:.1%}; this measures preference ordering, not task success.
[Saved-tensor checks](verification/post_training_verification.json) observe updates
in attention/router adapters and the head. They do not independently audit every
frozen base tensor.

![Training losses](plots/training.png)

## Held-out task outcomes

All policies receive the same six task IDs, fixed specialists, k=2 and common
budgets: three rounds, 12 activations and 8,192 token units. All-Agent is exempt
from the per-step sparse cap. Base MoE uses the same pretrained backbone with an
untrained coordination head.

{markdown_table(policies, [('label','Policy'),('success','Success'),('mean_agent_calls','Mean activations'),('mean_wall_clock_seconds','Mean wall s'),('p95_wall_clock_seconds','p95 wall s')])}

SFT and DPO both route every held-out task to coder and then stop. Lower training
loss and stronger offline preference ranking do not transfer to better task
outcomes. A [label diagnostic](verification/supervision_diagnostics.json) found
two distinct successful first-action targets for every training initial state.
Whole-trajectory success filtering can retain unnecessary random-policy steps.
This is a plausible supervision weakness, not a proven cause of collapse.

The prompted supervisor is the actual frozen Qwen2.5-0.5B-Instruct model. It
returned invalid structured decisions on all six tasks and failed closed; its
result does not establish superiority over a reliable prompted supervisor.
The Rule-Based mapping has a strong task-type prior for these synthetic tasks.

The primary dense baseline calls every specialist in one parallel round.
Dependencies cannot use another specialist's output within that round. Two
additional sequential controls expose this limitation:

{markdown_table(dense, [('control','Dense control'),('success','Success'),('mean_agent_calls','Mean activations')])}

The one-round control uses a fixed capability order; the two-round control uses
registry order and exhausts its call budget after eight plus four activations.
They were added after inspecting the pilot and are sensitivity analyses.
Held-out compositions share synthetic template families with training. Six test
tasks are too few to establish broad generalization; all improvement claim gates
remain insufficient. Any follow-up tuning needs a new locked test set.

![Quality and token usage](plots/quality_cost.png)

## Controller inference

Thirty CPU configurations produce 720 timed requests from eight replayed states
covering math, retrieval and code. Batch limits are one/four, concurrency
one/four and specialist k is one/two/three. Native offline batches and queued
individual/dynamic requests have separate arrival semantics. Request latency
includes queueing; stage profiles and profiler traces are separate diagnostics.

{markdown_table([r for r in inference['paired_optimizations'] if r['k'] == 2], [('batch_size','Batch limit'),('concurrency','Concurrency'),('request_throughput_ratio','Dynamic/single throughput'),('p95_latency_ratio','Dynamic/single p95')])}

At batch limit four, concurrency four and k=2, dynamic batching lowers observed
throughput and raises p95. Ratios favorable at concurrency one cannot be
attributed to multi-request batching. Repeats are grouped within seeded randomized
configuration order, so host variation limits interpretation. No reliable CPU
batching gain is established. A 24-operation serialization microbenchmark also
shows no benefit; no matched end-to-end disabled-cache run is included.

The controller dominates wall time on these inexpensive deterministic tasks
(about 98% for SFT/DPO). HF input tokens are actual tokenizer counts; downstream
tool tokens are estimates. Zero configured prices mean monetary costs are
unknown. Fewer activations do not establish inference-cost savings. CUDA memory,
GPU timing, mixed-precision gains and KV-cache reuse are not measured here.

An actual trained-checkpoint HTTP smoke completes {serving['request_count']}
authenticated requests in batches of two/four/two, rejects unauthenticated calls
with 401 and drains its queue. macOS cleanup reported one leftover semaphore;
this is not a leak-free teardown claim. Trained-model overload/deadline tests
remain pending; unit tests cover their admission/timer logic.

Merged export and reload preserve selected agents, mode and termination across
{export['probe_count']} initial-state probes at k=1/2/3. Maximum reload logit drift is
{export['maximum_reload_logit_difference']:.8g}; masked probability drift is
{export['maximum_reload_probability_difference']:.8g}. Checks apply documented
absolute/relative tolerances, finite values and masked probability/argmax gates.
The on-disk source checkpoint remains unchanged. A preceding copy attempt was
interrupted under memory pressure; an initial merge trial rejected numerical
drift and published nothing. These limited checks are not bitwise equivalence,
task correctness or a measured export speedup.

## Next experiment

Compare reward-filtered SFT targets and context/representation choices before
scaling. Then run multiple seeds with frozen LLM specialists, larger independent
task corpora and a locked test set. A real NVIDIA host must validate CUDA/NCCL,
the container, precision and serving paths, followed by matched inference
measurements. The research hypothesis remains unresolved.

## Evidence and reproduction

[Provenance](provenance.json), [sealed data](data/manifest.json),
[evaluation](phases/evaluation/metrics.json), [request timings](phases/inference/requests.csv),
[expert probes](phases/evaluation/initial_state_probe.json),
[HTTP smoke](phases/serving/metrics.json) and
[export checks](phases/export-verification/metrics.json) are retained.
The [export validation subset](phases/export-verification/validation.json) preserves
numerical fields and file hashes without the original host-specific model path.
Expert counts describe load changes on matched states; they do not establish
semantic specialization. Raw records preserve original working paths and commits.
Weights, optimizer/RNG state, console logs and the large profiler trace remain
local; [omission descriptors](omitted_artifacts.json) retain hashes for the latter
diagnostics. Metadata alone is not an executable checkpoint.

Regenerate model runs using `bash scripts/run_granite_pilot.sh` from the repository
root with the tested environment and fresh working outputs. Sequential controls,
HTTP smoke and export use the corresponding Granite configs and scripts. To
rebuild this report from its existing archive:

```bash
python scripts/report_pilot.py --archive results/granite-pilot
```
"""
    (root / "report.md").write_text(text)
    plot_training(root)
    write_json(root / "measurement_summary.json", {"training": training, "policies": policies,
               "dense_controls": dense, "paired_inference": inference["paired_optimizations"],
               "recorded_metrics_sha256": fingerprint,
               "scope": "CPU synthetic pilot; NVIDIA validation and improvement claims remain unestablished"})
    write_json(root / "checksums.json", {str(path.relative_to(root)): file_digest(path)
               for path in sorted(root.rglob("*")) if path.is_file() and path.name != "checksums.json"})


def plot_training(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, phase, title in zip(axes, ("sft", "preference"), ("SFT cross-entropy", "DPO preference loss")):
        rows = [json.loads(line) for line in (root / "phases" / phase / "training_metrics.jsonl").read_text().splitlines() if line.strip()]
        epochs = [row["epoch"] for row in rows]
        axis.plot(epochs, [r["train_loss"] for r in rows], "o-", color="#2563eb", label="Training epoch")
        axis.plot(epochs, [r["validation_loss"] for r in rows], "s-", color="#d97706", label="Internal validation")
        axis.set(title=title, xlabel="Completed epoch", ylabel="Loss", xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    (root / "plots").mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(root / "plots" / f"training.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("results/granite-pilot"))
    args = parser.parse_args()
    report(args.archive)
    print(args.archive / "report.md")


if __name__ == "__main__":
    main()
