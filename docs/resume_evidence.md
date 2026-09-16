# Claims supported by measurements

The [recorded Granite pilot](../outputs/reports/granite-pilot/research_report.md)
contains completed pretrained SFT, categorical DPO, held-out evaluation and
controller inference measurements. The original tiny-model experiment is a
separate development-path result. Neither run establishes NVIDIA performance.

## Completed model work

The coordinator uses a pinned pretrained Granite sparse MoE backbone and a
categorical coordination head. SFT and DPO update 942,565 parameters: 445,440
attention/router LoRA parameters and 497,125 head parameters, approximately
0.07% of the 1,335,567,845-parameter coordinator. Saved adapter verification
observed nonzero LoRA B tensors in all 24 expert-router layers; DPO changed the
saved SFT adapters and head. Specialists are outside the training optimizer.

The development inventory contains 36 reference trajectories for 12 training
tasks, plus 18 reference trajectories for six held-out tasks. It produces 72
SFT examples and 216 exact-state preference pairs. Task-level partitioning
leaves nine optimizer-training tasks and three internal validation tasks:
54 SFT fitting examples/18 validation examples and 162 DPO fitting pairs/54
validation pairs. These counts are different units and must not be added.

SFT and DPO each solved two of six held-out tasks; the Base MoE with its random
action head solved three and the Rule-Based router solved six. DPO's 92.6%
internal-validation preference-ranking accuracy is **not task success**. This
pilot supports a post-training claim, not an accuracy-improvement claim.

## Resume wording

- Post-trained a pretrained 1.3B-parameter sparse MoE as a coordination
  controller using attention/router LoRA, supervised fine-tuning and categorical
  DPO, updating 0.07% of parameters with frozen specialist interfaces and
  task-level held-out evaluation.
- Benchmarked pretrained-controller inference using replayed execution states,
  native and dynamic batching, queue-inclusive p50/p95 latency and stage
  profiling; implemented bounded asynchronous serving and resumable DDP/SLURM
  experiment paths.

Add a performance number only with its device, precision, workload, matched
baseline, independent sample count and raw timing evidence. NVIDIA deployment
paths alone do not support NVIDIA validation. Lower external agent top-k does
not reduce the base model's internal expert top-k.

## Audit an experiment

The read-only auditor inventories journals, checkpoints and run provenance:

```bash
python -m conductor.audit \
  --dataset data/generated/granite-pilot \
  --checkpoint outputs/research/granite-pilot/sft \
  --checkpoint outputs/research/granite-pilot/preference \
  --run outputs/research/granite-pilot/evaluation \
  --run outputs/research/granite-pilot/inference \
  --output outputs/evidence/granite-pilot.json
```

`--dataset`, `--checkpoint`, `--run` and `--doctor` are repeatable. Dataset
inventories check duplicate IDs, public task-text collisions and sealed record
checksums. Invalid inventories cannot support a count claim. Independent
inventories are not summed to manufacture scale. The auditor never repairs data.

A JSON claim file can request narrow artifact checks:

```json
[
  {"kind": "training_trajectory_count", "minimum": 36},
  {"kind": "pretrained_post_training", "stages": ["sft", "preference"]}
]
```

Pass it with `--claims CLAIMS.json --strict`; unsupported declarations or invalid
artifacts produce a nonzero exit. `--resume-claims` is a negative-control audit
of unsupported numerical declarations, not a source of measurements.

Pretrained checks require pinned base identity, local weight files, positive
completed optimizer updates, saved provenance, dataset hashes, unchanged
specialists and the DPO reference fingerprint. This checks artifact consistency;
it does not independently authenticate an experiment. Metadata-only report
archives are not executable checkpoints.

## Comparative claims

Inspect task-paired raw measurements, common specialist/budget identities,
failures, billing status and uncertainty intervals. A CUDA arithmetic doctor
probe establishes compatibility only, not model performance. A six-task synthetic
evaluation does not satisfy the default 30-independent-task evidence gate.

Development-tool tokens are estimates. Zero price rates mean no monetary cost
model was configured; neither establishes real inference savings. Tokenization
and serialization caches need matched enabled/disabled measurements before an
end-to-end cache speedup claim. Diagnostic profiler samples stay outside ordinary
throughput timing. CPU measurements must retain the CPU label.
