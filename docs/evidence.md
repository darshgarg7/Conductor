# Claims supported by measurements

The [original Granite pilot](../results/granite-pilot/report.md) and
[completed routing repair](../results/routing-repair/report.md) contain pretrained
SFT, categorical DPO, and held-out evaluation. The original pilot also contains
controller inference benchmarks. Both use CPU and deterministic specialists;
neither establishes NVIDIA performance or superiority over simpler routing.
The tiny-model experiment is a separate development-path result.

## Completed model work

The coordinator uses a pinned pretrained Granite sparse MoE backbone and a
categorical coordination head. SFT and DPO update 942,565 parameters: 445,440
attention/router LoRA parameters and 497,125 head parameters, approximately
0.07% of the 1,335,567,845-parameter coordinator. Saved adapter verification
observed nonzero LoRA B tensors in all 24 expert-router layers; DPO changed the
saved SFT adapters and head. Specialists are outside the training optimizer.

The original pilot inventory contains 36 reference trajectories for 12 training
tasks, plus 18 reference trajectories for six held-out tasks. It produces 72
SFT examples and 216 exact-state preference pairs. Task-level partitioning
leaves nine optimizer-training tasks and three internal validation tasks:
54 SFT fitting examples/18 validation examples and 162 DPO fitting pairs/54
validation pairs. These counts are different units and must not be added.

SFT and DPO each solved two of six held-out tasks; the Base MoE with its random
action head solved three and the Rule-Based router solved six. DPO's 92.6%
internal-validation preference-ranking accuracy is **not task success**. This
pilot supports a post-training claim, not an accuracy-improvement claim.

## Completed routing repair

The repair uses 36 training tasks and 108 reference trajectories on those tasks.
Counterfactual curation yields 180 state labels and 360 preference pairs. The
27-task fitting partition contains 135 SFT states and 270 DPO pairs; the
nine-task internal validation partition contains 45 states and 90 pairs. These
counts describe distinct units, not independent samples to sum together.

Both repaired checkpoints solve 32/48 tasks on the same new inventory where
the original SFT and DPO checkpoints each solve 16/48. The repaired result is
32/32 on seen templates and 0/16 on unseen compositions. Rules solve 48/48,
Random Top-K 39/48, the frozen-backbone fitted-head control 24/48, and one-round
sequential dense execution 40/48. DPO does not improve task success over SFT.

The repair bundles a 256-token priority representation, teacher curation, more
data, and head warmup; original checkpoints retain their 128-token inputs.
This is an observed improvement over the original checkpoints on this inventory,
not evidence identifying one causal fix, preserving broad quality, or requiring
a MoE. The raw outcomes remain in the
[repair report](../results/routing-repair/report.md); the
[combined failure analysis](routing_failure_analysis.md) examines both studies.

Coverage remains narrow: all 180 chosen labels are stopping or single-agent
calls, with no chosen multi-agent mode/order examples. Internal validation has
code and retrieval tasks but no math tasks. Its accuracy therefore cannot
validate math routing, dependency execution, or unseen composition transfer.
The [factorization/coverage design](factorized_routing.md) specifies future work;
it is not implemented evidence. Neither repaired model has been shown to pass
the separate customer-support acceptance gates.

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
artifacts produce a nonzero exit. Claim inputs are declarations to verify,
not a source of measurements.

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
throughput timing. CPU measurements must retain the CPU label. In the original
matched k=2/concurrency-four comparison, dynamic batching records 0.843×
throughput and 1.133× p95 latency relative to queued individual inference;
those regressions remain published, not relabeled as optimizations.

The proposed [coordination-v2 protocol](coordination_v2_protocol.md) freezes
eight ordered steps and seeds 42, 137, and 2027. Its
[YAML](../configs/research/coordination_v2_protocol.yaml) is a declarative plan,
not an executable configuration or achieved result. Both inspected pilot and
repair inventories are development evidence; a new final suite must be sealed
before its outcomes guide any claim. The plan supplies no additional training,
GPU, inference-saving, or production-readiness evidence.
