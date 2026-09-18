# Routing repair: measured CPU results

The original Granite pilot collapsed to coder → stop. This follow-up repairs
state truncation and teacher labels, then evaluates all candidates on a newly
locked 48-task inventory. It tests correctness on synthetic templates, with
the same eight frozen deterministic specialists and execution budgets.

**The repair does not establish a superior coordination policy.** Repaired SFT
solves 32/48 tasks and repaired DPO solves
32/48. Rules solve 48/48
and random sparse routing solves 39/48.
The original SFT and DPO checkpoints solve 16/48
and 16/48 on this same inventory.

Repaired LoRA SFT: 32/32 seen-template tasks, 0/16 unseen compositions; Repaired DPO: 32/32 seen-template tasks, 0/16 unseen compositions. SFT and DPO average
1.000 and
1.000 calls per task, respectively.

The curated label counts are `{'coder': 27, 'math': 12, 'retriever': 14, 'stop': 127}`. Required
multi-agent handoffs and execution modes are absent from this training inventory.
The inspected inventory is now development evidence; the next
experiment needs a fresh locked test with unseen dependency structures.

## Training and selection

Generation uses 36 training tasks and 48 held-out tasks, seed 314159. Evaluation
has eight examples per concrete template. Two composition categories are absent
from training. Inputs, labels, source commits and hardware remain in the raw
phase records; [the committed protocol](../../docs/routing_repair_protocol.md)
describes the selection rule and limitations.

[The tokenizer-only context audit](verification/context_audit.json) covers all
72 original pilot training states. Under the legacy 128-token format,
48/48
later states lack all four critical progress keys in decoded input. Priority
serialization at 256 tokens preserves the complete priority prefix in
72/72
states, truncates 0 task texts and
23 histories. Literal field preservation
does not establish model comprehension. Reproduce with
`python scripts/audit_controller_context.py --output FRESH_AUDIT.json`.

The curator retains 180 distinct-state SFT labels and
360 preference pairs. It selects successful measured
counterfactual winners under a shared rule continuation. That is useful offline
supervision, not evidence of a globally optimal controller.

All four probes share frozen pretrained weights and a 256-token priority input.
The selected representation is **mean/none**. Epoch and
representation selection use internal validation only; no held-out score
selects the model. The unchanged partition fits 27 tasks and reserves nine,
with 135 fitting labels and 45 validation labels. Validation contains three
lookup, three reverse-string and three vowel-count tasks, with no arithmetic
addition task or math-agent target. Of its 45 labels, 32 are stopping actions;
perfect head accuracy describes this limited, correlated validation set.

| Head input | Selected epoch | Global action accuracy | Cross entropy | Feature RMS | Initial logit standard deviation |
| --- | --- | --- | --- | --- | --- |
| last/none | 110 | 1.0000 | 0.0000 | 26.1601 | 14.9339 |
| last/layer_norm | 18 | 0.9556 | 0.2751 | 1.0000 | 0.5695 |
| mean/none | 97 | 1.0000 | 0.0000 | 22.4310 | 12.9649 |
| mean/layer_norm | 200 | 0.9778 | 0.0359 | 1.0000 | 0.5775 |

The fitted head warms up 2-epoch attention/router LoRA SFT, then
1-epoch categorical DPO against the exact frozen new SFT checkpoint.
SFT fits 135 labels and reserves 45;
DPO fits 270 pairs and reserves 90.
The frozen-head candidate is a distinct ablation and does not update the backbone.
DPO validation pair-ranking accuracy is 96.7%;
it is neither full-catalog argmax accuracy nor online task success.

## Online execution

| Policy | Exact success | Mean agent calls | Controller tokens | Tool token estimates | System p95 seconds |
| --- | --- | --- | --- | --- | --- |
| Original DPO | 16/48 | 1.0000 | 216.7917 | 36.3333 | 7.0140 |
| Original SFT | 16/48 | 1.0000 | 216.7917 | 36.3333 | 12.4821 |
| Frozen-backbone head | 24/48 | 1.2500 | 274.5625 | 51.3750 | 9.4933 |
| All-Agent parallel | 32/48 | 8.0000 | 0.0000 | 286.3333 | 0.0112 |
| Pretrained + random head | 27/48 | 3.7917 | 344.7917 | 370.8958 | 6.6023 |
| Repaired DPO | 32/48 | 1.0000 | 231.4583 | 33.3333 | 3.6630 |
| Repaired LoRA SFT | 32/48 | 1.0000 | 231.4583 | 33.3333 | 3.2973 |
| Random top-k | 39/48 | 6.0000 | 0.0000 | 671.7292 | 0.0095 |
| Rules | 48/48 | 1.3333 | 0.0000 | 53.6667 | 0.0095 |
| All-Agent sequential | 40/48 | 8.0000 | 0.0000 | 1306.3333 | 0.0114 |

[Paired differences](paired_repair_comparisons.json) compare identical task IDs.
The old checkpoints keep their original 128-token format. The repaired pipeline
bundles input format, 256-token cap, teacher curation, more data and head warmup;
old/new differences cannot identify a single causal change. Probe comparisons
isolate pooling and normalization within the new representation.

Per-template/category summaries, exact output traces, requested stopping actions,
expert probes and task-weighted routing diagnostics are archived.
[Saved-tensor verification](verification/post_training_verification.json)
checks finite FP32 attention/router LoRA updates, SFT-to-DPO adapter/head changes
and the exact SFT reference identity. It loads adapter/head tensors only;
no complete frozen-base tensor audit or NVIDIA run is claimed. Gzip traces
are round-trip checked against original uncompressed hashes and byte counts.
The permutation
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
