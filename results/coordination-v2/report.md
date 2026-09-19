# Coordination-v2 development-gate study

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
540 trajectories, including
232 successes and
308 failures.
Successful public-state-rule trajectories yielded 186
fitting states and 93 group-disjoint
development states. Coverage includes initial, intermediate, handoff, failure,
stopping, one/two/three-agent, parallel, and sequential labels. Repeated
single-specialist ablations solved zero tasks.

The coordinator is pinned `ibm-granite/granite-3.1-1b-a400m-base`. Rank-four
LoRA plus the action head trained 942,565
of 1,335,567,845 controller
parameters (0.071%). Saved-tensor checks observed updates in all 144 adapter
tensors for every seed, including attention and expert-router targets. The eight
deterministic specialists remained outside the optimizer.

## Online development gates

| Policy | Success by seed | Mean calls | Mean CPU wall time |
| --- | ---: | ---: | ---: |
| Public-state rules | 36/36, 36/36, 36/36 | 2.83 | 0.047 s |
| Frozen catalog head | 36/36, 36/36, 36/36 | 2.83 | 1.959 s |
| Factorized head | 25/36, 27/36, 6/36 | 2.92 | 3.313 s |
| LoRA SFT | 36/36, 36/36, 36/36 | 2.86 | 1.490 s |

![Coordination-v2 development results](development.png)

Public-state rules, sparse linear routing, the small MLP, the fitted catalog
head, and LoRA SFT each solve 36/36 tasks for all three seeds. The factorized
head is unstable: 25/36, 27/36, and 6/36. Under the frozen rule, only the
all-seed-passing catalog head advanced to LoRA.

LoRA SFT also solves 36/36 for every seed, but does not reduce activation count:
it averages 2.86 calls per task versus
2.83 for rules. Its descriptive one-repetition
CPU wall time averages 1.490 seconds per
task versus 0.047 seconds for rules.
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
