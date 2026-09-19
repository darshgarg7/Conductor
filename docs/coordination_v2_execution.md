# Running coordination-v2

Coordination-v2 turns the failure analysis into an executable, gated study. It
does not assume that a learned policy will pass. Every gate writes a machine-readable
result; a failed gate blocks the dependent stage and remains part of the record.

The workload contains six controlled workflow families. Every request has the
same public task type, and neither the controller nor a specialist receives the
offline family, dependency graph, required route, expected answer, or grader
score. Frozen specialists expose disjoint public capabilities. The private grader
accepts any supported route that produces the correct result and, when requested,
a source-linked verification artifact.

Generate the 72-task fitting and 36-task development inventory with:

```bash
python -m conductor.coordination.study \
  --config configs/research/coordination_v2_study.yaml \
  --stage data
```

This produces five policies per task: public-state rules, random top-k, iterative
sequential All-Agent, parallel All-Agent, and immediate stop. The supervised
records come only from successful public-state-rule trajectories. Before any
fit, generation rejects missing initial, intermediate, handoff, failure,
stopping, stop/one/two/three-agent, parallel, or sequential label cells. It also
replays every workflow with each specialist in isolation and requires zero
single-capability successes.

Fit the sparse linear and small MLP controls for all three fixed seeds and run
their development rollouts with:

```bash
python -m conductor.coordination.study \
  --config configs/research/coordination_v2_study.yaml \
  --stage cheap
```

The primary cheap baseline is locked by the predeclared ordering: family-balanced
workflow success, agent calls, total CPU latency, then identifier. Learned-control
gates use full online workflow success by dependency stage. Classification
accuracy alone cannot advance the study.

If that gate passes, extract the four frozen Granite representation variants
once, fit both complete-catalog and factorized heads for all three seeds, and
measure their online development behavior:

```bash
python -m conductor.coordination.study \
  --config configs/research/coordination_v2_study.yaml \
  --stage probe
```

Only a head type that passes the online gate for all three seeds advances to
rank-four LoRA SFT:

```bash
python -m conductor.coordination.study \
  --config configs/research/coordination_v2_study.yaml \
  --stage sft
```

`conductor/coordination/preferences.py` constructs DPO data only from mistakes
made by a saved SFT checkpoint. Each pair replays the actual action and a public
rule rescue from the identical serialized state under separately instantiated,
hash-identical continuation policies. Training-task pairs are the only fitting
partition; development pairs are diagnostics and validation; final tasks are
never converted into preferences.

The protocol and thresholds were committed before these runs in
`configs/research/coordination_v2_protocol.yaml`. The generated corpus,
development outputs, checkpoints, and later final lock belong under
`data/coordination-v2` and `outputs/coordination-v2`; they are measurements, not
hard-coded documentation values.
