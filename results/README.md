# Recorded experiments

The [completed routing repair](routing-repair/report.md) compares original and
repaired Granite checkpoints on the same 48 tasks. Original SFT/DPO each solve
16/48; repaired SFT/DPO each solve 32/48, comprising 32/32 seen-template successes
and 0/16 unseen-composition successes. Rules solve 48/48, random sparse routing
39/48, a frozen fitted head 24/48, and sequential dense execution 40/48.
The archive preserves raw traces, data units, label-coverage gaps and failures.
It does not establish superiority, monetary savings, or NVIDIA performance.
The [combined failure analysis](../docs/routing_failure_analysis.md) explains
the remaining label-coverage and generalization gaps.

The [support service demonstration](support-demo/report.md) compares rules with
actual trained-controller HTTP routing on synthetic support tickets. It records
diagnostic contract checks separately from HTTP acceptance and concurrent load.
The fixed tickets/runbook are an integration fixture, not an independent customer
quality benchmark. Raw JSON/JSONL/CSV and checksums accompany the report.

The [original Granite CPU pilot](granite-pilot/report.md) retains its 2/6 SFT and
2/6 DPO outcome and measured post-training,
held-out routing and controller inference results. Start with the report;
`phases/` holds raw metrics and configurations, `data/` holds sealed synthetic
records, and `plots/` holds exportable figures.

`provenance.json` preserves each phase's source commit and seed.
`checksums.json` covers every published pilot file. Omitted console logs, local
pointers and large diagnostics have separate hash descriptors. Model weights
and optimizer/RNG state stay in working outputs and must be regenerated.

The earlier random tiny-model development run is retained in
[Git history](https://github.com/darshgarg7/Conductor/tree/b75568efad534005dd6ba3f991cb404f134f9bcf/outputs/reports/dev).
It checks the inexpensive development path and is separate from pretrained-model
evidence. New working runs belong under ignored `outputs/`, not this archive.

The [coordination-v2 protocol](../docs/coordination_v2_protocol.md),
[declarative YAML](../configs/research/coordination_v2_protocol.yaml), and
[factorized-routing design](../docs/factorized_routing.md) propose the next
eight-step, three-seed study. They are not additional completed experiments.
Inspected pilot and repair inventories must not be reused as fresh final tests.
