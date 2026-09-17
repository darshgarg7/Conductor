# Recorded experiments

The [support service demonstration](support-demo/report.md) compares rules with
actual trained-controller HTTP routing on synthetic support tickets. It records
diagnostic contract checks separately from HTTP acceptance and concurrent load.
The fixed tickets/runbook are an integration fixture, not an independent customer
quality benchmark. Raw JSON/JSONL/CSV and checksums accompany the report.

The [Granite CPU pilot](granite-pilot/report.md) contains measured post-training,
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
