# Resume claims and experiment evidence

Conductor is a research implementation with local validation and a path to GPU
experiments. The original resume numbers—50,000 trajectories, 84% to 91% task
success, 35% inference cost reduction and 28% p95 routing latency reduction—have
no supplied supporting experiment logs. Do not use them.

The checked-in development experiments use synthetic task templates and frozen,
deterministic specialists. The tiny controller is randomly initialized; its
results establish that the pipeline executes, not that a pretrained language
model was post-trained. Downloading a pretrained checkpoint also does not
establish a training result. A real pretrained run must save completed SFT and
preference training artifacts, the immutable base revision and measured held-out
outcomes before that claim becomes appropriate.

## Wording supported by the implementation

Use these engineering statements while numerical and pretrained experiments are
pending:

- Built a configurable sparse MoE coordination research system with categorical
  SFT and DPO objectives, frozen specialist interfaces, counterfactual routing
  preferences, held-out evaluation and expert utilization instrumentation.
- Implemented bounded asynchronous routing serving, compatible request batching,
  state tokenization caching, resumable training checkpoints, SLURM launch scripts
  and machine-readable inference benchmarks with explicit device synchronization.

Change “implemented” to “benchmarked” only when the corresponding raw timing
records exist. “Reduced inference cost” requires meaningful model token billing
or measured GPU time; synthetic whitespace token proxies and configured prices
do not establish real savings. GPU deployment paths alone do not support
“NVIDIA production validated.” No NVIDIA device was available for local validation.

After completing a genuine pretrained run, a precise alternative is:

> Post-trained a pinned open-weight MoE coordination backbone with LoRA and a
> constrained routing head using SFT and DPO on **[actual training count]**
> synthetic execution trajectories; evaluated agent activation and task success
> on **[actual held-out count]** tasks with frozen development specialists.

Replace the brackets from saved artifacts, and name the local device. If DPO
lowers task quality, report the result rather than presenting the objective as a
successful cost optimization. Training examples, preference pairs, tasks and
trajectories are distinct counts.

## Generate an evidence inventory

Run the read-only auditor against the artifacts actually used:

```bash
python -m conductor.audit \
  --dataset data/dev \
  --checkpoint outputs/checkpoints/tiny-sft \
  --run outputs/evaluation/dev \
  --resume-claims \
  --output outputs/evidence/development.json
```

`--dataset`, `--checkpoint`, `--run` and `--doctor` are repeatable. A directory
containing multiple dataset shards is checked for duplicate record IDs and task
ID/public-text collisions. Audit counts are unusable when an inventory is
invalid. Independent inventories are not summed to manufacture a scale claim.
Legacy unsealed JSONL is counted and explicitly distinguished from newly
checksum-verified journals. The auditor does not modify or repair damaged data.

For automation, `--strict` exits nonzero if an artifact is invalid or a declared
claim is unsupported. The original resume claims are declarative inputs, never
measurements. A successful CUDA arithmetic doctor probe establishes basic
compatibility only; it does not count as model validation.

A JSON claim file can request narrow artifact checks:

```json
[
  {"kind": "training_trajectory_count", "minimum": 100},
  {"kind": "pretrained_post_training", "stages": ["sft", "preference"]}
]
```

```bash
python -m conductor.audit \
  --dataset data/generated/YOUR_RUN \
  --checkpoint outputs/checkpoints/YOUR_SFT \
  --checkpoint outputs/checkpoints/YOUR_PREFERENCE \
  --run outputs/evaluation/YOUR_RUN \
  --claims YOUR_CLAIMS.json --strict \
  --output outputs/evidence/YOUR_RUN.json
```

Pretrained checks require HF metadata, a pinned base identity, inference weight
files, completed positive training updates, saved run provenance and dataset
hashes, unchanged specialists and a saved DPO reference fingerprint. This is an
artifact consistency audit, not independent authentication of an experiment.
Comparative performance claims receive no automatic endorsement: inspect paired
raw measurements, matching task/specialist provenance, billing status,
uncertainty intervals and the analysis claim gates.

## Keep the evidence with each claim

Archive the source commit, configuration, random seed, data/checkpoint hashes,
run hardware, all raw measurements, failure counts and the exact comparison.
Disclose whether held-out examples share template families with training and
whether downstream agents are development fixtures or actual language models.
Use repeated matched timing trials for latency claims and report the workload,
context length, batch size, concurrency, precision and device. Keep CUDA event
forward timings separate from wall-clock queue-inclusive service latency.
