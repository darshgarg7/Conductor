# Conductor: Post-Trained MoE Coordination

**Research hypothesis:** a pretrained sparse Mixture-of-Experts language model
can be post-trained to activate the specialists a task needs while preserving
task success and reducing inference cost, token usage, latency and communication.
Conductor treats coordination as a learned model problem: supervised routing
imitation, cost-sensitive preference optimization, held-out evaluation, internal
expert analysis and controller inference benchmarks.

```text
Task + execution state → MoE coordination model → constrained agent top-k
                      → frozen specialists → updated state → controller
```

The working local demonstration uses **a tiny randomly initialized neural
MoE and deterministic specialists**. Its measured results verify the pipeline;
they are not evidence about pretrained language-model post-training or realistic
LLM inference economics. Configurations and a tested Hugging Face/LoRA backend
provide the next GPU experiment. No benchmark number is supplied by a constant.

## Run the complete local experiment

Use Python 3.11–3.14; Python 3.12 is used for the recorded development run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/run_dev.sh
pytest -q
```

Individual stages are reproducible YAML-driven commands:

```bash
python -m conductor.generate --config configs/generation.yaml
python -m conductor.train --config configs/training/tiny_sft.yaml
python -m conductor.train --config configs/training/tiny_preference.yaml \
  --checkpoint outputs/checkpoints/tiny-sft
python -m conductor.evaluate --config configs/evaluation/heldout.yaml
python -m conductor.benchmark --config configs/inference/development.yaml
python scripts/benchmark_optimizations.py --config configs/inference/optimization_study.yaml
python -m conductor.analyze --evaluation outputs/evaluation/dev \
  --benchmark outputs/benchmarks/dev --output outputs/reports/dev
```

Generated trajectories and supervision live in `data/dev/`. Checkpoints,
run metadata, metric JSON/CSV, request timing records and expert statistics
live in `outputs/`. The [measured development report](outputs/reports/dev/report.md)
and exported figures explain observed results and unanswered research questions.
Checkpoint tensors and large generated artifacts are ignored by Git; regenerate
them with the commands above. Never treat checked-in development timings as
portable performance guarantees.

## Pretrained MoE experiment

```bash
python -m pip install -e '.[hf]'
python -m conductor.train --config configs/training/olmoe_sft.yaml
python -m conductor.train --config configs/training/olmoe_preference.yaml \
  --checkpoint outputs/checkpoints/conductor-sft
python -m conductor.evaluate --config configs/evaluation/olmoe.yaml
python -m conductor.benchmark --config configs/inference/olmoe.yaml \
  --checkpoint outputs/checkpoints/conductor-preference
```

The default research model is
[OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924):
1B active and 7B total parameters. Its total weight and training memory remain
substantial. Change `model.name`, revision, device, dtype and adapter settings
in `configs/models/`. Pin a model commit revision for a reproducible large run.
The backend requires an actual MoE architecture; dense models are not silently
accepted. [Transformers OLMoE documentation](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/olmoe)
describes its router-logit interface.

The pretrained backbone produces execution-state representations. A learned
coordination head predicts probabilities over valid complete routing actions:
agent selection and count, execution mode and stopping. Optional LoRA updates
the coordination backbone, including configured neural router projections.
The base comparison uses that same pretrained backbone with an untrained
routing head, explicitly labeled. The tiny backend starts from random weights.
Specialist weights stay frozen in both cases.

SFT minimizes categorical cross entropy on successful training trajectories.
[DPO](https://arxiv.org/abs/2305.18290) operates on categorical action log
probabilities against the frozen SFT policy. Preferences compare exact-state
counterfactual executions with configurable quality/cost reward weights and
normalization scales. The local preference dataset covers initial states;
late-state counterfactual coverage is future work.

## Experimental controls and measurements

Policies include All-Agent, Rule-Based, Random Top-K, a frozen prompting-only
Static Supervisor, Base MoE, Conductor-SFT and Conductor-Preference. The local
run marks Static Supervisor unavailable until `supervisor.name` is configured
with a real frozen language model. The other policies use identical specialists,
task sets and total budgets. All-Agent is exempt from the sparse per-step cap.

Agent top-k (`routing.k`, typically 1–3) is separate from the MoE's internal
expert top-k. Sequential actions preserve agent order. Execution states contain
the public task, category, conversation, previous outputs and routes, tool
results, budgets and step number. Private answer keys only enter the grader.

Evaluation exports task success, grader score, controller and specialist tokens,
activations, rounds, actual elapsed latency and p50/p95, estimated cost,
communication edges, routing sparsity and controller overhead. Internal expert
reports include utilization, load variation, entropy and task-conditioned
activations. HF reports mask padding; expert/task correlations are not causal
specialization evidence.

Controller benchmarks measure real forwards under batching, concurrency,
context sizes, top-k and routing-frequency workloads, plus queue-inclusive
dynamic batching and bounded serialization caching. They export JSON/CSV and
RSS/CUDA memory measurements. Unsupported optimizations, including prefix/KV
caching in the classification backend, are explicit. Improvements are only
claimed from matched measured comparisons. Input-token processing throughput
is distinct from generated-token throughput; the classifier emits no language
tokens. Tiny token counts are documented proxies. Zero cost rates mean a
monetary cost model has not been configured.

The matched optimization study separately measures float32, bfloat16 compute
and repeated-state feature caching, retaining both gains and regressions in
`outputs/benchmarks/optimizations/`. Its warm cache workload does not estimate
the cache hit rate of changing execution states.

## Project map

| Path | Responsibility |
| --- | --- |
| `conductor/controller/` | Tiny sparse MoE, pretrained HF MoE, action head, expert instrumentation |
| `conductor/agents/` | Frozen Planner, Retriever, Researcher, Coder, Tool Executor, Critic, Verifier, Math interfaces |
| `conductor/datasets/`, `conductor/generate.py` | Task splits, trajectories, SFT records, counterfactual preferences |
| `conductor/training/`, `conductor/preference/` | SFT, categorical DPO, configurable rewards |
| `conductor/routing/`, `conductor/orchestration/` | Independent baselines, serialization and budgeted async execution |
| `conductor/evaluation/`, `conductor/metrics/` | Held-out audits, aggregation, paired task comparisons |
| `conductor/inference/`, `benchmarks/` | Synchronized inference timings and dynamic batching |
| `configs/`, `scripts/slurm/` | Reproducible experiments and cluster job templates |
| `tests/`, `docs/` | Behavioral tests, architecture and research protocol |

Every run records configuration, Git commit and dirty status, seed, hardware,
checkpoint, metrics and elapsed runtime. Logging is JSON; optional W&B tracking
is enabled with `tracking.enabled: true` and defaults to offline mode. See
[SLURM/MSI instructions](docs/hpc.md), the [architecture diagram](docs/architecture.md)
and [research protocol](docs/research_protocol.md).

This repository supports two future evidence-backed claims: pretrained MoE
post-training and measured coordinator inference optimization. The development
run alone supports implementation and pipeline validation; a real pretrained
experiment with frozen LLM specialists is required for the broader claims.
