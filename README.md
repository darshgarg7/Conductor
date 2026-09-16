# Conductor

**Post-training a sparse MoE for multi-agent coordination, with measured controller inference.**

Conductor tests whether a pretrained Mixture-of-Experts model can learn which
frozen specialists a task needs. A coordination head predicts agent selection,
execution mode and stopping; LoRA adapts attention and neural expert-router
projections. Specialist parameters stay frozen. The study compares supervised
post-training and categorical DPO with dense, rule-based, random and prompted
orchestration.

```text
Task + execution state → pretrained MoE + coordination head → constrained agent top-k
                      → frozen specialists → updated state → controller
```

The recorded pilot uses a pinned Granite sparse MoE on a Mac CPU and inexpensive
deterministic specialists. **NVIDIA execution, NCCL and the container image still
need validation on a GPU host.** The CUDA/SLURM paths are implemented; CPU results
are not GPU performance claims.

Read the [measured research report](results/granite-pilot/report.md),
[design decisions](docs/design_decisions.md) or [code review guide](docs/review_guide.md).

## What to inspect

| Area | Implementation | Evidence |
| --- | --- | --- |
| MoE post-training | [HF controller](conductor/controller/hf.py), [SFT/DPO loop](conductor/training/runner.py) | Pinned pretrained backbone, attention/router LoRA, frozen base parameters, action masks, exact SFT reference |
| Data and evaluation | [Counterfactual trajectories](conductor/generate.py), [held-out evaluation](conductor/evaluate.py) | Task-level splits, initial/late stopping preferences, failures, fixed specialists/budgets, task-paired intervals |
| Controller inference | [Benchmarks](conductor/benchmark.py), [stage profiling](conductor/inference/profiling.py) | Replayed execution states, queue-inclusive latency, native batches, actual HF tokens, device timing and allocator memory |
| Serving and recovery | [Exclusive model worker](conductor/serving/engine.py), [checkpoint state](conductor/training/state.py) | Bounded admission, compatible dynamic batches, deadlines, atomic optimizer windows, per-rank RNG/cursor |

## Recorded CPU pilot

| Measurement | Recorded scope |
| --- | --- |
| Coordinator | Pinned Granite sparse MoE, 1.34B parameters including the action head |
| Post-training | 942,565 trainable parameters (0.07%); attention/router LoRA and categorical SFT → DPO |
| Development corpus | 36 training-task reference trajectories; 72 SFT labels and 216 preference pairs, including task-level validation |
| Held-out evaluation | Seven policies on six synthetic tasks with eight frozen specialists |
| Controller benchmark | 30 CPU configurations, 720 timed requests, eight replayed states; no reliable batching/cache speedup established |
| Main finding | SFT and DPO each solved 2/6 tasks; Base MoE solved 3/6 and rules solved 6/6 |

Post-training reduced training loss but collapsed held-out routing to coder then
stop. DPO improved preference ranking without improving task success. This
pilot establishes an executable post-training and measurement pipeline; the
research hypothesis remains unresolved. Stronger sequential dense controls and
matched controller timing results are included in the report.

The corpus uses arithmetic, lookup and string transformation tasks, plus held-out
compositions. These fixtures make routing outcomes reproducible; they do not
establish broad language-task accuracy or production LLM cost savings. The base
MoE comparator has the same pretrained backbone and an untrained action head.
Agent top-k and internal neural expert top-k are separate controls.

## Reproduce

Use Python 3.12. The recorded environment, configurations and per-phase Git
commits are archived with the report. HF/PEFT versions are pinned to the tested
API; install a suitable PyTorch build for your execution device.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,serve,profiling]'
pytest -q
bash scripts/run_granite_pilot.sh
```

The pretrained pilot downloads model weights and runs genuine CPU training.
For the smaller randomly initialized controller, use `bash scripts/run_dev.sh`.
Working datasets and weights are ignored; the report archives copies of pilot
data, raw measurements and checkpoint metadata. Regenerate weights from the pinned base.
Use a fresh output directory for a new experiment or `--resume` for compatible
trusted checkpoints.

```bash
python -m conductor.train --config configs/training/granite_sft.yaml
python -m conductor.train --config configs/training/granite_preference.yaml
python -m conductor.evaluate --config configs/evaluation/granite_pilot.yaml
python -m conductor.benchmark --config configs/inference/granite_pilot.yaml
python scripts/smoke_serving.py --config configs/serving/granite_pilot.yaml
```

The [results directory](results/granite-pilot/report.md) contains the measured
pilot, sealed development data, request timings, expert probes and per-phase
provenance. `scripts/report_pilot.py` rebuilds this recorded report and training
plot; it rejects changed measurements so the interpretation cannot silently
carry over to a new experiment. Use `conductor.analyze` for new runs.
Working outputs, environments, weights and temporary
files are excluded from the repository.

## NVIDIA and SLURM

[Cluster instructions](docs/hpc.md) cover OLMoE, GPU allocation, sharded
trajectory generation, single-node torchrun, sweeps and recovery. The
[NVIDIA container recipe](containers/Dockerfile.nvidia) preserves NGC's PyTorch
stack. [Deployment notes](docs/deployment.md) describe authentication, artifact
export, resource limits and the remaining validation boundary.

```bash
python -m conductor.doctor --device cuda:0 --dtype bfloat16 --require-cuda --probe
bash scripts/validate_nvidia.sh configs/inference/nvidia.yaml CHECKPOINT
sbatch scripts/slurm/distributed_sft.slurm configs/training/olmoe_sft.yaml
python -m conductor.controller.export --checkpoint CHECKPOINT \
  --output outputs/exports/controller --device cpu --dtype float32
```

Serving owns one model worker per process. A client timeout cancels delivery,
not an already-running accelerator kernel; a stalled process needs a supervisor
restart. State/tokenization caches are bounded. The classification path does not
claim validated generative KV-cache reuse.

## Measurement boundaries

Benchmarks distinguish input tokens/second from generated tokens/second,
synchronized forward timing from queue-inclusive request latency, and allocated
CUDA memory from reserved memory and process RSS. Profiler traces are diagnostic
samples outside throughput timing. Configuration order is seeded and randomized;
repeats are grouped within each configuration.

Development-tool tokens are documented estimates. Zero price rates mean no
monetary cost model is configured. Missing baselines, failed calls, uncertain
billing and small paired corpora remain visible. The [research protocol](docs/research_protocol.md)
and [evidence audit](docs/evidence.md) define what a reported improvement
requires. Confidence is action probability, not calibrated task-success probability.

Future validation needs frozen LLM specialists, larger independent held-out
corpora, multiple seeds and real NVIDIA measurements. Those experiments should
answer the research hypothesis rather than assume it.
