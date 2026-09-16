# Reviewing Conductor

Conductor investigates learned sparse coordination. Review the model
post-training, inference measurements and recovery behavior together: the
controller is both a learned policy and a component on the execution path.

Start with the measured research report linked from the README. It separates
completed CPU experiments from NVIDIA work that still needs device validation.
The pilot is deliberately small and uses frozen deterministic specialists;
that makes routing outcomes cheap to reproduce, but limits the research claim.

For model work, inspect [the HF controller](../conductor/controller/hf.py),
[training loop](../conductor/training/runner.py) and
[counterfactual generator](../conductor/generate.py). The distinctive choices
are coordinator-only attention/router LoRA, structured action masks, categorical
DPO with the exact SFT reference, task-level validation splits and expert probes
on identical states before and after training.

For inference work, inspect [request benchmarks](../conductor/benchmark.py),
[stage profiling](../conductor/inference/profiling.py) and
[service ownership](../conductor/serving/engine.py). They distinguish CPU/GPU
measurements, native batch and queue-inclusive latency, actual tokenizer counts,
cache behavior, reserved/allocated memory and profiler overhead.

For reliability, inspect [training tests](../tests/test_training.py),
[service tests](../tests/test_serving.py) and
[dataset recovery tests](../tests/test_datasets.py). The useful regressions
exercise exact resume, partial accumulation, uneven two-worker Gloo shards,
request identity, overload/deadlines and durable counterfactual replay.
CUDA tests are marked and skipped on hosts without a GPU; the validation script
rejects those skips rather than counting them as NVIDIA evidence.

The [design notes](design_decisions.md) explain tradeoffs and remaining limits.
A next research run should use frozen LLM specialists, larger independent task
corpora, multiple seeds and a real NVIDIA host. The repository includes the
launch and measurement paths for that run; it does not claim those experiments
have already happened.
