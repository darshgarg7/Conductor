# Measured inference benchmarks

Run `python -m conductor.benchmark --config configs/inference/dev.yaml` after training. Use `--checkpoint` to select another trained controller.

Each workload varies batch size, requested context word count, request concurrency, top-k, and routing interval. Warmup runs are excluded. Real synchronized timings produce request throughput, input processing throughput, queue-inclusive request p50/p95, and separate process RSS/CUDA allocator memory. Input accounting comes from the same backend path, including truncated HF attention-mask token counts; tiny models report a lexical proxy. Generated-token throughput is not measured.

`single` admits concurrent requests to one exclusive worker, which executes individual forwards. Worker wait counts as queue delay. `batched` routes native offline batches (or explicitly labeled sequential fallback), with concurrency explicitly inapplicable. `dynamic` uses a reusable asynchronous queue and native batches under actual request concurrency. Its timeout, exceptions, cancellation, and shutdown paths are covered by tests. Controller microbenchmarks reuse one real call over the specified orchestration-step count. Separate `orchestration_tasks.csv` measurements execute actual heldout tasks through the runner at every top-k and interval, recording quality, calls, tokens, and wall latency.

`serialization.csv` compares bounded caching against no caching on identical states. That microbenchmark does not establish an end-to-end inference speedup. Mixed precision and HF cache capabilities are reported as supported/unsupported; they are never simulated. Controller overhead is taken from a specified measured end-to-end evaluation, or remains null.

Artifacts include configuration/checkpoint/git/hardware provenance, raw `requests.csv`, aggregate `benchmark.csv`, `serialization.csv`, optimization statuses, and a human-readable report. `python -m conductor.analyze` exports plots and paired heldout quality/cost analysis. Compare matched workloads and checkpoints before claiming a gain. Local CPU results establish no GPU or cluster performance improvement.

The recorded Granite pilot replays real execution states with per-request hashes.
See [the measured report](../outputs/reports/granite-pilot/research_report.md).
HF `stages.json` separates host serialization/tokenization, transfer, model
forward and decision timing. Optional Chrome profiler traces and NVTX ranges
are diagnostic runs outside ordinary timing samples. Allocated and reserved
CUDA memory are reported separately; non-CUDA measurements remain null.
