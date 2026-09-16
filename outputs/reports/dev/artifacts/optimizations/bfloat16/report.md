# Measured controller benchmarks

Controller routing only; no downstream agent quality or generated-token throughput measured.

Queue-inclusive per-request p50/p95 after explicit device synchronization.

Input tokens/second uses backend input accounting (tiny lexical proxy or actual truncated HF attention-mask counts), not generation.
Microbenchmark routing interval reports reuse frequency; orchestration_tasks.csv records actual heldout trajectories for each interval and k.
CPU RSS and CUDA allocated memory are reported separately. Unsupported optimizations are not enabled.
Controller overhead: measured evaluation: outputs/evaluation/dev/metrics.json, conductor_preference.

No improvement is claimed from unmatched configurations.
