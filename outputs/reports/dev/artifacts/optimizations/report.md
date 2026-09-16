# Measured precision and feature-cache study

Matched local workload and checkpoint; repeated-state feature cache, configured compute dtype.

Sequential runs and finite repeats do not establish statistically significant or GPU gains.

| Variant | Batch | Concurrency | Strategy | Throughput ratio | p95 ratio |
| --- | ---: | ---: | --- | ---: | ---: |
| bfloat16 | 1 | 1 | single | 1.006 | 1.015 |
| bfloat16 | 1 | None | batched | 1.044 | 0.976 |
| bfloat16 | 1 | 1 | dynamic | 1.043 | 0.959 |
| bfloat16 | 1 | 4 | single | 1.026 | 0.971 |
| bfloat16 | 1 | 4 | dynamic | 1.048 | 0.954 |
| bfloat16 | 4 | 1 | single | 1.038 | 0.975 |
| bfloat16 | 4 | None | batched | 1.011 | 1.005 |
| bfloat16 | 4 | 1 | dynamic | 0.976 | 1.024 |
| bfloat16 | 4 | 4 | single | 1.001 | 1.000 |
| bfloat16 | 4 | 4 | dynamic | 1.018 | 0.983 |
| feature_cache | 1 | 1 | single | 3.958 | 0.252 |
| feature_cache | 1 | None | batched | 5.280 | 0.192 |
| feature_cache | 1 | 1 | dynamic | 3.730 | 0.261 |
| feature_cache | 1 | 4 | single | 9.362 | 0.104 |
| feature_cache | 1 | 4 | dynamic | 3.754 | 0.260 |
| feature_cache | 4 | 1 | single | 4.089 | 0.241 |
| feature_cache | 4 | None | batched | 10.718 | 0.093 |
| feature_cache | 4 | 1 | dynamic | 1.221 | 0.805 |
| feature_cache | 4 | 4 | single | 9.146 | 0.110 |
| feature_cache | 4 | 4 | dynamic | 7.379 | 0.129 |
