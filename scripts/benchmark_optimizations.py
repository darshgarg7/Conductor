"""Matched measurements of feature memoization and reduced precision.

Reports observations, including regressions. Cache workloads intentionally reuse
identical states; these results are not claims about unseen-state cache hits.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from conductor.benchmark import benchmark
from conductor.evaluation.io import write_csv
from conductor.utils.config import load_config, merge
from conductor.utils.runs import Run, write_json


async def study(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config.get("output", "outputs/benchmarks/optimizations"))
    run = Run(root, config, config.get("checkpoint"))
    variants = [("float32", {"model": {"dtype": "float32"}, "inference": {"feature_cache_size": 0}}),
                ("bfloat16", {"model": {"dtype": "bfloat16"}, "inference": {"feature_cache_size": 0}})]
    if config.get("model", {}).get("backend", "tiny") == "tiny":
        variants.append(("feature_cache", {"model": {"dtype": "float32"},
                                          "inference": {"feature_cache_size": 1024}}))
    observed = {}
    for name, overrides in variants:
        child = merge(config, overrides)
        child["output"] = str(root / name)
        observed[name] = await benchmark(child)
    fields = ("batch_size", "context_size_requested_words", "concurrency", "k", "routing_interval", "strategy")
    reference = {tuple(row.get(key) for key in fields): row for row in observed["float32"]["benchmarks"]}
    pairs = []
    for name, result in observed.items():
        if name == "float32":
            continue
        for row in result["benchmarks"]:
            baseline = reference[tuple(row.get(key) for key in fields)]
            pairs.append({"optimization": name, **{key: row.get(key) for key in fields},
                          "baseline_requests_per_second": baseline["request_per_second"],
                          "candidate_requests_per_second": row["request_per_second"],
                          "throughput_ratio": row["request_per_second"] / baseline["request_per_second"],
                          "p95_latency_ratio": row["latency_p95_seconds"] / baseline["latency_p95_seconds"]})
    result = {"paired_measurements": pairs, "variants": list(observed),
              "scope": "Matched local workload and checkpoint; repeated-state feature cache, configured compute dtype.",
              "quality": {name: item.get("orchestration", []) for name, item in observed.items()},
              "limitations": "Sequential runs and finite repeats do not establish statistically significant or GPU gains."}
    write_csv(root / "paired_measurements.csv", pairs)
    write_json(root / "comparison.json", result)
    report = ["# Measured precision and feature-cache study", "", result["scope"], "", result["limitations"], "",
              "| Variant | Batch | Concurrency | Strategy | Throughput ratio | p95 ratio |",
              "| --- | ---: | ---: | --- | ---: | ---: |"]
    report.extend(f"| {row['optimization']} | {row['batch_size']} | {row['concurrency']} | {row['strategy']} | "
                  f"{row['throughput_ratio']:.3f} | {row['p95_latency_ratio']:.3f} |" for row in pairs)
    (root / "report.md").write_text("\n".join(report) + "\n")
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/optimization_study.yaml")
    args = parser.parse_args()
    asyncio.run(study(load_config(args.config)))


if __name__ == "__main__":
    main()
