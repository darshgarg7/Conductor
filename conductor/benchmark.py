"""Synchronized controller microbenchmarks, including real queue-inclusive latencies."""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
from pathlib import Path
from typing import Any

from conductor.evaluation.io import write_csv
from conductor.inference.batching import DynamicBatcher
from conductor.inference.timing import memory_measurements, reset_peak_memory, synchronize
from conductor.metrics.aggregate import percentile
from conductor.schema import ExecutionState
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json


def make_state(context_size: int, index: int = 0) -> ExecutionState:
    # context_size is a requested word count; actual serialized tokens are measured separately.
    words = [f"context{i % 31}" for i in range(context_size)]
    return ExecutionState(user_task=f"Request {index}: " + " ".join(words), task_type="math")


def processed_token_counts(policy: Any, states: list[ExecutionState], k: int) -> tuple[list[int], str]:
    """Probe the same input path outside timing, including HF truncation/attention masks."""
    batch_route(policy, states, k)
    counts = getattr(policy, "last_batch_tokens", None)
    if counts is not None and len(counts) == len(states):
        return [int(value) for value in counts], getattr(policy, "token_accounting", "backend_reported_input_tokens")
    from conductor.routing.serialization import serialize_state
    return [len(serialize_state(state).split()) for state in states], "whitespace_proxy; backend accounting unavailable"


def batch_route(policy: Any, states: list[ExecutionState], k: int) -> list[Any]:
    fn = getattr(policy, "batch_route", None)
    return fn(states, k) if callable(fn) else [policy.route(state, k) for state in states]


async def measure_requests(policy: Any, states: list[ExecutionState], *, strategy: str, batch_size: int,
                           concurrency: int, k: int, routing_interval: int, timeout: float,
                           batch_wait: float) -> dict[str, Any]:
    """A request contains routing_interval orchestration steps, with one actual routing call."""
    latencies: list[float] = []
    decisions: list[Any] = []
    semaphore = asyncio.Semaphore(concurrency)
    device = getattr(policy, "device", None)
    synchronize(device)
    started = time.perf_counter()
    if strategy == "batched":
        # Request arrival for an offline batch occurs before all queued batch service.
        arrivals = [time.perf_counter()] * len(states)
        for offset in range(0, len(states), batch_size):
            synchronize(device)
            routed = batch_route(policy, states[offset:offset + batch_size], k)
            synchronize(device)
            finished = time.perf_counter()
            decisions.extend(routed)
            latencies.extend(finished - arrivals[index] for index in range(offset, min(offset + batch_size, len(states))))
    else:
        batcher = DynamicBatcher(lambda items, top_k: batch_route(policy, items, top_k), batch_size, batch_wait)

        async def request(state: ExecutionState) -> None:
            arrival = time.perf_counter()
            async with semaphore:
                if strategy == "dynamic":
                    decision = await batcher.submit(state, k, timeout)
                elif strategy == "single":
                    decision = await asyncio.wait_for(asyncio.to_thread(policy.route, state, k), timeout)
                else:
                    raise ValueError(f"Unknown benchmark strategy: {strategy}")
                # Reused decisions represent actual configured routing frequency, not fictitious controller forwards.
                for _ in range(routing_interval - 1):
                    _ = decision
                synchronize(device)
                decisions.append(decision)
                latencies.append(time.perf_counter() - arrival)
        try:
            await asyncio.gather(*(request(state) for state in states))
        finally:
            await batcher.close()
    synchronize(device)
    wall = time.perf_counter() - started
    return {"request_latencies_seconds": latencies, "wall_seconds": wall, "request_count": len(decisions),
            "request_per_second": len(decisions) / wall, "routing_calls": len(decisions),
            "orchestration_steps": len(decisions) * routing_interval,
            "routing_calls_per_orchestration_step": 1 / routing_interval,
            "latency_p50_seconds": percentile(latencies, 50), "latency_p95_seconds": percentile(latencies, 95),
            "latency_semantics": "Per request, from arrival through queue/admission wait and synchronized completion"}


async def measure_orchestration(config: dict[str, Any], policy: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from conductor.agents import build_agents
    from conductor.datasets.io import read_jsonl
    from conductor.evaluation.tasks import heldout_tasks
    from conductor.orchestration.runner import run_trajectory

    source = Path(config.get("data", "data/dev/tasks.jsonl"))
    if not source.exists():
        return [], {"status": "unavailable", "reason": f"Heldout task data absent: {source}"}
    tasks, _ = heldout_tasks(read_jsonl(source), tuple(config.get("splits", ["eval", "test"])))
    representative = []
    for category in sorted({task.task_type for task in tasks}):
        representative.extend([task for task in tasks if task.task_type == category][:config.get("tasks_per_category", 1)])
    agents = build_agents(config)
    rows = []
    for k, interval in itertools.product(config.get("k_values", [1, 2]), config.get("routing_intervals", [1, 2])):
        seed_everything(config.get("seed", 42))
        for task in representative:
            synchronize(getattr(policy, "device", None))
            item = await run_trajectory(task, policy, agents, k=k, max_rounds=config.get("max_rounds", 3),
                                        token_budget=config.get("token_budget", 8192),
                                        agent_call_budget=config.get("agent_call_budget", 12), routing_interval=interval)
            synchronize(getattr(policy, "device", None))
            from conductor.metrics.aggregate import trajectory_metrics
            row = trajectory_metrics(item)
            row.update(k=k, routing_interval=interval)
            rows.append(row)
    return rows, {"status": "measured", "task_count": len(representative),
                  "scope": "Actual runner trajectories with fixed specialists and budgets; interval and k paired by task ID."}


def _controller_overhead(config: dict[str, Any]) -> tuple[float | None, str]:
    source = config.get("evaluation_metrics")
    if not source or not Path(source).exists():
        return None, "unavailable: an end-to-end measured evaluation was not supplied"
    result = json.loads(Path(source).read_text())
    policy_name = config.get("evaluation_policy", "conductor_preference")
    matching = [row for row in result.get("policies", []) if row.get("policy") == policy_name]
    if not matching:
        return None, f"unavailable: no measured {policy_name} policy in evaluation"
    return matching[0].get("mean_controller_overhead_fraction"), f"measured evaluation: {source}, {policy_name}"


def paired_optimization_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare dynamic vs individual routing only at identical measured workload dimensions."""
    dimensions = ("batch_size", "context_size_requested_words", "concurrency", "k", "routing_interval", "request_count")
    baseline = {tuple(row[key] for key in dimensions): row for row in rows if row["strategy"] == "single"}
    result = []
    for row in rows:
        key = tuple(row[field] for field in dimensions)
        if row["strategy"] != "dynamic" or key not in baseline:
            continue
        other = baseline[key]
        result.append({**dict(zip(dimensions, key)), "baseline": "single", "candidate": "dynamic", "status": "measured",
                       "request_throughput_ratio": row["request_per_second"] / other["request_per_second"],
                       "p95_latency_ratio": row["latency_p95_seconds"] / other["latency_p95_seconds"],
                       "interpretation": "Ratios are measured for this workload/checkpoint only; >1 means more throughput or more latency respectively."})
    return result


async def benchmark(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    from conductor.controller.factory import build_controller
    from conductor.routing.serialization import StateSerializer

    seed_everything(config.get("seed", 42))
    run = Run(config.get("output", "outputs/benchmarks/dev"), config, checkpoint)
    policy = build_controller(config, checkpoint=checkpoint or config.get("checkpoint"))
    supported = getattr(policy, "supported_optimizations", {})
    import torch
    forward_dtype = str(next(policy.model.parameters()).dtype) if hasattr(policy, "model") else None
    compute_dtype = getattr(policy, "dtype", None)
    optimization_status = {
        "mixed_precision": {"status": "supported" if supported.get("mixed_precision") else "unsupported",
                            "benchmarked": compute_dtype in {torch.float16, torch.bfloat16},
                            "actual_parameter_dtype": forward_dtype,
                            "actual_compute_precision": getattr(policy, "compute_precision", str(compute_dtype)),
                            "reason": "Only actual configured forward dtype is measured; supported does not imply enabled."},
        "hf_kv_cache": {"status": "supported" if supported.get("prefix_kv_cache") else "unsupported",
                        "benchmarked": False,
                        "reason": "Controller route is classification; generation cache requires a supporting HF backend."},
        "cuda_memory": {"status": "measured" if getattr(getattr(policy, "device", None), "type", None) == "cuda" else "unavailable",
                        "reason": "CUDA allocator measurements are separate from process RSS."},
    }
    rows, requests = [], []
    warmup, repeats = config.get("warmup", 2), config.get("repeats", 5)
    request_count = config.get("requests", 16)
    if warmup < 0 or repeats < 1 or request_count < 1:
        raise ValueError("warmup must be nonnegative, repeats and requests positive")
    controller_overhead, overhead_source = _controller_overhead(config)
    dimensions = itertools.product(config.get("batch_sizes", [1, 4]),
                                   config.get("context_sizes", config.get("sequence_lengths", [32, 128])),
                                   config.get("concurrency", [1, 4]), config.get("k_values", [1, 2]),
                                   config.get("routing_intervals", [1, 2]), config.get("strategies", ["single", "batched", "dynamic"]))
    for batch_size, context_size, concurrency, k, interval, strategy in dimensions:
        if min(batch_size, context_size, concurrency, k, interval) < 1:
            raise ValueError("Benchmark dimensions must be positive")
        if strategy == "batched" and concurrency != config.get("concurrency", [1, 4])[0]:
            continue  # Offline batches have no request concurrency dimension.
        states = [make_state(context_size, index) for index in range(request_count)]
        counts, token_basis = processed_token_counts(policy, states, k)
        for _ in range(warmup):
            await measure_requests(policy, states, strategy=strategy, batch_size=batch_size, concurrency=concurrency,
                                   k=k, routing_interval=interval, timeout=config.get("request_timeout", 30),
                                   batch_wait=config.get("batch_wait_seconds", 0.002))
        reset_peak_memory(getattr(policy, "device", None))
        samples, wall = [], 0.0
        for repeat in range(repeats):
            measured = await measure_requests(policy, states, strategy=strategy, batch_size=batch_size,
                                              concurrency=concurrency, k=k, routing_interval=interval,
                                              timeout=config.get("request_timeout", 30),
                                              batch_wait=config.get("batch_wait_seconds", 0.002))
            wall += measured["wall_seconds"]
            samples.extend(measured["request_latencies_seconds"])
            for request_index, latency in enumerate(measured["request_latencies_seconds"]):
                requests.append({"batch_size": batch_size, "context_size": context_size,
                                 "concurrency": None if strategy == "batched" else concurrency,
                                 "k": k, "routing_interval": interval, "strategy": strategy,
                                 "repeat": repeat, "request_index": request_index, "latency_seconds": latency})
        tokens = sum(counts) / len(counts)
        total_requests = request_count * repeats
        rows.append({"batch_size": batch_size, "context_size_requested_words": context_size,
                     "serialized_input_tokens": tokens, "token_count_basis": token_basis,
                     "concurrency": None if strategy == "batched" else concurrency,
                     "concurrency_status": "not_applicable_offline_batch" if strategy == "batched" else "measured",
                     "k": k, "routing_interval": interval, "strategy": strategy,
                     "batch_route_implementation": "native" if callable(getattr(policy, "batch_route", None)) else "sequential_fallback",
                     "warmup": warmup, "repeats": repeats, "request_count": total_requests,
                     "wall_seconds": wall, "request_per_second": total_requests / wall,
                     "input_tokens_per_second": total_requests * tokens / wall,
                     "latency_p50_seconds": percentile(samples, 50), "latency_p95_seconds": percentile(samples, 95),
                     "routing_calls_per_orchestration_step": 1 / interval,
                     "controller_overhead_fraction": controller_overhead, "controller_overhead_source": overhead_source,
                     **memory_measurements(getattr(policy, "device", None))})
    # Paired serialization microbenchmarks use identical states and repeat counts.
    serialization = []
    states = [make_state(max(config.get("context_sizes", [128])), index) for index in range(request_count)]
    for cache in (False, True):
        serializer = StateSerializer(cache_size=request_count * 2 if cache else 0)
        for state in states:
            serializer.serialize(state)
        synchronize(getattr(policy, "device", None))
        started = time.perf_counter()
        for _ in range(repeats):
            for state in states:
                serializer.serialize(state)
        synchronize(getattr(policy, "device", None))
        serialization.append({"cache_enabled": cache, "wall_seconds": time.perf_counter() - started,
                              "operations": request_count * repeats, "hits": serializer.hits, "misses": serializer.misses,
                              "scope": "Serialization microbenchmark; not an end-to-end inference speedup claim"})
    orchestration_rows, orchestration_status = await measure_orchestration(config, policy)
    from conductor.metrics.aggregate import aggregate_metrics
    orchestration_summary = aggregate_metrics(orchestration_rows, ("policy", "k", "routing_interval", "category"))
    write_csv(run.output / "orchestration_tasks.csv", orchestration_rows)
    write_csv(run.output / "orchestration_summary.csv", orchestration_summary)
    optimization_pairs = paired_optimization_ratios(rows)
    write_csv(run.output / "paired_optimizations.csv", optimization_pairs)
    result = {"benchmarks": rows, "serialization": serialization, "optimization_status": optimization_status,
              "paired_optimizations": optimization_pairs,
              "orchestration": orchestration_summary, "orchestration_tasks": orchestration_rows,
              "orchestration_status": orchestration_status,
              "scope": "Controller routing only; no downstream agent quality or generated-token throughput measured.",
              "latency_semantics": "Queue-inclusive per-request p50/p95 after explicit device synchronization.",
              "model_label": "Tiny-MoE-Init" if config.get("model", {}).get("backend", "tiny") == "tiny" and not checkpoint and not config.get("checkpoint") else "checkpoint/backend specified"}
    write_csv(run.output / "benchmark.csv", rows)
    write_csv(run.output / "requests.csv", requests)
    write_csv(run.output / "serialization.csv", serialization)
    write_json(run.output / "optimization_status.json", optimization_status)
    report = ["# Measured controller benchmarks", "", result["scope"], "", result["latency_semantics"], "",
              "Input tokens/second uses backend input accounting (tiny lexical proxy or actual truncated HF attention-mask counts), not generation.",
              "Microbenchmark routing interval reports reuse frequency; orchestration_tasks.csv records actual heldout trajectories for each interval and k.",
              "CPU RSS and CUDA allocated memory are reported separately. Unsupported optimizations are not enabled.",
              "Controller overhead: " + overhead_source + ".", "", "No improvement is claimed from unmatched configurations."]
    (run.output / "report.md").write_text("\n".join(report) + "\n")
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/dev.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    asyncio.run(benchmark(load_config(args.config), args.checkpoint))


if __name__ == "__main__":
    main()
