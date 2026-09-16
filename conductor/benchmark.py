"""Synchronized controller microbenchmarks, including real queue-inclusive latencies."""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from conductor.evaluation.io import write_csv
from conductor.inference.batching import DynamicBatcher
from conductor.inference.timing import memory_measurements, reset_peak_memory, synchronize
from conductor.metrics.aggregate import percentile
from conductor.schema import ExecutionState
from conductor.utils.config import load_config
from conductor.utils.runs import Run, log_event, seed_everything, write_json


def make_state(context_size: int, index: int = 0) -> ExecutionState:
    # context_size is a requested word count; actual serialized tokens are measured separately.
    words = [f"context{i % 31}" for i in range(context_size)]
    return ExecutionState(user_task=f"Request {index}: " + " ".join(words), task_type="math")


def processed_token_counts(policy: Any, states: list[ExecutionState], k: int) -> tuple[list[int], str]:
    """Probe the same input path outside timing, including HF truncation/attention masks."""
    if hasattr(policy, "tokenizer") and callable(getattr(policy, "tokenize_states", None)):
        inputs = policy.tokenize_states(states)
        return inputs["attention_mask"].sum(-1).tolist(), policy.token_accounting
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
    latencies: list[float] = [0.0] * len(states)
    decisions: list[Any] = [None] * len(states)
    request_samples: list[dict[str, Any]] = [{} for _ in states]
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
            for index, decision in enumerate(routed, offset):
                decisions[index] = decision
                latencies[index] = finished - arrivals[index]
                request_samples[index] = {"request_index": index, "latency_seconds": latencies[index],
                                          "queue_seconds": None, "service_seconds": None,
                                          "actual_batch_size": len(routed), "semantics": "offline completion from common arrival"}
    else:
        batcher = DynamicBatcher(lambda items, top_k: batch_route(policy, items, top_k), batch_size, batch_wait)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conductor-benchmark")

        def individual(state: ExecutionState) -> tuple[Any, float, float]:
            began = time.perf_counter()
            synchronize(device)
            decision = policy.route(state, k)
            synchronize(device)
            return decision, began, time.perf_counter() - began

        async def request(index: int, state: ExecutionState) -> None:
            arrival = time.perf_counter()
            async with semaphore:
                admitted = time.perf_counter()
                if strategy == "dynamic":
                    measured = await batcher.submit_timed(state, k, timeout)
                    decision = measured["decision"]
                    queue_seconds = admitted - arrival + measured["queue_seconds"]
                    service_seconds, actual_batch_size = measured["service_seconds"], measured["actual_batch_size"]
                elif strategy == "single":
                    decision, began, service_seconds = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(executor, individual, state), timeout)
                    queue_seconds = began - arrival
                    actual_batch_size = 1
                else:
                    raise ValueError(f"Unknown benchmark strategy: {strategy}")
                # Reused decisions represent actual configured routing frequency, not fictitious controller forwards.
                for _ in range(routing_interval - 1):
                    _ = decision
                synchronize(device)
                decisions[index] = decision
                latencies[index] = time.perf_counter() - arrival
                request_samples[index] = {"request_index": index, "latency_seconds": latencies[index],
                                          "queue_seconds": queue_seconds, "service_seconds": service_seconds,
                                          "actual_batch_size": actual_batch_size, "semantics": "queue-inclusive service request"}
        try:
            await asyncio.gather(*(request(index, state) for index, state in enumerate(states)))
        finally:
            await batcher.close()
            executor.shutdown(wait=True, cancel_futures=True)
    synchronize(device)
    wall = time.perf_counter() - started
    return {"request_latencies_seconds": latencies, "wall_seconds": wall, "request_count": len(decisions),
            "request_samples": request_samples,
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
    dimensions = list(itertools.product(config.get("batch_sizes", [1, 4]),
                                   config.get("context_sizes", config.get("sequence_lengths", [32, 128])),
                                   config.get("concurrency", [1, 4]), config.get("k_values", [1, 2]),
                                   config.get("routing_intervals", [1, 2]), config.get("strategies", ["single", "batched", "dynamic"])))
    random.Random(config.get("seed", 42)).shuffle(dimensions)
    replay = None
    if config.get("replay_trajectories"):
        from conductor.inference.workloads import replay_states
        replay = replay_states(config["replay_trajectories"], request_count)
    from conductor.evaluation.provenance import canonical_hash, checkpoint_sha256
    for order, (batch_size, context_size, concurrency, k, interval, strategy) in enumerate(dimensions):
        if min(batch_size, context_size, concurrency, k, interval) < 1:
            raise ValueError("Benchmark dimensions must be positive")
        if strategy == "batched" and concurrency != config.get("concurrency", [1, 4])[0]:
            continue  # Offline batches have no request concurrency dimension.
        states = replay[0] if replay else [make_state(context_size, index) for index in range(request_count)]
        identities = replay[1] if replay else [{"request_id": f"synthetic-{index}",
                      "state_sha256": canonical_hash(state.to_dict())} for index, state in enumerate(states)]
        actual_requests = len(states)
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
            for request_index, sample in enumerate(measured["request_samples"]):
                requests.append({"batch_size": batch_size, "context_size": context_size,
                                 "concurrency": None if strategy == "batched" else concurrency,
                                 "k": k, "routing_interval": interval, "strategy": strategy,
                                 "configuration_order": order, "repeat": repeat,
                                 **sample, **identities[request_index]})
        tokens = sum(counts) / len(counts)
        total_requests = actual_requests * repeats
        rows.append({"batch_size": batch_size, "context_size_requested_words": context_size,
                     "serialized_input_tokens": tokens, "token_count_basis": token_basis,
                     "concurrency": None if strategy == "batched" else concurrency,
                     "concurrency_status": "not_applicable_offline_batch" if strategy == "batched" else "measured",
                     "k": k, "routing_interval": interval, "strategy": strategy,
                     "batch_route_implementation": "native" if callable(getattr(policy, "batch_route", None)) else "sequential_fallback",
                     "configuration_order": order, "warmup": warmup, "repeats": repeats, "request_count": total_requests,
                     "wall_seconds": wall, "request_per_second": total_requests / wall,
                     "input_tokens_per_second": total_requests * tokens / wall,
                     "latency_p50_seconds": percentile(samples, 50), "latency_p95_seconds": percentile(samples, 95),
                     "routing_calls_per_orchestration_step": 1 / interval,
                     "controller_overhead_fraction": controller_overhead, "controller_overhead_source": overhead_source,
                     **memory_measurements(getattr(policy, "device", None))})
        log_event("benchmark_configuration_completed", configuration_order=order,
                  strategy=strategy, batch_size=batch_size,
                  concurrency=None if strategy == "batched" else concurrency, k=k,
                  timed_requests=total_requests, wall_seconds=wall)
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
    from conductor.inference.profiling import profile_stages
    from conductor.inference.timing import nvml_measurements
    stage_states = (replay[0] if replay else [make_state(config.get("stage_context_size", 32), i)
                                            for i in range(config.get("stage_batch_size", 1))])[:config.get("stage_batch_size", 1)]
    stages = profile_stages(policy, stage_states, config.get("k_values", [2])[0], repeats=config.get("stage_repeats", 3),
                            nvtx=config.get("nvtx", False),
                            trace_path=run.output / "controller_trace.json" if config.get("profiler_trace", False) else None)
    write_json(run.output / "stages.json", stages)
    write_csv(run.output / "paired_optimizations.csv", optimization_pairs)
    result = {"benchmarks": rows, "serialization": serialization, "optimization_status": optimization_status,
              "paired_optimizations": optimization_pairs,
              "stage_profile": stages, "nvml": nvml_measurements(getattr(policy, "device", None), config.get("nvml", False)),
              "checkpoint_sha256": checkpoint_sha256(checkpoint or config.get("checkpoint")),
              "workload": replay[2] if replay else {"kind": "synthetic_context_microbenchmark"},
              "configuration_order": "seeded randomized configuration order; repeats grouped within each configuration",
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
