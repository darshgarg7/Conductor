import asyncio
import time

import pytest

from conductor.benchmark import measure_requests, processed_token_counts
from conductor.inference.batching import DynamicBatcher
from conductor.schema import ExecutionState, RoutingDecision


def test_dynamic_batching_reuses_worker_and_drains_on_close():
    batches = []
    def route(states, k):
        batches.append(len(states))
        return [state * k for state in states]
    async def scenario():
        batcher = DynamicBatcher(route, max_batch_size=4, max_wait_seconds=0.01)
        assert await asyncio.gather(*(batcher.submit(index, 2) for index in range(8))) == list(range(0, 16, 2))
        worker = batcher.worker
        assert await batcher.submit(9, 1) == 9
        assert batcher.worker is worker
        await batcher.close()
        assert worker.done()
        with pytest.raises(RuntimeError, match="closed"):
            await batcher.submit(1, 1)
    asyncio.run(scenario())
    assert max(batches) == 4


def test_batch_error_propagates_and_later_request_recovers():
    def route(states, k):
        if "bad" in states:
            raise ValueError("backend failed")
        return states
    async def scenario():
        async with DynamicBatcher(route, 2, 0.001) as batcher:
            with pytest.raises(ValueError, match="backend failed"):
                await batcher.submit("bad", 1)
            assert await batcher.submit("good", 1) == "good"
    asyncio.run(scenario())


def test_timeout_cancels_request_without_poisoning_worker():
    def route(states, k):
        time.sleep(0.02)
        return states
    async def scenario():
        async with DynamicBatcher(route, 2, 0.001) as batcher:
            with pytest.raises(asyncio.TimeoutError):
                await batcher.submit("slow", 1, timeout=0.003)
            assert await batcher.submit("next", 1, timeout=1) == "next"
    asyncio.run(scenario())


class SlowPolicy:
    def route(self, state, k):
        time.sleep(0.01)
        return RoutingDecision(["math"])


def test_request_latency_includes_admission_queue():
    result = asyncio.run(measure_requests(SlowPolicy(), [ExecutionState(str(i), "math") for i in range(3)],
                                        strategy="single", batch_size=1, concurrency=1, k=1, routing_interval=2,
                                        timeout=2, batch_wait=0.001))
    assert result["request_count"] == 3
    assert result["latency_p95_seconds"] >= 0.025
    assert result["routing_calls_per_orchestration_step"] == 0.5
    assert len(result["request_latencies_seconds"]) == 3


def test_token_counts_use_actual_backend_path_including_truncation():
    class TruncatedPolicy:
        token_accounting = "actual attention-mask tokens"
        def batch_route(self, states, k):
            self.last_batch_tokens = [4] * len(states)
            return [RoutingDecision(["math"]) for _ in states]
    counts, basis = processed_token_counts(TruncatedPolicy(), [ExecutionState("word " * 1000, "math")], 1)
    assert counts == [4]
    assert basis == "actual attention-mask tokens"
