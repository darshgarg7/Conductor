"""Admission control and dynamic batching around one exclusive model worker.

Timeouts cancel the response, not a running accelerator kernel. A hung CUDA
worker requires a process supervisor restart; readiness detects stalled work.
"""
from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from conductor.inference.timing import synchronize
from conductor.schema import AGENT_NAMES, ExecutionState


class Overloaded(RuntimeError):
    pass


class Unavailable(RuntimeError):
    pass


@dataclass
class Request:
    id: str
    state: ExecutionState
    k: int
    arrived: float
    deadline: float
    future: asyncio.Future


class RoutingEngine:
    def __init__(self, controller: Any, max_batch_size: int = 8, max_pending: int = 128,
                 batch_wait_seconds: float = 0.002, request_timeout: float = 30,
                 shutdown_timeout: float = 10) -> None:
        if (type(max_batch_size) is not int or type(max_pending) is not int or
                min(max_batch_size, max_pending) <= 0 or
                any(isinstance(value, bool) or not math.isfinite(value) or value <= 0
                    for value in (request_timeout, shutdown_timeout)) or
                isinstance(batch_wait_seconds, bool) or not math.isfinite(batch_wait_seconds) or batch_wait_seconds < 0):
            raise ValueError("invalid service batching/admission/deadline configuration")
        self.controller = controller
        self.max_batch_size, self.max_pending = max_batch_size, max_pending
        self.batch_wait, self.request_timeout, self.shutdown_timeout = batch_wait_seconds, request_timeout, shutdown_timeout
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conductor-model")
        self.worker: asyncio.Task | None = None
        self.pending: set[asyncio.Future] = set()
        self.accepting = False
        self.closed = False
        self.inflight_started: float | None = None
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies: deque[float] = deque(maxlen=1024)

    async def start(self, warmup: ExecutionState | None = None, k: int = 1) -> None:
        if self.closed or self.worker is not None:
            raise Unavailable("engine is already started or closed")
        if warmup is not None:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(self.executor, self._infer, [warmup], k),
                self.request_timeout,
            )
        self.worker = asyncio.create_task(self._run(), name="conductor-batcher")
        self.accepting = True

    def ready(self) -> bool:
        return bool(self.accepting and self.worker and not self.worker.done() and
                    (self.inflight_started is None or time.perf_counter() - self.inflight_started < self.request_timeout))

    async def submit(self, state: ExecutionState, k: int, timeout: float | None = None) -> dict[str, Any]:
        max_agents = getattr(getattr(self.controller, "catalog", None), "max_agents", len(AGENT_NAMES))
        if type(k) is not int or not 1 <= k <= max_agents:
            raise ValueError("k must be an integer supported by the controller action catalog")
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                                    not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("request timeout must be a positive finite number")
        if not self.ready():
            raise Unavailable("routing worker is unavailable or stalled")
        if len(self.pending) >= self.max_pending:
            self.counters["rejected_overload"] += 1
            raise Overloaded("routing admission capacity exhausted")
        duration = self.request_timeout if timeout is None else min(timeout, self.request_timeout)
        if duration <= 0:
            raise ValueError("request timeout must be positive")
        arrived = time.perf_counter()
        future = asyncio.get_running_loop().create_future()
        request = Request(str(uuid.uuid4()), copy.deepcopy(state), k, arrived, arrived + duration, future)
        self.pending.add(future)
        future.add_done_callback(self.pending.discard)
        try:
            self.queue.put_nowait(request)
        except asyncio.QueueFull as error:
            self.pending.discard(future)
            future.cancel()
            self.counters["rejected_overload"] += 1
            raise Overloaded("routing queue is full") from error
        self.counters["accepted"] += 1
        try:
            return await asyncio.wait_for(future, duration)
        except asyncio.TimeoutError:
            self.counters["timed_out"] += 1
            raise
        except asyncio.CancelledError:
            future.cancel()
            self.counters["cancelled"] += 1
            raise

    def _infer(self, states: list[ExecutionState], k: int) -> tuple[list[Any], list[int], list[float], float]:
        started = time.perf_counter()
        decisions = self.controller.batch_route(states, k)
        synchronize(getattr(self.controller, "device", None))
        seconds = time.perf_counter() - started
        if len(decisions) != len(states):
            raise RuntimeError("controller returned an invalid batch cardinality")
        tokens = list(getattr(self.controller, "last_batch_tokens", [0] * len(states)))
        costs = list(getattr(self.controller, "last_batch_costs", [0.0] * len(states)))
        if len(tokens) != len(states) or len(costs) != len(states):
            raise RuntimeError("controller telemetry has an invalid batch cardinality")
        if any(type(count) is not int or count < 0 for count in tokens):
            raise RuntimeError("controller token telemetry must contain nonnegative integers")
        if any(isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0
               for cost in costs):
            raise RuntimeError("controller cost telemetry must contain nonnegative finite numbers")
        for decision in decisions:
            decision.validate(k, AGENT_NAMES)
        return decisions, tokens, costs, seconds

    async def _run(self) -> None:
        stopping = False
        try:
            while not stopping:
                first = await self.queue.get()
                if first is None:
                    break
                batch = [first]
                deadline = time.perf_counter() + self.batch_wait
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                    if item is None:
                        stopping = True
                        break
                    batch.append(item)
                groups: dict[int, list[Request]] = defaultdict(list)
                for request in batch:
                    if not request.future.done() and request.deadline > time.perf_counter():
                        groups[request.k].append(request)
                    elif not request.future.done():
                        request.future.set_exception(asyncio.TimeoutError())
                for k, requests in groups.items():
                    # A previous k-group can exhaust deadlines while this group waits.
                    eligible = []
                    for request in requests:
                        if request.future.done():
                            continue
                        if request.deadline <= time.perf_counter():
                            request.future.set_exception(asyncio.TimeoutError())
                        else:
                            eligible.append(request)
                    requests = eligible
                    if not requests:
                        continue
                    started = time.perf_counter()
                    self.inflight_started = started
                    self.counters["model_batches"] += 1
                    try:
                        decisions, tokens, costs, seconds = await asyncio.get_running_loop().run_in_executor(
                            self.executor, self._infer, [request.state for request in requests], k)
                        finished = time.perf_counter()
                        for request, decision, count, cost in zip(requests, decisions, tokens, costs):
                            if request.future.done():
                                self.counters["completed_after_cancellation"] += 1
                                continue
                            if finished >= request.deadline:
                                request.future.set_exception(asyncio.TimeoutError())
                                continue
                            latency = finished - request.arrived
                            self.latencies.append(latency)
                            self.counters["completed"] += 1
                            request.future.set_result({"request_id": request.id, "decision": decision.to_dict(),
                                "controller_tokens": int(count), "estimated_cost_usd": float(cost),
                                "queue_seconds": started - request.arrived, "batch_inference_seconds": seconds,
                                "latency_seconds": latency, "batch_size": len(requests),
                                "token_accounting": getattr(self.controller, "token_accounting", "unavailable")})
                    except Exception as error:
                        self.counters["failed_batches"] += 1
                        for request in requests:
                            if not request.future.done():
                                request.future.set_exception(error)
                    finally:
                        self.inflight_started = None
        finally:
            self.accepting = False
            for future in list(self.pending):
                if not future.done():
                    future.set_exception(Unavailable("routing worker stopped"))

    async def close(self) -> None:
        if self.closed:
            return
        self.accepting = False
        self.closed = True
        try:
            if self.worker is not None:
                async def drain() -> None:
                    await self.queue.put(None)
                    await self.worker
                await asyncio.wait_for(drain(), self.shutdown_timeout)
        except asyncio.TimeoutError:
            self.counters["shutdown_timeout"] += 1
            if self.worker is not None:
                self.worker.cancel()
                await asyncio.gather(self.worker, return_exceptions=True)
        finally:
            for future in list(self.pending):
                if not future.done():
                    future.set_exception(Unavailable("service is shutting down"))
            self.pending.clear()
            while not self.queue.empty():
                self.queue.get_nowait()
            self.executor.shutdown(wait=False, cancel_futures=True)

    def metrics(self) -> dict[str, Any]:
        return {"ready": self.ready(), "pending": len(self.pending), "queue_depth": self.queue.qsize(),
                "counters": dict(self.counters), "recent_latency_count": len(self.latencies),
                "recent_latency_sum_seconds": sum(self.latencies)}
