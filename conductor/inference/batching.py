"""Reusable asynchronous batching with measured queue-inclusive request latency."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class _Request:
    state: Any
    k: int
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, batch_route: Callable, max_batch_size: int = 8, max_wait_seconds: float = 0.002) -> None:
        if max_batch_size < 1 or max_wait_seconds < 0:
            raise ValueError("Invalid batching limits")
        self.batch_route = batch_route
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_seconds
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker: asyncio.Task | None = None
        self.closed = False
        self.pending: set[asyncio.Future] = set()

    async def __aenter__(self) -> DynamicBatcher:
        self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def start(self) -> None:
        if self.closed:
            raise RuntimeError("Batcher is closed")
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())

    async def submit(self, state: Any, k: int, timeout: float | None = None) -> Any:
        self.start()
        future = asyncio.get_running_loop().create_future()
        self.pending.add(future)
        future.add_done_callback(self.pending.discard)
        await self.queue.put(_Request(state, k, future))
        return await asyncio.wait_for(future, timeout) if timeout is not None else await future

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.worker is not None:
            await self.queue.put(None)
            await self.worker

    async def _run(self) -> None:
        stopping = False
        try:
            while not stopping:
                first = await self.queue.get()
                if first is None:
                    break
                batch = [first]
                deadline = asyncio.get_running_loop().time() + self.max_wait_seconds
                while len(batch) < self.max_batch_size:
                    wait = deadline - asyncio.get_running_loop().time()
                    if wait <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), wait)
                    except asyncio.TimeoutError:
                        break
                    if item is None:
                        stopping = True
                        break
                    batch.append(item)
                groups: dict[int, list[_Request]] = defaultdict(list)
                for request in batch:
                    if not request.future.cancelled():
                        groups[request.k].append(request)
                for k, requests in groups.items():
                    try:
                        results = await asyncio.to_thread(self.batch_route, [request.state for request in requests], k)
                        if len(results) != len(requests):
                            raise RuntimeError("Batched controller returned the wrong number of decisions")
                    except Exception as error:
                        for request in requests:
                            if not request.future.done():
                                request.future.set_exception(error)
                    else:
                        for request, decision in zip(requests, results):
                            if not request.future.done():
                                request.future.set_result(decision)
        finally:
            for future in list(self.pending):
                if not future.done():
                    future.set_exception(RuntimeError("Batcher worker stopped"))
            while not self.queue.empty():
                request = self.queue.get_nowait()
                if request is not None and not request.future.done():
                    request.future.set_exception(RuntimeError("Batcher worker stopped"))
