"""Bounded specialist concurrency and per-call timeout configuration."""
from __future__ import annotations
import asyncio
import math
from typing import Any
from conductor.schema import AgentOutput, ExecutionState


class LimitedAgent:
    def __init__(self, agent: Any, settings: dict[str, Any], model_limit: asyncio.Semaphore) -> None:
        self.agent, self.config = agent, settings
        self.name, self.capability, self.frozen = agent.name, agent.capability, agent.frozen
        self.model = getattr(agent, "model", None)
        self.agent_limit = asyncio.Semaphore(int(settings.get("max_concurrency", 4)))
        self.model_limit = model_limit
        self._inflight: set[asyncio.Task[AgentOutput]] = set()
        self.timeout_seconds = float(settings.get("timeout_seconds", 60))
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or int(settings.get("max_concurrency", 4)) < 1:
            raise ValueError("agent timeouts/concurrency must be positive")

    def admission_tokens(self, state: ExecutionState) -> int | None:
        from conductor.agents.backends import prompt
        backend = self.config.get("backend", "deterministic")
        if backend == "deterministic":
            return None  # legacy estimated dev work is explicitly not a hard model bound
        if backend == "hf":
            text = prompt(self.name, state)
            if self.agent.tokenizer.chat_template:
                text = self.agent.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                               add_generation_prompt=True)
            actual = len(self.agent.tokenizer.encode(text))
            maximum = self.config.get("max_prompt_tokens")
            if maximum is not None and actual > int(maximum):
                return int(1e18)  # fail admission before model dispatch
            return actual + int(self.config.get("max_new_tokens", 128))
        # No whitespace estimate can prove a remote-model hard token bound.
        # Byte-BPE endpoints may supply a conservative byte bound plus special-token reserve.
        if self.config.get("tokenizer_family") == "byte_bpe":
            actual = len(prompt(self.name, state).encode()) + int(self.config.get("special_token_reserve", 64))
            return actual + int(self.config.get("max_new_tokens", 128))
        return None

    async def execute(self, state: ExecutionState) -> AgentOutput:
        started = False
        async def guarded() -> AgentOutput:
            nonlocal started
            async with self.agent_limit, self.model_limit:
                started = True
                return await self.agent.execute(state)
        invocation = asyncio.create_task(guarded())
        self._inflight.add(invocation)
        def finished(task: asyncio.Task[AgentOutput]) -> None:
            self._inflight.discard(task)
            if not task.cancelled():
                task.exception()  # Consume late failures after the caller timed out.
        invocation.add_done_callback(finished)
        try:
            # Deadline includes waiting for capacity. Once a thread-backed call
            # starts, retain semaphore ownership until its real work completes.
            return await asyncio.wait_for(asyncio.shield(invocation), self.timeout_seconds)
        except (TimeoutError, asyncio.CancelledError):
            if not started:
                invocation.cancel()  # Never dispatch an already timed-out queued call.
            raise
