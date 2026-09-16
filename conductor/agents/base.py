"""Frozen downstream agent interfaces and honest usage accounting."""
from __future__ import annotations

from typing import Protocol
from conductor.schema import AgentOutput, ExecutionState

CAPABILITIES = {
    "planner": "Decompose a task and suggest an execution plan; does not produce the answer.",
    "retriever": "Extract requested facts from reference material supplied with the task.",
    "researcher": "Inspect reference material and synthesize a supported factual answer.",
    "coder": "Perform bounded string transformations; no execution of generated programs.",
    "tool_executor": "Execute an allow-listed arithmetic or string tool, never arbitrary code.",
    "critic": "Identify missing evidence or contradictions in previous specialist answers.",
    "verifier": "Check existing answers against the public problem specification.",
    "math": "Calculate arithmetic expressions, including quantities supplied by other agents.",
}


class Agent(Protocol):
    name: str
    capability: str
    frozen: bool

    async def execute(self, state: ExecutionState) -> AgentOutput: ...


def estimated_tokens(text: str) -> int:
    """Development-only whitespace token estimate, not a model tokenizer measurement."""
    return len(text.split())
