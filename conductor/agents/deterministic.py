"""Cheap, inspectable specialists for pipeline validation, not LLM research results."""
from __future__ import annotations

import ast
import asyncio
import json
import operator
import re
import time
from dataclasses import dataclass
from typing import Any

from conductor.agents.base import CAPABILITIES, estimated_tokens
from conductor.schema import AgentOutput, ExecutionState

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def safe_arithmetic(expression: str) -> str:
    """Evaluate a bounded arithmetic AST with no names, calls, or exponentiation."""
    if len(expression) > 128:
        raise ValueError("arithmetic expression exceeds limit")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 48:
        raise ValueError("arithmetic expression exceeds node limit")

    def evaluate(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            value = OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        else:
            raise ValueError("only bounded arithmetic is allowed")
        if abs(value) > 1e12:
            raise ValueError("arithmetic result exceeds limit")
        return value

    answer = evaluate(tree.body)
    return str(int(answer)) if answer == int(answer) else format(answer, ".10g")


def previous_answer(state: ExecutionState, names: set[str] | None = None) -> str | None:
    for output in reversed(state.previous_agent_outputs):
        answer = output.get("metadata", {}).get("answer")
        if answer is not None and (names is None or output["agent"] in names):
            return str(answer)
    return None


def reference_answer(text: str) -> str | None:
    reference = re.search(r"Reference:\s*(\{[^\n]+\})", text)
    requested = re.search(r"(?:Find|What is) ([A-Za-z]+)'s value", text)
    if reference and requested:
        facts = json.loads(reference.group(1))
        value = facts.get(requested.group(1))
        return str(value) if value is not None else None
    return None


def compute_answer(text: str, state: ExecutionState) -> str | None:
    expression = re.search(r"Compute ([0-9+*/().% \-]+)\.", text)
    if expression:
        return safe_arithmetic(expression.group(1))
    multiplier = re.search(r"multiply that value by (\d+)", text)
    if multiplier:
        value = previous_answer(state, {"retriever", "researcher"})
        return safe_arithmetic(f"{value} * {multiplier.group(1)}") if value is not None else None
    return None


def string_answer(text: str, state: ExecutionState) -> str | None:
    reverse = re.search(r'Reverse the text "([A-Za-z ]+)"', text)
    vowels = re.search(r'Count vowels in the text "([A-Za-z ]+)"', text)
    if reverse:
        return reverse.group(1)[::-1]
    if vowels:
        return str(sum(letter.lower() in "aeiou" for letter in vowels.group(1)))
    if "then reverse the digits of the result" in text:
        number = previous_answer(state, {"math", "tool_executor"})
        return number[::-1] if number is not None else None
    return None


@dataclass(frozen=True)
class DeterministicAgent:
    name: str
    token_price_per_million: float = 0.0
    frozen: bool = True

    @property
    def capability(self) -> str:
        return CAPABILITIES[self.name]

    async def execute(self, state: ExecutionState) -> AgentOutput:
        started = time.perf_counter()
        # Yield once so parallel orchestration exercises actual asynchronous scheduling.
        await asyncio.sleep(0)
        answer: str | None = None
        if self.name in {"retriever", "researcher"}:
            answer = reference_answer(state.user_task)
            content = f"Reference supports value {answer}." if answer is not None else "No requested reference fact found."
        elif self.name == "math":
            answer = compute_answer(state.user_task, state)
            content = f"Arithmetic result: {answer}." if answer is not None else "Need an arithmetic expression or retrieved value."
        elif self.name == "coder":
            answer = string_answer(state.user_task, state)
            content = f"String operation result: {answer}." if answer is not None else "Need string input or an earlier arithmetic result."
        elif self.name == "tool_executor":
            answer = string_answer(state.user_task, state) or compute_answer(state.user_task, state)
            content = f"Allow-listed tool result: {answer}." if answer is not None else "No allow-listed tool input available."
        elif self.name == "planner":
            content = "Plan: retrieve supplied facts when needed; calculate numeric expressions; transform strings; verify the resulting answer."
        else:
            candidate = previous_answer(state)
            if candidate is None:
                content = "No answer available to inspect."
            else:
                checks: list[str] = []
                ref = reference_answer(state.user_task)
                numeric = compute_answer(state.user_task, state)
                string = string_answer(state.user_task, state)
                if "multiply that value" in state.user_task:
                    checks = [numeric] if numeric is not None else []
                elif "then reverse the digits" in state.user_task:
                    checks = [string] if string is not None else []
                else:
                    checks = [value for value in (ref, numeric, string) if value is not None]
                valid = candidate in checks
                answer = candidate if valid else None
                content = f"Candidate {'verified' if valid else 'does not match the public specification'}: {candidate}."
        metadata: dict[str, Any] = {"answer": answer, "token_accounting": "estimated_whitespace", "backend": "deterministic"}
        tokens = estimated_tokens(json.dumps(state.to_dict(), sort_keys=True)) + estimated_tokens(content)
        return AgentOutput(self.name, content, tokens, time.perf_counter() - started,
                           tokens * self.token_price_per_million / 1e6, metadata)
