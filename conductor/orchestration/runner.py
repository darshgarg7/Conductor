"""Budgeted state transitions with immutable snapshots and measured execution time."""
from __future__ import annotations

import asyncio
import copy
import json
import time
import math
from dataclasses import asdict
from typing import Any

from conductor.agents.base import Agent, estimated_tokens
from conductor.agents.audit import specialist_audit
from conductor.schema import AgentOutput, ExecutionState, Policy, RoutingDecision, StepRecord, Task, Trajectory


def initial_state(task: Task, token_budget: int = 8192, agent_call_budget: int = 12) -> ExecutionState:
    # Private expected_answer and grading metadata are deliberately absent.
    return ExecutionState(task.user_task, task.task_type,
                          remaining_budget={"agent_calls": agent_call_budget, "tokens": token_budget})


def assemble_answer(state: ExecutionState) -> str:
    if state.task_type == "coordination_workflow":
        from conductor.coordination.grading import assemble_workflow_answer
        return assemble_workflow_answer(state)
    for output in reversed(state.previous_agent_outputs):
        answer = output.get("metadata", {}).get("answer")
        if answer is not None:
            return str(answer)
    return ""


def grade_answer(task: Task, answer: str) -> tuple[bool, float]:
    """Independent held-out exact grader; only this function reads private labels."""
    success = answer.strip() == task.expected_answer.strip()
    return success, float(success)


async def run_trajectory(task: Task, policy: Policy, agents: dict[str, Agent], k: int = 2,
                         max_rounds: int = 3, token_budget: int = 8192,
                         agent_call_budget: int = 12, routing_interval: int = 1,
                         initial_execution_state: ExecutionState | None = None,
                         hard_token_budget: bool = False, agent_timeout_seconds: float | None = None) -> Trajectory:
    if not agents or max_rounds < 1 or token_budget < 1 or agent_call_budget < 1 or routing_interval < 1:
        raise ValueError("positive rounds, budgets, interval, and available agents required")
    if any(not agent.frozen for agent in agents.values()):
        raise ValueError("specialists must remain frozen")
    if not 1 <= k <= len(agents):
        raise ValueError("k exceeds the available specialist set")
    if agent_timeout_seconds is not None and (not math.isfinite(agent_timeout_seconds) or agent_timeout_seconds <= 0):
        raise ValueError("agent_timeout_seconds must be a finite positive number")
    started = time.perf_counter()
    state = copy.deepcopy(initial_execution_state) if initial_execution_state is not None else initial_state(task, token_budget, agent_call_budget)
    if state.user_task != task.user_task or state.task_type != task.task_type:
        raise ValueError("resumed state must belong to the same public task")
    if set(state.to_dict()) != set(ExecutionState("", "").to_dict()):
        raise ValueError("private grading fields are forbidden in resumed states")
    initial_calls = len(state.agents_already_called)
    start_step = state.current_step if initial_execution_state is not None else 0
    original_tokens = float(state.remaining_budget["tokens"])
    audit = specialist_audit(agents)
    admission_rejections: list[dict[str, Any]] = []
    steps: list[StepRecord] = []
    graph: list[dict[str, Any]] = []
    requested_decisions: list[dict[str, Any]] = []
    clipped_steps: list[int] = []
    invalid_routes: list[dict[str, Any]] = []
    previous: RoutingDecision | None = None
    stop_reason = "max_rounds"
    agent_tokens = controller_tokens = 0
    cost = 0.0
    for step_index in range(start_step, start_step + max_rounds):
        if state.remaining_budget["agent_calls"] < 1 or state.remaining_budget["tokens"] < 1:
            stop_reason = "budget_exhausted"
            break
        state.current_step = step_index
        snapshot = copy.deepcopy(state.to_dict())
        reused = previous is not None and step_index % routing_interval != 0
        route_started = time.perf_counter()
        if reused:
            decision = copy.deepcopy(previous)
            tokens, controller_cost = 0, 0.0
        else:
            decision = policy.route(copy.deepcopy(state), k)
            tokens = int(getattr(policy, "last_tokens", 0))
            controller_cost = float(getattr(policy, "last_cost_usd", 0.0))
        controller_latency = time.perf_counter() - route_started
        if not reused and getattr(policy, "last_invalid", False):
            invalid_routes.append({"step": step_index, "error": getattr(policy, "last_error", "nonfinite routing probabilities")})
        is_all_agent = policy.name in {"all_agent", "All-Agent"}
        decision.validate(len(agents) if is_all_agent else k, tuple(agents))
        requested_decisions.append(copy.deepcopy(decision.to_dict()))
        previous = copy.deepcopy(decision)
        controller_tokens += tokens
        cost += controller_cost
        state.remaining_budget["tokens"] = max(0, state.remaining_budget["tokens"] - tokens)
        if decision.terminate:
            steps.append(StepRecord(snapshot, decision.to_dict(), [], tokens, controller_latency, controller_cost, reused))
            state.previous_routing_decisions.append(decision.to_dict())
            stop_reason = "controller_terminated"
            break
        # An indivisible model invocation can consume more tokens than available. We stop
        # future calls and record the overrun rather than fabricating truncated usage.
        selected = decision.selected_agents[:int(state.remaining_budget["agent_calls"])]
        if state.remaining_budget["tokens"] <= 0:
            selected = []
        outputs: list[dict[str, Any]] = []

        async def execute(name: str, supplied: ExecutionState) -> AgentOutput:
            call_started = time.perf_counter()
            try:
                invocation = agents[name].execute(supplied)
                return await asyncio.wait_for(invocation, agent_timeout_seconds) if agent_timeout_seconds else await invocation
            except Exception as error:
                return AgentOutput(name, f"Agent execution failed: {type(error).__name__}: {error}", 0,
                                   time.perf_counter() - call_started,
                                   metadata={"answer": None, "error": type(error).__name__, "backend": "failed",
                                             "token_usage_unknown": True, "cost_usage_unknown": True,
                                             "token_accounting": "unknown_after_execution_error"})

        def update(output: AgentOutput, supplied: ExecutionState) -> None:
            nonlocal agent_tokens, cost
            record = output.to_dict()
            outputs.append(record)
            state.previous_agent_outputs.append(copy.deepcopy(record))
            state.agents_already_called.append(output.agent)
            state.conversation_state.append({"role": output.agent, "content": output.content})
            state.remaining_budget["agent_calls"] -= 1
            state.remaining_budget["tokens"] = max(0, state.remaining_budget["tokens"] - output.tokens)
            agent_tokens += output.tokens
            cost += output.cost_usd
            if output.agent == "tool_executor":
                state.tool_results.append(copy.deepcopy(record))
            # These represent messages actually supplied and returned by this runner.
            serialized = json.dumps(supplied.to_dict(), sort_keys=True)
            graph.extend([
                {"source": "controller", "target": output.agent, "step": step_index,
                 "message_bytes": len(serialized.encode()), "estimated_message_tokens": estimated_tokens(serialized)},
                {"source": output.agent, "target": "controller", "step": step_index,
                 "message_bytes": len(output.content.encode()), "estimated_message_tokens": estimated_tokens(output.content)},
            ])
            for previous_output in supplied.previous_agent_outputs:
                if previous_output["agent"] == output.agent:
                    continue  # own history is context, not inter-agent communication
                graph.append({"source": previous_output["agent"], "target": output.agent, "step": step_index,
                              "message_bytes": len(previous_output["content"].encode()),
                              "estimated_message_tokens": estimated_tokens(previous_output["content"]),
                              "transport": "forwarded_in_state"})

        if hard_token_budget:
            admitted, reserved = [], 0
            for name in selected:
                estimate = getattr(agents[name], "admission_tokens", lambda supplied: None)(state)
                if estimate is None or estimate < 1 or reserved + estimate > state.remaining_budget["tokens"]:
                    admission_rejections.append({"step": step_index, "agent": name,
                                                 "reason": "unverifiable_token_bound" if estimate is None else "insufficient_reserved_tokens",
                                                 "reservation_tokens": estimate})
                else:
                    admitted.append(name)
                    if decision.execution_mode == "parallel":
                        reserved += estimate
            selected = admitted
        if decision.execution_mode == "parallel":
            supplied_states = [copy.deepcopy(state) for _ in selected]
            result = await asyncio.gather(*(execute(name, supplied) for name, supplied in zip(selected, supplied_states)))
            for output, supplied in zip(result, supplied_states):
                update(output, supplied)
        else:
            for name in selected:
                if state.remaining_budget["tokens"] < 1:
                    break
                supplied = copy.deepcopy(state)
                if hard_token_budget:
                    bound = agents[name].admission_tokens(supplied)
                    if bound is None or bound > state.remaining_budget["tokens"]:
                        admission_rejections.append({"step": step_index, "agent": name, "reason": "sequential_updated_state_bound"})
                        break
                update(await execute(name, supplied), supplied)
        actual_decision = decision.to_dict()
        activated = [output["agent"] for output in outputs]
        if activated != decision.selected_agents:
            clipped_steps.append(step_index)
            actual_decision["selected_agents"] = activated
        if not activated:
            actual_decision["terminate"] = True  # infrastructure stopped at the exhausted budget
        steps.append(StepRecord(snapshot, actual_decision, outputs, tokens, controller_latency, controller_cost, reused))
        state.previous_routing_decisions.append(copy.deepcopy(actual_decision))
        if not outputs:
            stop_reason = "budget_exhausted"
            break
    final_answer = assemble_answer(state)
    grading_details: dict[str, Any] = {}
    if task.task_type == "coordination_workflow":
        from conductor.coordination.grading import grade_workflow
        success, score, grading_details = grade_workflow(task, state, final_answer)
    else:
        success, score = grade_answer(task, final_answer)
    return Trajectory(asdict(task), policy.name, steps, final_answer, success, score,
                      time.perf_counter() - started, graph, cost,
                      metadata={"stop_reason": stop_reason, "controller_tokens": controller_tokens,
                                "workflow_grading": grading_details,
                                "downstream_agent_tokens": agent_tokens, "total_tokens": controller_tokens + agent_tokens,
                                "agent_activations": len(state.agents_already_called) - initial_calls, "coordination_rounds": len(steps),
                                "token_budget_overrun": max(0, controller_tokens + agent_tokens - original_tokens),
                                "token_accounting": "backend_specific; deterministic specialists use estimated whitespace",
                                "token_usage_known": not any(output.get("metadata", {}).get("token_usage_unknown", False)
                                                             for step in steps for output in step.agent_outputs),
                                "inference_cost_known": not any(output.get("metadata", {}).get("cost_usage_unknown", False)
                                                                for step in steps for output in step.agent_outputs),
                                "specialist_audit": audit, "specialist_fingerprint": audit["fingerprint"],
                                "resumed_from_step": start_step, "hard_token_budget": hard_token_budget,
                                "admission_rejections": admission_rejections,
                                "k": k, "routing_interval": routing_interval,
                                "requested_routing_decisions": requested_decisions,
                                "invalid_routing_decisions": invalid_routes,
                                "budget_clipped_steps": clipped_steps,
                                "controller_agent_messages": sum(edge.get("transport") != "forwarded_in_state" for edge in graph),
                                "logical_inter_agent_edges": sum(edge.get("transport") == "forwarded_in_state" for edge in graph),
                                "serialized_state_and_output_bytes": sum(edge["message_bytes"] for edge in graph
                                                                           if edge.get("transport") != "forwarded_in_state"),
                                "communication_graph_semantics": "controller request/response state-message edges plus logical dependency forwarding edges; forwarding bytes overlap request payload, not separate physical requests; self-history excluded"})
