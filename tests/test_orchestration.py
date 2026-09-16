import asyncio
from dataclasses import asdict
from conductor.agents import build_agents
from conductor.orchestration.runner import run_trajectory
from conductor.routing.policies import AllAgentPolicy, RuleBasedPolicy
from conductor.schema import ExecutionState, RoutingDecision, Task


class FixedPolicy:
    name = "fixed"
    last_tokens = 0
    last_cost_usd = 0.0
    def __init__(self, names, mode="sequential"):
        self.names, self.mode = names, mode
    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        if state.current_step:
            return RoutingDecision([], terminate=True)
        return RoutingDecision(self.names, self.mode)


def test_sequential_outputs_are_visible_parallel_outputs_are_not() -> None:
    task = Task("dependency", 'Find Orion\'s value in reference, then multiply that value by 4. Reference: {"Orion": 7}', "composed_retrieval_math", expected_answer="28")
    sequential = asyncio.run(run_trajectory(task, FixedPolicy(["retriever", "math"]), build_agents()))
    parallel = asyncio.run(run_trajectory(task, FixedPolicy(["retriever", "math"], "parallel"), build_agents()))
    assert sequential.task_success and not parallel.task_success
    assert sequential.steps[0].state["previous_agent_outputs"] == []
    assert len(sequential.steps[0].agent_outputs) == 2
    assert len(sequential.steps[1].state["previous_agent_outputs"]) == 2
    assert len(sequential.communication_graph) > len(parallel.communication_graph)


def test_budget_enforced_and_terminal_round_preserved() -> None:
    task = Task("budget", "Compute 4 + 3.", "math", expected_answer="7")
    trajectory = asyncio.run(run_trajectory(task, AllAgentPolicy(), build_agents(), k=1, agent_call_budget=2))
    assert sum(len(step.agent_outputs) for step in trajectory.steps) == 2
    assert trajectory.metadata["stop_reason"] == "budget_exhausted"
    assert len(trajectory.metadata["requested_routing_decisions"][0]["selected_agents"]) == 8
    assert trajectory.steps[0].decision["selected_agents"] == ["planner", "retriever"]
    assert trajectory.metadata["budget_clipped_steps"] == [0]
    terminated = asyncio.run(run_trajectory(task, RuleBasedPolicy(), build_agents(), k=1))
    assert terminated.task_success
    assert terminated.steps[-1].decision["terminate"]
    assert terminated.steps[-1].agent_outputs == []
    assert terminated.wall_clock_latency > 0


def test_calling_agent_cannot_see_private_grader_information() -> None:
    seen = []
    class InspectingPolicy(FixedPolicy):
        def route(self, state, k):
            seen.append(asdict(state))
            return super().route(state, k)
    task = Task("private", "Compute 2 + 3.", "math", expected_answer="WRONG PRIVATE LABEL", metadata={"secret": "answer"})
    trajectory = asyncio.run(run_trajectory(task, InspectingPolicy(["math"]), build_agents()))
    assert trajectory.final_answer == "5" and not trajectory.task_success
    assert all("expected_answer" not in state and "metadata" not in state for state in seen)


def test_reduced_routing_frequency_is_recorded() -> None:
    task = Task("reuse", "Compute 4 + 5.", "math", expected_answer="9")
    trajectory = asyncio.run(run_trajectory(task, FixedPolicy(["math"]), build_agents(), routing_interval=2))
    assert trajectory.steps[1].routing_reused
    assert trajectory.steps[1].controller_tokens == 0


def test_indivisible_token_overrun_is_reported_without_clamping_usage() -> None:
    task = Task("tokens", "Compute 4 + 3.", "math", expected_answer="7")
    trajectory = asyncio.run(run_trajectory(task, FixedPolicy(["math", "verifier"]), build_agents(), token_budget=1))
    assert len(trajectory.steps[0].agent_outputs) == 1
    actual = trajectory.steps[0].agent_outputs[0]["tokens"]
    assert actual > 1
    assert trajectory.metadata["total_tokens"] == actual
    assert trajectory.metadata["token_budget_overrun"] == actual - 1
    assert trajectory.steps[0].decision["selected_agents"] == ["math"]


def test_backend_failure_marks_usage_unknown_instead_of_free_inference() -> None:
    class FailingAgent:
        name, capability, frozen = "math", "arithmetic", True
        async def execute(self, state):
            raise RuntimeError("endpoint failed after request")
    task = Task("unknown", "Compute 4 + 3.", "math", expected_answer="7")
    trajectory = asyncio.run(run_trajectory(task, FixedPolicy(["math"]), {"math": FailingAgent()}, k=1))
    error = trajectory.steps[0].agent_outputs[0]
    assert error["metadata"]["token_usage_unknown"]
    assert not trajectory.metadata["token_usage_known"]
    assert not trajectory.metadata["inference_cost_known"]
    assert not trajectory.task_success
