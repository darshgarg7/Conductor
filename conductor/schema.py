"""Backend-independent, versioned records for coordination experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

AGENT_NAMES = ("planner", "retriever", "researcher", "coder", "tool_executor", "critic", "verifier", "math")


@dataclass
class Task:
    id: str
    user_task: str
    task_type: str
    split: str = "train"
    expected_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    user_task: str
    task_type: str
    conversation_state: list[dict[str, str]] = field(default_factory=list)
    previous_agent_outputs: list[dict[str, Any]] = field(default_factory=list)
    agents_already_called: list[str] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    remaining_budget: dict[str, float] = field(default_factory=lambda: {"agent_calls": 12, "tokens": 8192})
    current_step: int = 0
    previous_routing_decisions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingDecision:
    selected_agents: list[str]
    execution_mode: str = "parallel"
    confidence: float = 1.0
    terminate: bool = False

    def validate(self, k: int, available: tuple[str, ...] = AGENT_NAMES) -> RoutingDecision:
        if not isinstance(self.selected_agents, list) or any(not isinstance(name, str) for name in self.selected_agents):
            raise ValueError("selected_agents must be a list of strings")
        if type(self.terminate) is not bool:
            raise ValueError("terminate must be a boolean")
        if not isinstance(self.confidence, (int, float)) or isinstance(self.confidence, bool):
            raise ValueError("confidence must be a number")
        if k < 1 or k > len(available):
            raise ValueError("k must be between 1 and the number of available agents")
        if self.execution_mode not in {"parallel", "sequential"}:
            raise ValueError("execution_mode must be parallel or sequential")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if len(set(self.selected_agents)) != len(self.selected_agents):
            raise ValueError("duplicate agents")
        if any(name not in available for name in self.selected_agents):
            raise ValueError("unknown agent")
        if len(self.selected_agents) > k:
            raise ValueError("routing decision exceeds k")
        if self.terminate and self.selected_agents:
            raise ValueError("termination cannot activate agents")
        if not self.terminate and not self.selected_agents:
            raise ValueError("non-terminal decision must activate an agent")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentOutput:
    agent: str
    content: str
    tokens: int
    latency_seconds: float
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepRecord:
    state: dict[str, Any]
    decision: dict[str, Any]
    agent_outputs: list[dict[str, Any]]
    controller_tokens: int = 0
    controller_latency_seconds: float = 0.0
    controller_cost_usd: float = 0.0
    routing_reused: bool = False


@dataclass
class Trajectory:
    task: dict[str, Any]
    policy: str
    steps: list[StepRecord]
    final_answer: str
    task_success: bool
    grader_score: float
    wall_clock_latency: float
    communication_graph: list[dict[str, Any]] = field(default_factory=list)
    estimated_inference_cost: float = 0.0
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Trajectory:
        if record.get("schema_version", 1) != 1:
            raise ValueError("unsupported trajectory schema")
        value = dict(record)
        value["steps"] = [StepRecord(**step) for step in value["steps"]]
        return cls(**value)


class Policy(Protocol):
    name: str
    last_tokens: int
    last_cost_usd: float

    def route(self, state: ExecutionState, k: int) -> RoutingDecision: ...
