import pytest
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.routing.policies import AllAgentPolicy, RandomTopKPolicy, RuleBasedPolicy, build_policy
from conductor.routing.serialization import StateSerializer


@pytest.mark.parametrize("k", [1, 2, 3])
def test_sparse_baselines_and_termination(k: int) -> None:
    state = ExecutionState("Compute 2 + 3", "arithmetic")
    assert RuleBasedPolicy().route(state, k).selected_agents == ["math"]
    decision = RandomTopKPolicy(4).route(state, k)
    assert len(decision.selected_agents) == k
    decision.validate(k)
    state.agents_already_called = ["math"]
    assert RuleBasedPolicy().route(state, k).terminate


def test_all_agent_is_dense_comparator() -> None:
    assert AllAgentPolicy().route(ExecutionState("task", "lookup"), 1).selected_agents == list(AGENT_NAMES)


@pytest.mark.parametrize("decision", [RoutingDecision(["alien"]), RoutingDecision(["math", "math"]),
    RoutingDecision(["math"], terminate=True), RoutingDecision([]), RoutingDecision(["math"], "broken"),
    RoutingDecision([], terminate="false"), RoutingDecision(["math"], confidence=float("nan"))])
def test_invalid_routes_rejected(decision: RoutingDecision) -> None:
    with pytest.raises(ValueError):
        decision.validate(2)


def test_cache_content_tracks_mutation() -> None:
    cache = StateSerializer(1)
    state = ExecutionState("task", "lookup")
    first = cache.serialize(state)
    assert cache.serialize(state) == first and cache.hits == 1
    state.current_step = 1
    assert cache.serialize(state) != first and cache.misses == 2


def test_supervisor_requires_model() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        build_policy("static_supervisor", {})


def test_baselines_obey_configured_specialist_set() -> None:
    config = {"agents": {"names": ["math", "coder"]}}
    state = ExecutionState("Compute 1 + 2.", "math")
    for name in ("all_agent", "random_top_k", "rule_based"):
        assert set(build_policy(name, config).route(state, 2).selected_agents) <= {"math", "coder"}


def test_checkpoint_cannot_be_mislabelled_as_another_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from conductor.controller import factory
    monkeypatch.setattr(factory, "build_controller", lambda config, checkpoint: SimpleNamespace(stage="sft"))
    with pytest.raises(ValueError, match="preference checkpoint"):
        build_policy("conductor_preference", {}, checkpoint="sft")
    with pytest.raises(ValueError, match="without a trained checkpoint"):
        build_policy("base_moe", {}, checkpoint="sft")
