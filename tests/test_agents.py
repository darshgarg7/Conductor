"""Specialists compute from public inputs; no hidden labels or arbitrary code."""
import asyncio
from dataclasses import FrozenInstanceError
import pytest

from conductor.agents import build_agents
from conductor.agents.deterministic import safe_arithmetic
from conductor.schema import ExecutionState


def test_frozen_specialist_solves_real_problem() -> None:
    agents = build_agents()
    state = ExecutionState("Compute 31 + 14.", "math")
    output = asyncio.run(agents["math"].execute(state))
    assert output.metadata["answer"] == "45"
    assert output.tokens > 0 and output.latency_seconds >= 0
    assert output.metadata["token_accounting"] == "estimated_whitespace"
    with pytest.raises(FrozenInstanceError):
        agents["math"].name = "coder"
    assert all(agent.frozen for agent in agents.values())


def test_dependency_requires_earlier_output() -> None:
    agents = build_agents()
    state = ExecutionState('Find Orion\'s value in reference, then multiply that value by 4. Reference: {"Orion": 7}', "composed_retrieval_math")
    assert asyncio.run(agents["math"].execute(state)).metadata["answer"] is None
    state.previous_agent_outputs.append(asyncio.run(agents["retriever"].execute(state)).to_dict())
    assert asyncio.run(agents["math"].execute(state)).metadata["answer"] == "28"


def test_tools_reject_arbitrary_python() -> None:
    assert safe_arithmetic("(4 + 2) * 3") == "18"
    for expression in ("__import__('os')", "2 ** 999999", "[1][0]", "True + 1"):
        with pytest.raises(ValueError):
            safe_arithmetic(expression)


def test_string_specialist_does_not_answer_math() -> None:
    output = asyncio.run(build_agents()["coder"].execute(ExecutionState("Compute 1 + 2.", "math")))
    assert output.metadata["answer"] is None


def test_structured_backend_output_distinguishes_unfinished_from_invalid() -> None:
    from conductor.agents.backends import unpack
    assert unpack('{"work":"Need earlier result", "answer":null}') == ("Need earlier result", None, False)
    assert unpack('{"work":"Calculated", "answer":"12"}') == ("Calculated", "12", False)
    for invalid in ('not json', '{}', '[1]', '{"work":"x", "answer":12}'):
        assert unpack(invalid)[2]


def test_hf_backend_pins_revision_and_shares_only_identical_source(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    from conductor.agents.backends import HFAgent
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos")
    model = Mock()
    model.parameters.return_value = []
    model_factory = Mock(return_value=model)
    tokenizer_factory = Mock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_factory),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_factory)))
    shared = {}
    settings = {"model_name": "mock/frozen", "revision": "commit-a"}
    first = HFAgent("math", settings, shared)
    second = HFAgent("coder", settings, shared)
    HFAgent("retriever", {**settings, "revision": "commit-b"}, shared)
    assert first.model is second.model and first.lock is second.lock
    assert model_factory.call_count == tokenizer_factory.call_count == 2
    assert {call.kwargs["revision"] for call in model_factory.call_args_list} == {"commit-a", "commit-b"}
    assert {call.kwargs["revision"] for call in tokenizer_factory.call_args_list} == {"commit-a", "commit-b"}
