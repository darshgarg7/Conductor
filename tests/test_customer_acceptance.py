"""HTTP acceptance/accounting and real state-feedback integration without weights."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")
import httpx
from fastapi.testclient import TestClient

from conductor.demo import (HTTPModelPolicy, illustrative_objectives, load_phase, main,
                            service_contract_checks, summarize_trajectories, validated_model_response)
from conductor.orchestration.runner import run_trajectory
from conductor.schema import AgentOutput, ExecutionState, RoutingDecision, Task
from conductor.serving.api import create_app


def model_response(**changes: object) -> dict:
    value = {"request_id": str(uuid4()), "decision": RoutingDecision(["retriever"]).to_dict(),
             "decision_source": "model", "checkpoint_stage": "sft", "controller_tokens": 11,
             "estimated_cost_usd": .003, "queue_seconds": .01, "batch_inference_seconds": .02,
             "latency_seconds": .03, "batch_size": 1, "effective_k": 2, "token_accounting": "test fixture"}
    return {**value, **changes}


class FeedbackController:
    stage = "sft"
    catalog = SimpleNamespace(max_agents=3)
    token_accounting = "test fixture tokens; not model performance evidence"

    def __init__(self) -> None:
        self.states: list[ExecutionState] = []

    def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
        self.states.extend(states)
        self.last_batch_tokens = [11 + len(state.previous_agent_outputs) for state in states]
        self.last_batch_costs = [.003] * len(states)
        return [RoutingDecision([], terminate=True) if "researcher" in state.agents_already_called else
                RoutingDecision(["researcher"] if "retriever" in state.agents_already_called else ["retriever"])
                for state in states]


class FixedAgent:
    frozen = True
    capability = "fixed public-evidence support test fixture"

    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, state: ExecutionState) -> AgentOutput:
        answer = "grounded diagnosis" if self.name == "researcher" and state.previous_agent_outputs else None
        return AgentOutput(self.name, "public observation/runbook", 2, .0001,
                           metadata={"answer": answer, "backend": "deterministic", "commands_executed": False})


def config() -> dict:
    return {"serving": {"require_api_key": True, "api_key_env": "ACCEPTANCE_TEST_KEY", "warmup": False,
                        "expected_stage": "sft", "batch_wait_seconds": .002, "max_batch_size": 4}}


def test_real_asgi_routes_control_agents_and_receive_updated_public_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCEPTANCE_TEST_KEY", "ephemeral-test-key")
    controller = FeedbackController()
    with TestClient(create_app(config(), controller)) as client:
        policy = HTTPModelPolicy(client, "ephemeral-test-key", "sft")
        policy.ticket_id = "ticket-fixture"
        task = Task("ticket-fixture", "Observed public symptom", "support", expected_answer="grounded diagnosis")
        trajectory = asyncio.run(run_trajectory(task, policy, {name: FixedAgent(name) for name in
                                                ("retriever", "researcher")}, k=2, max_rounds=4))
        assert trajectory.task_success
        assert [state.agents_already_called for state in controller.states] == [[], ["retriever"], ["retriever", "researcher"]]
        assert controller.states[1].previous_agent_outputs[0]["content"] == "public observation/runbook"
        assert all("expected_answer" not in state.to_dict() for state in controller.states)
        assert trajectory.metadata["controller_tokens"] == 36
        assert trajectory.metadata["agent_activations"] == 2
        assert trajectory.estimated_inference_cost == pytest.approx(.009)
        assert len(policy.records) == 3
        assert len({row["request_id"] for row in policy.records}) == 3
        assert all(row["ticket_id"] == task.id and row["client_latency_seconds"] >= 0 for row in policy.records)
        trajectory.metadata["controller_token_accounting"] = controller.token_accounting
        summary = summarize_trajectories([trajectory.to_dict()])[0]
        assert summary["task_count"] == 1 and summary["success_rate"] == 1
        assert summary["mean_controller_tokens"] == 36
        assert summary["mean_downstream_tokens"] == 4
        assert summary["mean_agent_calls"] == 2
        assert summary["mean_routing_calls"] == 3


def test_service_contract_probes_do_not_dispatch_the_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCEPTANCE_TEST_KEY", "ephemeral-test-key")
    controller = FeedbackController()
    with TestClient(create_app(config(), controller)) as client:
        result = service_contract_checks(client, "ephemeral-test-key", 2)
        assert result["passed"]
        assert result["http_statuses"] == {"unauthorized": 401, "invalid_state": 422,
                                           "private_answer_field": 422, "zero_budget": 200}
        assert controller.states == []


def test_failed_http_route_clears_old_telemetry_and_records_unknown_usage() -> None:
    calls = 0
    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert "expected_answer" not in json.loads(request.content)["state"]
        return httpx.Response(200, json=model_response()) if calls == 1 else httpx.Response(504)
    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(handle)) as client:
        policy = HTTPModelPolicy(client, "key", "sft")
        assert not policy.route(ExecutionState("public symptom", "support"), 2).terminate
        assert policy.last_tokens == 11
        assert policy.route(ExecutionState("public symptom", "support"), 2).terminate
        assert policy.last_invalid and policy.last_tokens == 0 and policy.last_cost_usd == 0
        assert policy.records[-1]["http_status"] == 504
        assert policy.records[-1]["controller_usage_known"] is False
        assert "controller_tokens" not in policy.records[-1]


@pytest.mark.parametrize("changes", [
    {"decision_source": "budget_guard"}, {"checkpoint_stage": "preference"}, {"controller_tokens": True},
    {"queue_seconds": float("nan")}, {"effective_k": 3}, {"request_id": "not-uuid"},
    {"latency_seconds": .001}, {"batch_size": 0},
    {"decision": RoutingDecision(["retriever", "researcher", "math"]).to_dict()},
])
def test_model_response_contract_rejects_false_provenance_or_invalid_telemetry(changes: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        validated_model_response(model_response(**changes), 2, "sft")


def test_duplicate_request_id_is_a_failed_route() -> None:
    value = model_response()
    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=value))) as client:
        policy = HTTPModelPolicy(client, "key", "sft")
        assert not policy.route(ExecutionState("public", "support"), 2).terminate
        assert policy.route(ExecutionState("public", "support"), 2).terminate
        assert policy.last_invalid and policy.records[-1]["status"] == "failed"


@pytest.mark.parametrize("body", [[], model_response(request_id=42), model_response(request_id=["malformed"])])
def test_malformed_body_or_uuid_stops_safely_and_preserves_only_validated_usage(body: object) -> None:
    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=body))) as client:
        policy = HTTPModelPolicy(client, "key", "sft")
        assert policy.route(ExecutionState("public", "support"), 2).terminate
        assert policy.last_invalid and policy.records[0]["status"] == "failed"
        assert policy.records[0]["controller_usage_known"] is False
        if isinstance(body, dict):
            assert policy.last_tokens == policy.records[0]["controller_tokens"] == 11
            assert policy.last_cost_usd == pytest.approx(.003)
        else:
            assert policy.last_tokens == 0 and "controller_tokens" not in policy.records[0]


def test_untrusted_or_invalid_telemetry_is_not_counted() -> None:
    for body in (model_response(checkpoint_stage="preference"),
                 model_response(request_id=42, controller_tokens=True, estimated_cost_usd=-1)):
        with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=body))) as client:
            policy = HTTPModelPolicy(client, "key", "sft")
            assert policy.route(ExecutionState("public", "support"), 2).terminate
            assert policy.last_tokens == 0 and policy.last_cost_usd == 0
            assert "controller_tokens" not in policy.records[0]


def test_boundary_checks_require_successful_metrics_endpoint() -> None:
    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text=""))) as client:
        with pytest.raises(httpx.HTTPStatusError):
            service_contract_checks(client, "key", 2)


def test_closed_loop_load_dispatches_real_asgi_worker_and_records_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCEPTANCE_TEST_KEY", "ephemeral-test-key")
    async def exercise() -> None:
        controller = FeedbackController()
        app = create_app(config(), controller)
        async with app.router.lifespan_context(app):
            summaries = []
            for concurrency in (1, 4):
                summary, rows = await load_phase("http://fixture", "ephemeral-test-key", [ExecutionState("public", "support")],
                    k=2, stage="sft", concurrency=concurrency, request_count=12, timeout_seconds=2,
                    max_phase_seconds=4, transport=httpx.ASGITransport(app=app))
                assert summary["successful_requests"] == summary["requests"] == len(rows) == 12
                assert summary["failures"] == 0 and summary["completion_requests_per_second"] > 0
                assert summary["p95_latency_seconds"] >= summary["p50_latency_seconds"] >= 0
                assert sum(row["controller_tokens"] for row in rows) == summary["controller_tokens_known"] == 132
                assert len({row["request_id"] for row in rows}) == 12
                assert all(row["queue_seconds"] >= 0 and row["client_latency_seconds"] >= 0 for row in rows)
                summaries.append(summary)
            assert len(controller.states) == 24
            assert app.state.engine.metrics()["pending"] == 0
    asyncio.run(exercise())


def test_load_failures_and_deadline_remain_visible() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(.03)
        return httpx.Response(503)
    summary, rows = asyncio.run(load_phase("http://fixture", "key", [ExecutionState("public", "support")],
        k=2, stage="sft", concurrency=1, request_count=8, timeout_seconds=1, max_phase_seconds=.05,
        transport=httpx.MockTransport(handle)))
    assert len(rows) == 8
    assert summary["phase_deadline_reached"] and summary["failures"] == 8 and summary["failure_rate"] == 1
    assert summary["successful_requests"] == 0 and summary["completion_requests_per_second"] == 0
    assert any(row.get("http_status") == 503 for row in rows)
    assert any(row["client_latency_seconds"] is None and not row["dispatched"] for row in rows)
    assert all(row["status"] == "failed" for row in rows)


def test_illustrative_targets_never_replace_diagnostic_correctness_with_valid_json() -> None:
    targets = {"minimum_model_success_rate": 1, "load_client_p95_seconds": 2, "maximum_load_failure_rate": 0}
    latency = [{"concurrency": 4, "p95_latency_seconds": 3, "failure_rate": 0, "phase_deadline_reached": False}]
    result = illustrative_objectives({"success_rate": 0, "all_diagnostic_contracts_passed": True}, latency, targets)
    assert not result["diagnostic_quality_passed"] and not result["all_load_checks_passed"]
    assert result["recommendation"] == "shadow_candidate"
    assert "not an achieved customer SLA" in result["scope"]
    result = illustrative_objectives({"success_rate": 1, "all_diagnostic_contracts_passed": False}, latency, targets)
    assert not result["diagnostic_quality_passed"]


def test_zero_budget_requests_cannot_be_substituted_for_load_measurements() -> None:
    with pytest.raises(ValueError, match="positive routing budgets"):
        asyncio.run(load_phase("http://fixture", "key", [ExecutionState("public", "support",
                    remaining_budget={"tokens": 1024, "agent_calls": 0})], k=2, stage="sft", concurrency=1,
                    request_count=1, timeout_seconds=1, max_phase_seconds=1))


@pytest.mark.parametrize("strict,service,quality,load,should_reject", [
    (False, True, False, False, False), (True, True, False, True, True),
    (True, True, True, False, True), (False, False, True, True, True),
    (True, True, True, True, False),
])
def test_strict_cli_blocks_bad_candidate_independently_of_completed_demo(
        monkeypatch: pytest.MonkeyPatch, strict: bool, service: bool, quality: bool,
        load: bool, should_reject: bool) -> None:
    result = {"acceptance": {"passed": service}, "illustrative_acceptance": {
        "diagnostic_quality_passed": quality, "all_load_checks_passed": load}}
    monkeypatch.setattr("conductor.demo.load_config", lambda path: {})
    monkeypatch.setattr("conductor.demo.run_demo", lambda config, **kwargs: result)
    monkeypatch.setattr(sys, "argv", ["conductor.demo"] + (["--require-model-acceptance"] if strict else []))
    if should_reject:
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1
    else:
        main()


@pytest.mark.parametrize("extra,existing,expected_modules", [
    ([], False, ["conductor.generate", "conductor.train", "conductor.demo"]),
    ([], True, ["conductor.demo"]), (["--help"], False, ["conductor.demo"]),
    (["--checkpoint", "missing-explicit-artifact"], False, ["conductor.demo"]),
    (["--config", "custom.yaml"], False, ["conductor.demo"]),
])
def test_shell_prepares_only_missing_default_trained_artifact(
        tmp_path: Path, extra: list[str], existing: bool, expected_modules: list[str]) -> None:
    script_root = tmp_path / "project"
    (script_root / "scripts").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/run_support_demo.sh"
    shutil.copyfile(source, script_root / "scripts/run_support_demo.sh")
    artifact = script_root / "outputs/checkpoints/tiny-sft/controller.json"
    if existing:
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}")
    interpreter = tmp_path / "record interpreter"
    interpreter.write_text(f"#!{sys.executable}\n" +
        "import json,os,pathlib,sys\n" +
        "with open(os.environ['SUPPORT_CALL_LOG'], 'a') as handle: handle.write(json.dumps(sys.argv[1:])+'\\n')\n" +
        "if sys.argv[1:3] == ['-m','conductor.train']:\n" +
        " p=pathlib.Path('outputs/checkpoints/tiny-sft/controller.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')\n")
    interpreter.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    result = subprocess.run(["bash", str(script_root / "scripts/run_support_demo.sh"), *extra],
        env={**os.environ, "CONDUCTOR_PYTHON": str(interpreter), "SUPPORT_CALL_LOG": str(log)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[1] for call in calls] == expected_modules
    assert calls[-1][-len(extra):] == extra if extra else calls[-1][-1] == "configs/demos/support.yaml"
