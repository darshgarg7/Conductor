"""Exercise admission, deadlines, telemetry and real ASGI request validation."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from conductor.schema import ExecutionState, RoutingDecision
from conductor.serving.api import BodyLimitMiddleware, create_app
from conductor.serving.engine import Overloaded, RoutingEngine, Unavailable


class FakeController:
    device = "cpu"
    stage = "sft"
    catalog = SimpleNamespace(max_agents=3)
    token_accounting = "test fixture tokens"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.entered = threading.Event()
        self.calls: list[tuple[int, list[str]]] = []

    def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(2), "test did not release the fake model worker"
        self.calls.append((k, [state.user_task for state in states]))
        self.last_batch_tokens = [10 + len(state.user_task) for state in states]
        self.last_batch_costs = [len(state.user_task) / 1000 for state in states]
        return [RoutingDecision(["math"] if k == 1 else ["math", "verifier"]) for _ in states]


def settings(**values: Any) -> dict[str, Any]:
    return {"seed": 5, "serving": {"require_api_key": False, "warmup": False,
                                  "batch_wait_seconds": 0.005, "expected_stage": "sft", **values}}


def payload(**changes: Any) -> dict[str, Any]:
    state = {"user_task": "Compute 2 + 3", "task_type": "math"}
    state.update(changes.pop("state", {}))
    return {"state": state, "k": 2, **changes}


async def entered(controller: FakeController) -> None:
    for _ in range(1000):
        if controller.entered.is_set():
            return
        await asyncio.sleep(0.001)
    raise AssertionError("model worker did not start")


def test_compatible_k_groups_preserve_per_request_telemetry_and_snapshot() -> None:
    async def exercise() -> None:
        controller = FakeController()
        engine = RoutingEngine(controller, max_batch_size=3, batch_wait_seconds=0.02)
        await engine.start()
        state = ExecutionState("a", "math")
        tasks = [asyncio.create_task(engine.submit(state, 1)),
                 asyncio.create_task(engine.submit(ExecutionState("bbbb", "math"), 2)),
                 asyncio.create_task(engine.submit(ExecutionState("cc", "math"), 1))]
        await asyncio.sleep(0)
        state.user_task = "mutated after admission"
        results = await asyncio.gather(*tasks)
        assert controller.calls == [(1, ["a", "cc"]), (2, ["bbbb"])]
        assert [result["controller_tokens"] for result in results] == [11, 14, 12]
        assert [result["estimated_cost_usd"] for result in results] == [0.001, 0.004, 0.002]
        assert [result["batch_size"] for result in results] == [2, 1, 2]
        assert len({result["request_id"] for result in results}) == 3
        assert all(result["latency_seconds"] >= result["queue_seconds"] >= 0 for result in results)
        assert engine.metrics()["counters"]["completed"] == 3
        await engine.close()
        assert not engine.ready() and engine.worker.done()
    asyncio.run(exercise())


def test_pending_cap_includes_running_model_and_queue() -> None:
    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        engine = RoutingEngine(controller, max_pending=2, max_batch_size=1)
        await engine.start()
        first = asyncio.create_task(engine.submit(ExecutionState("first", "math"), 1))
        await entered(controller)
        second = asyncio.create_task(engine.submit(ExecutionState("second", "math"), 1))
        await asyncio.sleep(0)
        with pytest.raises(Overloaded):
            await engine.submit(ExecutionState("third", "math"), 1)
        assert engine.metrics()["pending"] == 2
        assert engine.metrics()["counters"]["rejected_overload"] == 1
        gate.set()
        await asyncio.gather(first, second)
        await engine.close()
    asyncio.run(exercise())


def test_deadline_stalls_readiness_and_discarded_work_recovers() -> None:
    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        engine = RoutingEngine(controller, request_timeout=0.03, batch_wait_seconds=0)
        await engine.start()
        request = asyncio.create_task(engine.submit(ExecutionState("slow", "math"), 1))
        await entered(controller)
        with pytest.raises(asyncio.TimeoutError):
            await request
        await asyncio.sleep(0.01)
        assert not engine.ready()
        with pytest.raises(Unavailable):
            await engine.submit(ExecutionState("new", "math"), 1)
        gate.set()
        for _ in range(100):
            if engine.ready():
                break
            await asyncio.sleep(0.001)
        assert engine.ready()
        assert engine.counters["timed_out"] == 1
        assert engine.counters["completed_after_cancellation"] == 1
        assert (await engine.submit(ExecutionState("recovered", "math"), 1))["controller_tokens"] == 19
        await engine.close()
    asyncio.run(exercise())


def test_client_cancellation_and_bounded_forced_shutdown() -> None:
    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        engine = RoutingEngine(controller, shutdown_timeout=0.02, batch_wait_seconds=0)
        await engine.start()
        cancelled = asyncio.create_task(engine.submit(ExecutionState("cancelled", "math"), 1))
        await entered(controller)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        pending = asyncio.create_task(engine.submit(ExecutionState("pending", "math"), 1))
        await asyncio.sleep(0)
        started = time.perf_counter()
        await engine.close()
        assert time.perf_counter() - started < 0.5
        assert engine.worker.done() and not engine.ready()
        assert engine.metrics()["pending"] == 0 and engine.metrics()["queue_depth"] == 0
        with pytest.raises(Unavailable):
            await pending
        assert engine.counters["cancelled"] == 1 and engine.counters["shutdown_timeout"] == 1
        gate.set()  # Running threads/kernels are deliberately not claimed to be preempted.
        await engine.close()  # Idempotent.
    asyncio.run(exercise())


def test_expired_second_k_group_never_dispatches() -> None:
    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        engine = RoutingEngine(controller, max_batch_size=2, batch_wait_seconds=0.005)
        await engine.start()
        first = asyncio.create_task(engine.submit(ExecutionState("first", "math"), 1))
        second = asyncio.create_task(engine.submit(ExecutionState("expired", "math"), 2, timeout=0.015))
        await entered(controller)
        with pytest.raises(asyncio.TimeoutError):
            await second
        gate.set()
        await first
        assert controller.calls == [(1, ["first"])]
        await engine.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("broken", ["decisions", "tokens", "negative_tokens", "nan_cost"])
def test_bad_model_batch_fails_requests_without_poisoning_worker(broken: str) -> None:
    class Broken(FakeController):
        def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
            decisions = super().batch_route(states, k)
            if broken == "decisions":
                return []
            if broken == "tokens":
                self.last_batch_tokens = []
            if broken == "negative_tokens":
                self.last_batch_tokens = [-1]
            if broken == "nan_cost":
                self.last_batch_costs = [float("nan")]
            return decisions

    async def exercise() -> None:
        engine = RoutingEngine(Broken(), batch_wait_seconds=0)
        await engine.start()
        with pytest.raises(RuntimeError, match="controller"):
            await engine.submit(ExecutionState("bad", "math"), 1)
        assert engine.ready() and engine.counters["failed_batches"] == 1
        await engine.close()
    asyncio.run(exercise())


def test_authentication_health_metrics_and_budget_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDUCTOR_ROUTING_API_KEY", "local-test-secret")
    controller = FakeController()
    with TestClient(create_app(settings(require_api_key=True), controller)) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/ready").json() == {"ready": True}
        assert client.get("/metrics").status_code == 401
        assert client.post("/v1/route", json=payload()).status_code == 401
        assert client.post("/v1/route", json=payload(), headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.post("/v1/route", json=payload(), headers={"X-API-Key": b"\xff"}).status_code == 401
        headers = {"X-API-Key": "local-test-secret"}
        result = client.post("/v1/route", json=payload(state={"remaining_budget": {"agent_calls": 1, "tokens": 90}}),
                             headers=headers)
        assert result.status_code == 200, result.text
        assert result.json()["effective_k"] == 1
        assert result.json()["decision"]["selected_agents"] == ["math"]
        for budget in ({"agent_calls": 0, "tokens": 90}, {"agent_calls": 3, "tokens": 0}):
            result = client.post("/v1/route", json=payload(state={"remaining_budget": budget}), headers=headers).json()
            assert result["decision_source"] == "budget_guard" and result["decision"]["terminate"]
            assert result["controller_tokens"] == 0
        assert len(controller.calls) == 1
        metrics = client.get("/metrics", headers=headers)
        assert metrics.status_code == 200 and "conductor_completed_total 1" in metrics.text
        assert "local-test-secret" not in metrics.text and "Compute" not in metrics.text
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize("changes", [
    {"k": True}, {"k": "2"}, {"k": 4}, {"unknown": 1}, {"timeout_seconds": True},
    {"state": {"current_step": True}}, {"state": {"agents_already_called": ["alien"]}},
    {"state": {"remaining_budget": {"agent_calls": 1.5, "tokens": 20}}},
    {"state": {"remaining_budget": {"agent_calls": True, "tokens": 20}}},
    {"state": {"remaining_budget": {"agent_calls": "2", "tokens": 20}}},
    {"state": {"remaining_budget": {"tokens": 20}}}, {"state": {"user_task": ""}},
    {"state": {"expected_answer": "grader leakage"}},
])
def test_strict_invalid_requests_never_invoke_model(changes: dict[str, Any]) -> None:
    controller = FakeController()
    with TestClient(create_app(settings(), controller)) as client:
        result = client.post("/v1/route", json=payload(**changes))
        assert result.status_code == 422, result.text
    assert not controller.calls


def test_nonfinite_nested_json_and_body_limits() -> None:
    controller = FakeController()
    with TestClient(create_app(settings(max_body_bytes=300), controller)) as client:
        result = client.post("/v1/route", content=json.dumps(payload(state={"tool_results": [{"score": float("nan")}] })),
                             headers={"Content-Type": "application/json"})
        assert result.status_code == 422
        assert client.post("/v1/route", content=b"x" * 301).status_code == 413
    assert not controller.calls


def test_excessive_json_depth_is_rejected_and_brackets_in_tasks_are_preserved() -> None:
    controller = FakeController()
    with TestClient(create_app(settings(max_json_depth=8), controller)) as client:
        nested = "[" * 20 + "0" + "]" * 20
        body = '{"state":{"user_task":"hello","task_type":"math","tool_results":' + nested + '},"k":1}'
        response = client.post("/v1/route", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 422 and "nesting" in response.json()["detail"]
        task = '[' * 40 + '\\"' + ']' * 40
        assert client.post("/v1/route", json=payload(state={"user_task": task})).status_code == 200
    assert controller.calls == [(2, [task])]


def test_startup_rejects_missing_key_checkpoint_and_wrong_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDUCTOR_ROUTING_API_KEY", raising=False)
    with pytest.raises(ValueError, match="authentication requires"):
        with TestClient(create_app(settings(require_api_key=True), FakeController())):
            pass
    with pytest.raises(ValueError, match="explicit trained checkpoint"):
        with TestClient(create_app(settings())):
            pass
    with pytest.raises(ValueError, match="expects checkpoint stage preference"):
        with TestClient(create_app(settings(expected_stage="preference"), FakeController())):
            pass


def test_trained_catalog_is_enforced() -> None:
    controller = FakeController()
    controller.catalog = SimpleNamespace(max_agents=1)
    with TestClient(create_app(settings(), controller)) as client:
        assert client.post("/v1/route", json=payload(k=2)).status_code == 422
        assert client.post("/v1/route", json=payload(k=1)).status_code == 200
    assert controller.calls == [(1, ["Compute 2 + 3"])]


def test_body_middleware_limits_chunked_requests_and_slow_reads() -> None:
    async def exercise() -> None:
        calls = []
        async def app(scope, receive, send):
            calls.append(await receive())
        async def run(messages, max_bytes=5, delayed=False):
            sent = []
            async def receive():
                if delayed:
                    await asyncio.sleep(0.02)
                return messages.pop(0)
            async def send(message):
                sent.append(message)
            await BodyLimitMiddleware(app, max_bytes, 0.005)({"type": "http", "method": "POST"}, receive, send)
            return sent
        sent = await run([{"type": "http.request", "body": b"abc", "more_body": True},
                          {"type": "http.request", "body": b"def", "more_body": False}])
        assert sent[0]["status"] == 413 and not calls
        sent = await run([{"type": "http.request", "body": b"a", "more_body": False}], delayed=True)
        assert sent[0]["status"] == 408 and not calls
        assert await run([{"type": "http.request", "body": b"abc", "more_body": True},
                          {"type": "http.request", "body": b"de", "more_body": False}]) == []
        assert calls == [{"type": "http.request", "body": b"abcde", "more_body": False}]
    asyncio.run(exercise())


def test_body_deadline_is_total_time_rather_than_reset_per_chunk() -> None:
    async def exercise() -> None:
        called, sent = [], []
        async def app(scope, receive, send):
            called.append(True)
        async def receive():
            await asyncio.sleep(0.004)
            return {"type": "http.request", "body": b" ", "more_body": True}
        async def send(message):
            sent.append(message)
        await BodyLimitMiddleware(app, 100, 0.01)({"type": "http", "method": "POST"}, receive, send)
        assert sent[0]["status"] == 408 and not called
    asyncio.run(exercise())


def test_http_overload_deadline_and_stall_health_have_distinct_statuses() -> None:
    import httpx

    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        app = create_app(settings(max_pending=1, request_timeout_seconds=0.04, batch_wait_seconds=0), controller)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                request = asyncio.create_task(client.post("/v1/route", json=payload()))
                await entered(controller)
                overload = await client.post("/v1/route", json=payload())
                assert overload.status_code == 429 and overload.headers["Retry-After"] == "1"
                assert (await request).status_code == 504
                await asyncio.sleep(0.005)
                assert (await client.get("/health/ready")).status_code == 503
                assert (await client.post("/v1/route", json=payload())).status_code == 503
                assert (await client.get("/health/live")).status_code == 200
                gate.set()
                for _ in range(100):
                    if app.state.engine.ready():
                        break
                    await asyncio.sleep(0.001)
                assert (await client.post("/v1/route", json=payload())).status_code == 200
    asyncio.run(exercise())


def test_http_model_error_does_not_disclose_internal_exception() -> None:
    class Broken(FakeController):
        def batch_route(self, states: list[ExecutionState], k: int) -> list[RoutingDecision]:
            raise RuntimeError("private-token-and-filesystem-path")
    with TestClient(create_app(settings(), Broken())) as client:
        response = client.post("/v1/route", json=payload())
        assert response.status_code == 500
        assert response.json() == {"detail": "routing model execution failed"}
        assert "private-token" not in response.text
        assert client.get("/health/ready").status_code == 200


def test_warmup_has_bounded_startup_deadline() -> None:
    async def exercise() -> None:
        gate = threading.Event()
        controller = FakeController(gate)
        engine = RoutingEngine(controller, request_timeout=0.01)
        try:
            with pytest.raises(asyncio.TimeoutError):
                await engine.start(ExecutionState("warmup", "math"))
            assert not engine.ready()
        finally:
            gate.set()
            await engine.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("options", [{"max_pending": 0}, {"max_batch_size": True},
    {"request_timeout": float("nan")}, {"batch_wait_seconds": float("inf")}, {"shutdown_timeout": -1}])
def test_invalid_service_capacity_configuration_is_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="configuration"):
        RoutingEngine(FakeController(), **options)


def test_real_tiny_controller_is_frozen_and_serves_valid_sparse_routes() -> None:
    from conductor.controller.tiny import TinyController
    controller = TinyController({"model": {"backend": "tiny", "hidden_dim": 8, "num_experts": 2,
                                           "expert_top_k": 1, "max_agents": 3}})
    with TestClient(create_app(settings(expected_stage="random_initialization"), controller)) as client:
        result = client.post("/v1/route", json=payload(k=1))
        assert result.status_code == 200, result.text
        observed = result.json()
        RoutingDecision(**observed["decision"]).validate(1)
        assert observed["controller_tokens"] > 0 and observed["checkpoint_stage"] == "random_initialization"
        assert not controller.model.training and all(not p.requires_grad for p in controller.model.parameters())
