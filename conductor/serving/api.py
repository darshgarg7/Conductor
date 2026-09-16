"""Validated routing HTTP API with bounded bodies, authentication and lifecycle."""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.serving.engine import Overloaded, RoutingEngine, Unavailable
from conductor.utils.runs import seed_everything


class StatePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    user_task: str = Field(min_length=1, max_length=64000)
    task_type: str = Field(min_length=1, max_length=256)
    conversation_state: list[dict[str, str]] = Field(default_factory=list, max_length=128)
    previous_agent_outputs: list[dict[str, Any]] = Field(default_factory=list, max_length=128)
    agents_already_called: list[str] = Field(default_factory=list, max_length=128)
    tool_results: list[dict[str, Any]] = Field(default_factory=list, max_length=128)
    remaining_budget: dict[str, float] = Field(default_factory=lambda: {"agent_calls": 12, "tokens": 8192})
    current_step: int = Field(default=0, ge=0, le=10000, strict=True)
    previous_routing_decisions: list[dict[str, Any]] = Field(default_factory=list, max_length=128)

    @field_validator("remaining_budget")
    @classmethod
    def validate_budget(cls, value: dict[str, float]) -> dict[str, float]:
        if not {"agent_calls", "tokens"} <= value.keys():
            raise ValueError("remaining_budget requires agent_calls and tokens")
        if any(not math.isfinite(number) or number < 0 for number in value.values()):
            raise ValueError("budgets must be nonnegative finite numbers")
        if value["agent_calls"] != int(value["agent_calls"]):
            raise ValueError("agent_calls must be an integer")
        return value

    @field_validator("agents_already_called")
    @classmethod
    def validate_agents(cls, value: list[str]) -> list[str]:
        if any(agent not in AGENT_NAMES for agent in value):
            raise ValueError("execution history contains an unknown specialist")
        return value


class RoutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    state: StatePayload
    k: int = Field(default=2, ge=1, le=3, strict=True)
    timeout_seconds: float | None = Field(default=None, gt=0, le=300)


class BodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int, read_timeout: float, max_json_depth: int = 64) -> None:
        if max_bytes < 1 or not math.isfinite(read_timeout) or read_timeout <= 0 or max_json_depth < 1:
            raise ValueError("body/read limits must be positive")
        self.app, self.max_bytes, self.read_timeout, self.max_json_depth = app, max_bytes, read_timeout, max_json_depth

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        pieces, size = [], 0
        deadline = asyncio.get_running_loop().time() + self.read_timeout
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                message = await asyncio.wait_for(receive(), remaining)
                if message["type"] == "http.disconnect":
                    return
                piece = message.get("body", b"")
                size += len(piece)
                if size > self.max_bytes:
                    await JSONResponse({"detail": "request body exceeds configured limit"}, 413)(scope, receive, send)
                    return
                pieces.append(piece)
                if not message.get("more_body", False):
                    break
        except asyncio.TimeoutError:
            await JSONResponse({"detail": "request body read timed out"}, 408)(scope, receive, send)
            return
        body = b"".join(pieces)
        # Bound parser nesting independently of body bytes. Ignore punctuation in
        # escaped JSON strings so tasks containing brackets retain their meaning.
        depth, quoted, escaped = 0, False, False
        for character in body:
            if quoted:
                if escaped:
                    escaped = False
                elif character == 92:
                    escaped = True
                elif character == 34:
                    quoted = False
            elif character == 34:
                quoted = True
            elif character in (91, 123):
                depth += 1
                if depth > self.max_json_depth:
                    await JSONResponse({"detail": "execution state exceeds JSON nesting limit"}, 422)(scope, receive, send)
                    return
            elif character in (93, 125):
                depth -= 1
        delivered = False

        async def replay() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(config: dict[str, Any], controller: Any = None) -> FastAPI:
    service = config.get("serving", {})
    require_key = bool(service.get("require_api_key", True))
    key_env = service.get("api_key_env", "CONDUCTOR_ROUTING_API_KEY")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        key = os.environ.get(key_env)
        if require_key and not key:
            raise ValueError(f"service authentication requires environment variable {key_env}")
        seed_everything(int(config.get("seed", 42)))
        model = controller
        if model is None:
            from conductor.controller.factory import build_controller
            checkpoint = config.get("checkpoint")
            if not checkpoint and not service.get("allow_random_initialization", False):
                raise ValueError("serving requires an explicit trained checkpoint")
            model = await asyncio.to_thread(build_controller, config, checkpoint)
        if hasattr(model, "model"):
            model.model.eval().requires_grad_(False)
        expected = service.get("expected_stage")
        if expected and getattr(model, "stage", None) != expected:
            raise ValueError(f"service expects checkpoint stage {expected}, got {getattr(model, 'stage', None)}")
        engine = RoutingEngine(model, max_batch_size=int(service.get("max_batch_size", 8)),
                    max_pending=int(service.get("max_pending", 128)),
                    batch_wait_seconds=float(service.get("batch_wait_seconds", 0.002)),
                    request_timeout=float(service.get("request_timeout_seconds", 30)),
                    shutdown_timeout=float(service.get("shutdown_timeout_seconds", 10)))
        app.state.engine, app.state.controller, app.state.api_key = engine, model, key
        try:
            warmup = ExecutionState("Compute 1 + 2.", "math") if service.get("warmup", True) else None
            await engine.start(warmup)
            yield
        finally:
            await engine.close()

    app = FastAPI(title="Conductor routing", version="0.2.0", lifespan=lifespan,
                  docs_url=None if service.get("disable_docs", True) else "/docs", redoc_url=None,
                  openapi_url=None if service.get("disable_docs", True) else "/openapi.json")
    app.add_middleware(BodyLimitMiddleware, max_bytes=int(service.get("max_body_bytes", 1_000_000)),
                       read_timeout=float(service.get("body_read_timeout_seconds", 5)),
                       max_json_depth=int(service.get("max_json_depth", 64)))

    def authorize(request: HTTPRequest, supplied: str | None) -> None:
        if require_key and (supplied is None or not secrets.compare_digest(
                supplied.encode("utf-8"), request.app.state.api_key.encode("utf-8"))):
            raise HTTPException(401, "invalid routing API key")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: HTTPRequest) -> JSONResponse:
        engine = getattr(request.app.state, "engine", None)
        available = engine is not None and engine.ready()
        return JSONResponse({"ready": available}, status_code=200 if available else 503)

    @app.get("/metrics")
    async def metrics(request: HTTPRequest, x_api_key: str | None = Header(default=None)) -> PlainTextResponse:
        authorize(request, x_api_key)
        observed = request.app.state.engine.metrics()
        lines = [f"conductor_ready {int(observed['ready'])}", f"conductor_pending {observed['pending']}",
                 f"conductor_queue_depth {observed['queue_depth']}"]
        lines.extend(f"conductor_{key}_total {count}" for key, count in observed["counters"].items())
        lines.extend([f"conductor_recent_latency_seconds_sum {observed['recent_latency_sum_seconds']}",
                      f"conductor_recent_latency_seconds_count {observed['recent_latency_count']}"])
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/v1/route")
    async def route(payload: RoutePayload, request: HTTPRequest,
                    x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(request, x_api_key)
        try:
            json.dumps(payload.state.model_dump(), allow_nan=False)
        except (ValueError, TypeError, RecursionError) as error:
            raise HTTPException(422, "execution state must contain finite JSON values") from error
        state = ExecutionState(**payload.state.model_dump())
        if state.remaining_budget["agent_calls"] < 1 or state.remaining_budget["tokens"] < 1:
            return {"decision": RoutingDecision([], confidence=0, terminate=True).to_dict(),
                    "decision_source": "budget_guard", "controller_tokens": 0, "estimated_cost_usd": 0.0}
        max_agents = getattr(getattr(request.app.state.controller, "catalog", None), "max_agents", 3)
        if payload.k > max_agents:
            raise HTTPException(422, "requested k exceeds the checkpoint's trained action catalog")
        effective_k = min(payload.k, int(state.remaining_budget["agent_calls"]))
        try:
            result = await request.app.state.engine.submit(state, effective_k, payload.timeout_seconds)
            result.update(decision_source="model", effective_k=effective_k,
                          checkpoint_stage=getattr(request.app.state.controller, "stage", "unavailable"))
            return result
        except Overloaded as error:
            raise HTTPException(429, "routing queue is full", headers={"Retry-After": "1"}) from error
        except Unavailable as error:
            raise HTTPException(503, "routing worker is unavailable") from error
        except asyncio.TimeoutError as error:
            raise HTTPException(504, "routing request deadline exceeded") from error
        except Exception as error:
            raise HTTPException(500, "routing model execution failed") from error

    return app
