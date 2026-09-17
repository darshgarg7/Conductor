"""A bounded support-ticket evaluation and HTTP acceptance demonstration.

The workload is synthetic, agents are fixed fixtures, and targets are illustrative.
Completing this demo is neither production certification nor an NVIDIA capacity claim.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import yaml

from conductor.controller.artifacts import resolve_checkpoint
from conductor.agents.audit import specialist_audit
from conductor.evaluation.io import write_csv, write_jsonl
from conductor.evaluation.provenance import canonical_hash, checkpoint_sha256
from conductor.metrics.aggregate import aggregate_metrics, percentile, trajectory_metrics
from conductor.orchestration.runner import initial_state, run_trajectory
from conductor.schema import ExecutionState, RoutingDecision
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def validated_model_response(body: dict[str, Any], k: int, stage: str) -> RoutingDecision:
    """Validate provenance and telemetry before accepting a service decision."""
    if not isinstance(body, dict):
        raise ValueError("model response must be an object")
    if body.get("decision_source") != "model" or body.get("checkpoint_stage") != stage:
        raise ValueError("response is not from the expected trained model")
    if not isinstance(body["request_id"], str):
        raise ValueError("request ID must be a UUID string")
    UUID(body["request_id"])
    tokens = body["controller_tokens"]
    if type(tokens) is not int or tokens < 0:
        raise ValueError("invalid controller token accounting")
    if type(body["batch_size"]) is not int or body["batch_size"] < 1:
        raise ValueError("invalid batch size")
    if type(body["effective_k"]) is not int or not 1 <= body["effective_k"] <= k:
        raise ValueError("invalid effective k")
    for key in ("estimated_cost_usd", "queue_seconds", "batch_inference_seconds", "latency_seconds"):
        _nonnegative(body[key], key)
    if body["latency_seconds"] < body["queue_seconds"]:
        raise ValueError("server latency cannot be shorter than queue time")
    if not isinstance(body.get("token_accounting"), str) or not body["token_accounting"]:
        raise ValueError("missing token accounting description")
    return RoutingDecision(**body["decision"]).validate(body["effective_k"])


def _known_controller_usage(body: Any, stage: str) -> dict[str, Any]:
    known: dict[str, Any] = {}
    if isinstance(body, dict) and body.get("decision_source") == "model" and body.get("checkpoint_stage") == stage:
        if type(body.get("controller_tokens")) is int and body["controller_tokens"] >= 0:
            known.update(controller_tokens=body["controller_tokens"], controller_tokens_known=True)
        try:
            cost = _nonnegative(body.get("estimated_cost_usd"), "cost")
            known.update(estimated_cost_usd=cost, controller_cost_known=True)
        except ValueError:
            pass
    return known


class HTTPModelPolicy:
    """A real HTTP policy; failed calls terminate and remain explicit in records."""
    name = "support_http_model"

    def __init__(self, client: Any, key: str, stage: str) -> None:
        self.client, self.key, self.stage = client, key, stage
        self.last_tokens = 0
        self.last_cost_usd = 0.0
        self.last_invalid = False
        self.last_error = ""
        self.ticket_id = ""
        self.records: list[dict[str, Any]] = []
        self.request_ids: set[str] = set()

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        self.last_tokens, self.last_cost_usd, self.last_invalid, self.last_error = 0, 0.0, False, ""
        started = time.perf_counter()
        record: dict[str, Any] = {"phase": "ticket", "ticket_id": self.ticket_id,
            "step": state.current_step, "state_sha256": canonical_hash(state.to_dict()), "requested_k": k}
        try:
            response = self.client.post("/v1/route", headers={"X-API-Key": self.key},
                                        json={"state": state.to_dict(), "k": k})
            record["http_status"] = response.status_code
            response.raise_for_status()
            body = response.json()
            # Preserve independently validated usage from the expected endpoint
            # even when a later decision/timing/ID contract check rejects it.
            known = _known_controller_usage(body, self.stage)
            record.update(known)
            self.last_tokens = known.get("controller_tokens", 0)
            self.last_cost_usd = known.get("estimated_cost_usd", 0.0)
            decision = validated_model_response(body, k, self.stage)
            if body["request_id"] in self.request_ids:
                raise ValueError("duplicate routing request ID")
            self.request_ids.add(body["request_id"])
            self.last_tokens, self.last_cost_usd = body["controller_tokens"], body["estimated_cost_usd"]
            record.update(status="measured", **body)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            self.last_invalid = True
            self.last_error = f"HTTP {record['http_status']}" if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
            record.update(status="failed", error=self.last_error, controller_usage_known=False)
            decision = RoutingDecision([], confidence=0, terminate=True)
        record["client_latency_seconds"] = time.perf_counter() - started
        self.records.append(record)
        return decision


def _counters(text: str) -> dict[str, float]:
    values = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, value = line.split()
            values[key] = _nonnegative(float(value), "service metric")
    if not {"conductor_ready", "conductor_pending", "conductor_queue_depth",
            "conductor_recent_latency_seconds_count"} <= values.keys():
        raise ValueError("service metrics are missing required counters")
    return values


def service_contract_checks(client: Any, key: str, k: int) -> dict[str, Any]:
    """Boundary probes are excluded from learned routing/task quality statistics."""
    state = ExecutionState("Synthetic support acceptance probe", "support").to_dict()
    headers = {"X-API-Key": key}
    before_response = client.get("/metrics", headers=headers)
    before_response.raise_for_status()
    before = _counters(before_response.text)
    ready = client.get("/health/ready")
    unauthorized = client.post("/v1/route", json={"state": state, "k": k})
    invalid = client.post("/v1/route", headers=headers, json={"state": {**state, "current_step": -1}, "k": k})
    private = client.post("/v1/route", headers=headers,
                          json={"state": {**state, "expected_answer": "private grader field"}, "k": k})
    exhausted = {**state, "remaining_budget": {"tokens": 1024, "agent_calls": 0}}
    guard = client.post("/v1/route", headers=headers, json={"state": exhausted, "k": k})
    body = guard.json()
    after_response = client.get("/metrics", headers=headers)
    after_response.raise_for_status()
    after = _counters(after_response.text)
    checks = {"ready": ready.status_code == 200 and ready.json().get("ready") is True,
              "unauthorized_401": unauthorized.status_code == 401,
              "invalid_state_422": invalid.status_code == 422,
              "private_answer_field_422": private.status_code == 422,
              "zero_budget_guard": guard.status_code == 200 and body.get("decision_source") == "budget_guard"
                  and body.get("decision", {}).get("terminate") is True
                  and body.get("decision", {}).get("selected_agents") == []
                  and body.get("controller_tokens") == 0 and body.get("estimated_cost_usd") == 0,
              "probes_dispatch_no_model_work": before == after}
    return {"checks": checks, "passed": all(checks.values()),
            "http_statuses": {"unauthorized": unauthorized.status_code, "invalid_state": invalid.status_code,
                              "private_answer_field": private.status_code, "zero_budget": guard.status_code},
            "metrics_before": before, "metrics_after": after,
            "scope": "HTTP boundary probes only; no diagnostic correctness or trained-model overload claim."}


async def load_phase(url: str, key: str, states: list[ExecutionState], *, k: int, stage: str,
                     concurrency: int, request_count: int, timeout_seconds: float,
                     max_phase_seconds: float, transport: Any = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Closed-loop HTTP workload with real model dispatch and bounded wall time."""
    if not states or any(min(s.remaining_budget["tokens"], s.remaining_budget["agent_calls"]) < 1 for s in states):
        raise ValueError("load states must have positive routing budgets")
    gate = asyncio.Semaphore(concurrency)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=url, timeout=timeout_seconds, transport=transport) as client:
        async def request(index: int) -> None:
            state = states[index % len(states)]
            row: dict[str, Any] = {"phase": "load", "concurrency": concurrency, "index": index,
                "state_sha256": canonical_hash(state.to_dict()), "requested_k": k, "dispatched": False}
            arrived: float | None = None
            try:
                async with gate:
                    arrived = time.perf_counter()
                    row["dispatched"] = True
                    response = await client.post("/v1/route", headers={"X-API-Key": key},
                                                 json={"state": state.to_dict(), "k": k})
                    row["http_status"] = response.status_code
                    response.raise_for_status()
                    body = response.json()
                    row.update(_known_controller_usage(body, stage))
                    validated_model_response(body, k, stage)
                    row.update(status="measured", **body)
            except asyncio.CancelledError:
                row.update(status="failed", error="load_phase_deadline", controller_usage_known=False)
                raise
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                row.update(status="failed", error=type(error).__name__, controller_usage_known=False)
            finally:
                row["client_latency_seconds"] = time.perf_counter() - arrived if arrived is not None else None
                records.append(row)
        deadline_reached = False
        try:
            await asyncio.wait_for(asyncio.gather(*(request(index) for index in range(request_count))), max_phase_seconds)
        except asyncio.TimeoutError:
            deadline_reached = True
    elapsed = time.perf_counter() - started
    records.sort(key=lambda row: row["index"])
    seen: set[str] = set()
    for row in records:
        if row["status"] == "measured":
            if row["request_id"] in seen:
                row.update(status="failed", error="duplicate_request_id", controller_usage_known=False)
            seen.add(row["request_id"])
    good = [row for row in records if row["status"] == "measured"]
    latencies = [row["client_latency_seconds"] for row in records if row["client_latency_seconds"] is not None]
    summary = {"concurrency": concurrency, "requests": request_count, "recorded_requests": len(records),
        "dispatched_requests": sum(row["dispatched"] for row in records), "successful_requests": len(good),
        "failures": request_count - len(good), "failure_rate": (request_count - len(good)) / request_count,
        "wall_seconds": elapsed, "dispatch_requests_per_second": sum(row["dispatched"] for row in records) / elapsed,
        "requests_per_second": sum(row["dispatched"] for row in records) / elapsed,
        "completion_requests_per_second": len(good) / elapsed, "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95), "phase_deadline_reached": deadline_reached,
        "controller_tokens_known": sum(row["controller_tokens"] for row in good),
        "scope": "Closed-loop controller HTTP load; client timing excludes waiting for a load-generator slot, "
                 "includes transport/server queueing; no downstream work or GPU capacity conclusion."}
    return summary, records


def illustrative_objectives(diagnostics: dict[str, Any], load: list[dict[str, Any]],
                            targets: dict[str, Any]) -> dict[str, Any]:
    """Evaluate declared hypothetical objectives without converting them into SLAs."""
    success_target = _nonnegative(targets["minimum_model_success_rate"], "success target")
    if success_target > 1:
        raise ValueError("success target exceeds one")
    p95_target = _nonnegative(targets["load_client_p95_seconds"], "latency target")
    failure_target = _nonnegative(targets["maximum_load_failure_rate"], "failure target")
    if failure_target > 1:
        raise ValueError("failure target exceeds one")
    quality_pass = diagnostics["success_rate"] >= success_target and diagnostics["all_diagnostic_contracts_passed"]
    loads = [{"concurrency": row["concurrency"], "passed": not row["phase_deadline_reached"]
               and row["p95_latency_seconds"] is not None and row["p95_latency_seconds"] <= p95_target
               and row["failure_rate"] <= failure_target} for row in load]
    return {"targets": targets, "diagnostic_quality_passed": quality_pass, "load_checks": loads,
            "all_load_checks_passed": bool(loads) and all(row["passed"] for row in loads),
            "recommendation": "shadow_candidate" if not quality_pass else "continue_target_host_validation",
            "scope": "Illustrative synthetic-demo targets; pass/fail is not an achieved customer SLA or release approval."}


def summarize_trajectories(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return aggregate_metrics([trajectory_metrics(row) for row in trajectories])


class LocalService:
    def __init__(self, config_path: Path, key_env: str, log_path: Path, *, startup_seconds: float,
                 shutdown_seconds: float) -> None:
        self.config_path, self.log_path = config_path, log_path
        self.startup_seconds, self.shutdown_seconds = startup_seconds, shutdown_seconds
        self.key_env, self.key = key_env, secrets.token_hex(24)
        self.process: subprocess.Popen | None = None
        self.log: Any = None
        self.shutdown: dict[str, Any] = {}
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log = self.log_path.open("w")
        self.process = subprocess.Popen([sys.executable, "-m", "conductor.serve", "--config", str(self.config_path),
            "--host", "127.0.0.1", "--port", str(self.port)],
            env={**os.environ, self.key_env: self.key}, stdout=self.log, stderr=self.log)
        try:
            deadline = time.perf_counter() + self.startup_seconds
            with httpx.Client(base_url=self.url, timeout=1) as client:
                while time.perf_counter() < deadline:
                    if self.process.poll() is not None:
                        raise RuntimeError("service exited during startup; inspect server.log")
                    try:
                        response = client.get("/health/ready")
                        if response.status_code == 200 and response.json().get("ready") is True:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(.1)
            raise TimeoutError("service startup deadline exceeded")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        killed = False
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.shutdown_seconds)
            except subprocess.TimeoutExpired:
                killed = True
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
        returncode = self.process.returncode if self.process is not None else None
        completed = self.log_path.exists() and "Application shutdown complete" in self.log_path.read_text()
        self.shutdown = {"returncode": returncode, "forced_kill": killed, "application_shutdown_complete": completed,
                         "passed": not killed and completed and returncode in (0, -15)}


def run_demo(config: dict[str, Any], *, checkpoint: str | None = None,
             serving_config: str | None = None, output: str | None = None) -> dict[str, Any]:
    from conductor.demos.support import (CORPUS_SHA256, build_support_agents, check_support_answer,
                                         support_tasks, SupportRulePolicy)

    checkpoint = checkpoint or config["checkpoint"]
    artifact = resolve_checkpoint(checkpoint)
    metadata = json.loads((artifact / "controller.json").read_text())
    stage = metadata["stage"]
    if stage not in {"sft", "preference"}:
        raise ValueError("support demo requires an actual SFT or preference checkpoint")
    k = config["orchestration"]["k"]
    if type(k) is not int or not 1 <= k <= metadata["configuration"]["model"]["max_agents"]:
        raise ValueError("demo k exceeds the trained action catalog")
    service_config = copy.deepcopy(load_config(serving_config or config["serving_config"]))
    service_config["checkpoint"] = checkpoint
    service_config.setdefault("serving", {}).update(require_api_key=True, allow_random_initialization=False,
                                                   expected_stage=stage, api_key_env="CONDUCTOR_DEMO_API_KEY")
    lifecycle = config["lifecycle"]
    for name, value in lifecycle.items():
        if _nonnegative(value, name) <= 0:
            raise ValueError("lifecycle limits must be positive")
    load_config_values = config["load"]
    concurrency_values = load_config_values["concurrency"]
    count = load_config_values["requests_per_level"]
    if not concurrency_values or len(set(concurrency_values)) != len(concurrency_values):
        raise ValueError("load concurrency must be nonempty and unique")
    if any(type(value) is not int or not 1 <= value <= 128 for value in concurrency_values):
        raise ValueError("load concurrency must be an integer in [1, 128]")
    if type(count) is not int or not 1 <= count <= 4096:
        raise ValueError("requests per load level must be an integer in [1, 4096]")
    for name in ("timeout_seconds", "max_phase_seconds"):
        if _nonnegative(load_config_values[name], name) <= 0:
            raise ValueError("load time limits must be positive")
    tasks = support_tasks()
    if not tasks or len({task.id for task in tasks}) != len(tasks):
        raise ValueError("support corpus must contain nonempty unique task IDs")
    agents = build_support_agents()
    task_inventory_hash = canonical_hash([asdict(task) for task in tasks])
    audit = specialist_audit(agents)
    identities = {"support_task_inventory_sha256": task_inventory_hash,
                  "support_runbook_corpus_sha256": CORPUS_SHA256, "specialist_fingerprint": audit["fingerprint"]}
    identity = canonical_hash(identities)
    effective = {**copy.deepcopy(config), "checkpoint": checkpoint, "resolved_serving_config": service_config,
                 **identities, "support_workload_sha256": identity, "specialist_audit": audit,
                 "checkpoint_sha256": checkpoint_sha256(checkpoint)}
    directory = Path(output or config["output"])
    directory.mkdir(parents=True, exist_ok=False)
    run = Run(directory, effective, checkpoint)
    seed_everything(config.get("seed", 42))
    write_jsonl(directory / "tasks.jsonl", [asdict(task) for task in tasks])
    write_json(directory / "specialist_audit.json", audit)
    service_path = directory / "serving_config.yaml"
    service_path.write_text(yaml.safe_dump(service_config, sort_keys=False))
    service = LocalService(service_path, "CONDUCTOR_DEMO_API_KEY", directory / "server.log",
                           startup_seconds=lifecycle["startup_seconds"], shutdown_seconds=lifecycle["shutdown_seconds"])
    trajectories, ticket_records, diagnostics, load_summaries, load_records = [], [], [], [], []
    acceptance: dict[str, Any] = {}
    try:
        service.start()
        with httpx.Client(base_url=service.url, timeout=lifecycle["request_timeout_seconds"]) as client:
            acceptance = service_contract_checks(client, service.key, k)
            model = HTTPModelPolicy(client, service.key, stage)
            ticket_records = model.records
            for policy in (SupportRulePolicy(), model):
                for task in tasks:
                    if isinstance(policy, HTTPModelPolicy):
                        policy.ticket_id = task.id
                    start_index = len(model.records)
                    trajectory = asyncio.run(run_trajectory(task, policy, agents, **config["orchestration"]))
                    if trajectory.metadata["specialist_fingerprint"] != audit["fingerprint"]:
                        raise ValueError("specialist identities changed between policies/tasks")
                    observed = [item for step in trajectory.steps for item in step.agent_outputs]
                    checks = check_support_answer(task, trajectory.final_answer, observed)
                    if not checks or any(type(value) is not bool for value in checks.values()):
                        raise ValueError("diagnostic checker must return nonempty boolean contract gates")
                    calls = model.records[start_index:] if isinstance(policy, HTTPModelPolicy) else []
                    failures = sum(row["status"] != "measured" for row in calls)
                    trajectory.metadata.update(diagnostic_contract=checks, routing_http_failures=failures,
                        **identities, support_workload_sha256=identity,
                        checkpoint_sha256=effective["checkpoint_sha256"] if calls else None,
                        controller_token_accounting=calls[0].get("token_accounting", "unknown") if calls else "none")
                    if failures:
                        trajectory.metadata.update(token_usage_known=False, inference_cost_known=False)
                    trajectories.append(trajectory.to_dict())
                    diagnostics.append({"ticket_id": task.id, "policy": policy.name, "task_success": trajectory.task_success,
                        "diagnostic_contract_passed": all(checks.values()), "contract_checks": checks,
                        "routing_http_failures": failures, "agent_activations": trajectory.metadata["agent_activations"],
                        "controller_tokens": trajectory.metadata["controller_tokens"], "wall_seconds": trajectory.wall_clock_latency})
            ticket_records = model.records
            states = [initial_state(task, config["orchestration"]["token_budget"],
                                    config["orchestration"]["agent_call_budget"]) for task in tasks]
            for concurrency in concurrency_values:
                summary, records = asyncio.run(load_phase(service.url, service.key, states, k=k, stage=stage,
                    concurrency=concurrency, request_count=count, timeout_seconds=load_config_values["timeout_seconds"],
                    max_phase_seconds=load_config_values["max_phase_seconds"]))
                load_summaries.append(summary)
                load_records.extend(records)
            drain_response = client.get("/metrics", headers={"X-API-Key": service.key})
            drain_response.raise_for_status()
            drained = _counters(drain_response.text)
            acceptance["checks"].update(queue_drained=drained.get("conductor_pending") == 0
                                        and drained.get("conductor_queue_depth") == 0,
                                        model_responses_identified=bool(ticket_records) and all(
                                            row["status"] == "measured" for row in ticket_records))
            acceptance["final_metrics"] = drained
            measured = [row for row in ticket_records + load_records if row["status"] == "measured"]
            identifiers = [row["request_id"] for row in measured]
            acceptance["checks"].update(unique_model_request_ids=bool(identifiers)
                                         and len(set(identifiers)) == len(identifiers),
                                         sparse_agent_cap=bool(measured) and all(
                                            len(row["decision"]["selected_agents"]) <= k for row in measured))
    except BaseException as error:
        write_json(directory / "failure.json", {"error_type": type(error).__name__,
                   "scope": "Incomplete demonstration; inspect local server.log. No completed performance claim."})
        raise
    finally:
        service.close()
        write_json(directory / "shutdown.json", service.shutdown)
        write_jsonl(directory / "trajectories.jsonl", trajectories)
        write_jsonl(directory / "routing_requests.jsonl", ticket_records)
        write_jsonl(directory / "load_requests.jsonl", load_records)
        write_csv(directory / "load_requests.csv", load_records)
    acceptance["checks"]["graceful_shutdown"] = service.shutdown["passed"]
    acceptance["passed"] = all(acceptance["checks"].values())
    diagnostic_summaries = {}
    for policy_name in sorted({row["policy"] for row in diagnostics}):
        cases = [row for row in diagnostics if row["policy"] == policy_name]
        successes = sum(row["task_success"] for row in cases)
        valid = sum(row["diagnostic_contract_passed"] for row in cases)
        diagnostic_summaries[policy_name] = {"tasks": len(cases), "successes": successes, "contract_valid": valid,
            "success_rate": successes / len(cases), "contract_valid_rate": valid / len(cases),
            "all_diagnostic_contracts_passed": valid == len(cases),
            "mean_agent_activations": sum(row["agent_activations"] for row in cases) / len(cases),
            "controller_tokens": sum(row["controller_tokens"] for row in cases),
            "routing_http_failures": sum(row["routing_http_failures"] for row in cases),
            "p95_case_wall_seconds": percentile([row["wall_seconds"] for row in cases], 95)}
    quality = diagnostic_summaries[HTTPModelPolicy.name]
    metrics = {"status": "measured", **identities, "support_workload_sha256": identity,
        "checkpoint": {"sha256": effective["checkpoint_sha256"], "stage": stage,
                       "pretrained": metadata.get("pretrained", False), "backend": metadata["backend"]},
        "policies": summarize_trajectories(trajectories),
        "tickets": diagnostics, "diagnostics": diagnostic_summaries,
        "acceptance": acceptance, "load": load_summaries,
        "illustrative_acceptance": illustrative_objectives(quality, load_summaries, config["illustrative_targets"]),
        "scope": "Synthetic support fixtures, fixed agents, actual trained controller through localhost HTTP. "
                 "This is a customer-demo/shadow evaluation, not a customer SLA, production deployment, "
                 "cost saving, broad generalization, or NVIDIA capacity claim."}
    write_csv(directory / "tickets.csv", [{k: v for k, v in row.items() if k != "contract_checks"} for row in diagnostics])
    write_json(directory / "service_acceptance.json", acceptance)
    run.finish(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demos/support.yaml")
    parser.add_argument("--checkpoint", help="Explicit trained SFT/preference artifact; never random fallback")
    parser.add_argument("--serving-config", help="Device/serving overrides, for example the Granite CPU config")
    parser.add_argument("--output", help="Fresh immutable output directory")
    parser.add_argument("--require-model-acceptance", action="store_true",
                        help="Reject the candidate if service, diagnostic quality or illustrative load gates fail")
    args = parser.parse_args()
    metrics = run_demo(load_config(args.config), checkpoint=args.checkpoint,
                       serving_config=args.serving_config, output=args.output)
    objectives = metrics["illustrative_acceptance"]
    if not metrics["acceptance"]["passed"] or (args.require_model_acceptance and (
            not objectives["diagnostic_quality_passed"] or not objectives["all_load_checks_passed"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
