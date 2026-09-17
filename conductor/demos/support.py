"""Synthetic HPC/AI support fixtures and frozen, deterministic read-only tools.

These fixtures and rules were designed together. They demonstrate coordination
and service integration, not trained specialist LLMs, enterprise diagnostic
quality, an independent benchmark, or measured NVIDIA hardware performance.
Runbook recommendations were checked against primary NVIDIA docs on 2026-09-17.
Command strings are illustrations only: no command, hardware probe or network
request is executed by this module. Private fixture labels never enter state.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from conductor.agents.base import Agent, estimated_tokens
from conductor.schema import AGENT_NAMES, AgentOutput, ExecutionState, RoutingDecision, Task


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Runbook:
    id: str
    title: str
    text: str
    sources: tuple[str, ...]
    illustrative_commands: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "text": self.text, "sources": list(self.sources),
                "illustrative_commands": list(self.illustrative_commands), "commands_executed": False}


SMI = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
RUNBOOKS = (
    Runbook("rb-triage", "Evidence limits and escalation",
            "A symptom alone does not establish a root cause. Keep process, container and host observations distinct. "
            "Request missing current observations before changing settings. A successful current check does not prove "
            "every workload healthy. An NCCL timeout can involve several subsystems; escalate with existing logs and topology.",
            (SMI, "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html"),
            ("nvidia-smi -q", "nvidia-smi topo -m")),
    Runbook("rb-device-visibility", "Process device visibility and ordinals",
            "An empty CUDA_VISIBLE_DEVICES hides all GPUs from a CUDA process. Visible devices are enumerated locally "
            "from zero, regardless of their host indices. Use the measured process-visible device count and requested "
            "ordinal together; an error string or host GPU count alone cannot establish the intended process mapping.",
            ("https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html",),
            ("nvidia-smi -L",)),
    Runbook("rb-driver-runtime", "Driver support is distinct from application runtime",
            "The CUDA version in nvidia-smi describes the driver's maximum CUDA support, not the framework runtime "
            "installed in a process. NVIDIA's minor-compatibility table gives minimum driver families 450 for CUDA 11, "
            "525 for CUDA 12 and 580 for CUDA 13. Meeting that floor alone does not guarantee architecture, PTX or "
            "feature compatibility. Compare the actual runtime and driver, and consult the documented limits.",
            ("https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html",
             "https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html"),
            ("nvidia-smi --query-gpu=driver_version --format=csv,noheader",)),
    Runbook("rb-container-visibility", "Container GPU exclusion",
            "NVIDIA_VISIBLE_DEVICES=none exposes no GPU while allowing driver capabilities. Host GPU visibility "
            "does not establish container visibility. Inspect the approved launch specification and the container's "
            "observed count; do not recreate containers or change permissions in this demonstration.",
            ("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html",),
            ("nvidia-smi -L",)),
    Runbook("rb-memory-observation", "GPU memory observations",
            "nvidia-smi reports device memory observations. A request larger than measured free memory at failure "
            "supports a capacity-pressure finding, without identifying every allocation or allocator mechanism. "
            "A snapshot taken after failure does not establish free memory at the failed allocation; collect "
            "contemporaneous observations before diagnosing fragmentation, leaks or a required hardware upgrade.",
            (SMI,), ("nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv",)),
)
BOOKS = {book.id: book for book in RUNBOOKS}
CORPUS_SHA256 = hashlib.sha256(_json([book.to_dict() for book in RUNBOOKS]).encode()).hexdigest()
ACTIONS = {
    "process_gpu_visibility_empty": "Review the approved GPU allocation and the process-visible device setting with the operator.",
    "process_device_ordinal_out_of_range": "Inspect the process-local GPU count and map the intended device to its visible local ordinal.",
    "driver_below_runtime_minimum": "Compare the actual runtime and driver with NVIDIA's compatibility table and escalate an approved software change.",
    "driver_banner_is_not_runtime_version": "Record driver support and framework runtime separately; no change is justified by this version difference alone.",
    "container_gpu_excluded": "Review the approved container GPU visibility and launch specification with the operator.",
    "insufficient_free_memory_at_failure": "Review contemporaneous device-memory observations and workload allocation requirements before proposing a change.",
    "no_current_gpu_failure_observed": "Record the successful current check and request fresh evidence if the symptom recurs.",
    "unknown_escalate": "Escalate with current process visibility, runtime details and relevant existing logs; do not infer an unobserved root cause.",
}
CAPABILITIES = {
    "planner": "Plan a read-only investigation and identify retrieval-before-synthesis dependencies.",
    "retriever": "Retrieve public NVIDIA runbook records relevant to supplied structured observations.",
    "researcher": "Synthesize a bounded diagnosis from retrieved runbooks and supplied observations.",
    "coder": "Canonicalize an existing grounded response; never execute or generate a fix program.",
    "tool_executor": "Return supplied telemetry snapshots; never execute commands or contact hardware.",
    "critic": "Identify unsupported claims and missing evidence in an existing response.",
    "verifier": "Check an existing response against public observations and retrieved runbooks.",
    "math": "Compare numeric memory quantities in supplied observations; never infer unobserved usage.",
}


def _ticket(text: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    ticket = json.loads(text)
    observations = ticket["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("a support ticket needs public observation records")
    values, ids = {}, {}
    seen = set()
    for record in observations:
        kind, identifier = record["kind"], record["id"]
        if not isinstance(kind, str) or not isinstance(identifier, str) or kind in values or identifier in seen:
            raise ValueError("observation IDs and kinds must be unique strings")
        values[kind], ids[kind] = record["value"], identifier
        seen.add(identifier)
    return ticket, values, ids


def _relevant_books(values: dict[str, Any]) -> set[str]:
    groups = {
        "rb-device-visibility": {"cuda_visible_devices", "visible_device_count", "requested_device_ordinal"},
        "rb-driver-runtime": {"driver_version", "framework_cuda_version", "nvidia_smi_cuda_version"},
        "rb-container-visibility": {"host_gpu_count", "container_gpu_count", "nvidia_visible_devices"},
        "rb-memory-observation": {"requested_mib", "free_mib", "memory_snapshot_at_failure"},
    }
    return {"rb-triage"} | {name for name, keys in groups.items() if keys & values.keys()}


def _retrieved_books(outputs: list[dict[str, Any]]) -> set[str]:
    result = set()
    for output in outputs:
        if output.get("agent") != "retriever":
            continue
        for record in output.get("metadata", {}).get("runbooks", []):
            if isinstance(record, dict) and record.get("id") in BOOKS and record == BOOKS[record["id"]].to_dict():
                result.add(record["id"])
    return result


def _number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _major(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    first = value.split(".", 1)[0]
    return int(first) if first.isdigit() else None


def _response(diagnosis: str, evidence: list[str]) -> dict[str, Any]:
    return {"diagnosis": diagnosis, "next_action": ACTIONS[diagnosis], "evidence": sorted(set(evidence))}


def _assessment(values: dict[str, Any], ids: dict[str, str], books: set[str]) -> dict[str, Any]:
    """Explicit public-evidence predicates; summaries, IDs and labels are unused."""
    def supported(diagnosis: str, book: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
        return _response(diagnosis, [book, *(ids[key] for key in keys)]) if book in books else None

    finding = None
    error = values.get("runtime_error")
    count, ordinal = values.get("visible_device_count"), values.get("requested_device_ordinal")
    if error == "no_cuda_devices" and values.get("cuda_visible_devices") == "" and type(count) is int and count == 0:
        finding = supported("process_gpu_visibility_empty", "rb-device-visibility",
                            ("runtime_error", "cuda_visible_devices", "visible_device_count"))
    elif (error == "invalid_device_ordinal" and type(count) is int and count > 0 and type(ordinal) is int
          and not 0 <= ordinal < count):
        finding = supported("process_device_ordinal_out_of_range", "rb-device-visibility",
                            ("runtime_error", "visible_device_count", "requested_device_ordinal"))
    elif (error == "no_cuda_devices" and type(values.get("host_gpu_count")) is int and values["host_gpu_count"] > 0
          and values.get("container_gpu_count") == 0 and values.get("nvidia_visible_devices") == "none"):
        finding = supported("container_gpu_excluded", "rb-container-visibility",
                            ("runtime_error", "host_gpu_count", "container_gpu_count", "nvidia_visible_devices"))
    elif error == "cuda_driver_incompatible":
        runtime, driver = _major(values.get("framework_cuda_version")), _major(values.get("driver_version"))
        minimum = {11: 450, 12: 525, 13: 580}.get(runtime)
        if minimum is not None and driver is not None and driver < minimum:
            finding = supported("driver_below_runtime_minimum", "rb-driver-runtime",
                                ("runtime_error", "framework_cuda_version", "driver_version"))
    elif (error == "cuda_out_of_memory" and values.get("memory_snapshot_at_failure") is True
          and _number(values.get("requested_mib")) and _number(values.get("free_mib"))
          and values["requested_mib"] > values["free_mib"] >= 0):
        finding = supported("insufficient_free_memory_at_failure", "rb-memory-observation",
                            ("runtime_error", "memory_snapshot_at_failure", "requested_mib", "free_mib"))
    elif error == "none" and values.get("current_validation_success") is True:
        if (isinstance(values.get("nvidia_smi_cuda_version"), str)
                and isinstance(values.get("framework_cuda_version"), str)
                and values["nvidia_smi_cuda_version"] != values["framework_cuda_version"]):
            finding = supported("driver_banner_is_not_runtime_version", "rb-driver-runtime",
                                ("runtime_error", "current_validation_success", "nvidia_smi_cuda_version", "framework_cuda_version"))
        else:
            finding = supported("no_current_gpu_failure_observed", "rb-triage",
                                ("runtime_error", "current_validation_success"))
    return finding or _response("unknown_escalate", [*books, *ids.values()])


def check_support_answer(task: Task, answer: str, observed_outputs: list[dict[str, Any]]) -> dict[str, bool]:
    """Public-evidence contract check, separate from private exact-answer grading.

    This fixture-aware validator shares public deterministic runbook predicates
    with synthesis. It is not an independent enterprise judge or LLM grader.
    """
    checks = {name: False for name in ("valid_json", "schema_valid", "evidence_grounded", "diagnosis_grounded",
                                      "read_only_next_action", "commands_not_executed", "contract_valid")}
    try:
        candidate = json.loads(answer)
        checks["valid_json"] = True
        checks["schema_valid"] = (isinstance(candidate, dict) and set(candidate) == {"diagnosis", "next_action", "evidence"}
                                  and isinstance(candidate["diagnosis"], str) and isinstance(candidate["next_action"], str)
                                  and isinstance(candidate["evidence"], list) and bool(candidate["evidence"])
                                  and all(isinstance(value, str) for value in candidate["evidence"])
                                  and len(set(candidate["evidence"])) == len(candidate["evidence"]))
        if not checks["schema_valid"]:
            return checks
        _, values, ids = _ticket(task.user_task)
        books = _retrieved_books(observed_outputs)
        required = _assessment(values, ids, books)
        checks["evidence_grounded"] = ("rb-triage" in books and set(candidate["evidence"]) <= books | set(ids.values())
                                       and set(required["evidence"]) <= set(candidate["evidence"]))
        checks["diagnosis_grounded"] = candidate["diagnosis"] == required["diagnosis"]
        checks["read_only_next_action"] = candidate["next_action"] == ACTIONS.get(candidate["diagnosis"])
        checks["commands_not_executed"] = all(output.get("metadata", {}).get("commands_executed") is False
                                              for output in observed_outputs)
        checks["contract_valid"] = all(value for name, value in checks.items() if name != "contract_valid")
    except (ValueError, TypeError, KeyError):
        pass
    return checks


def _candidate(outputs: list[dict[str, Any]]) -> str | None:
    return next((output.get("metadata", {}).get("answer") for output in reversed(outputs)
                 if isinstance(output.get("metadata", {}).get("answer"), str)), None)


@dataclass(frozen=True)
class SupportAgent:
    name: str
    frozen: bool = True

    @property
    def capability(self) -> str:
        return CAPABILITIES[self.name]

    @property
    def config(self) -> dict[str, Any]:
        return {"backend": "deterministic", "corpus_sha256": CORPUS_SHA256, "capability": self.capability,
                "scope": "synthetic read-only support tool; no trained specialist model"}

    async def execute(self, state: ExecutionState) -> AgentOutput:
        started = time.perf_counter()
        await asyncio.sleep(0)
        ticket, values, ids = _ticket(state.user_task)
        metadata: dict[str, Any] = {"answer": None, "backend": "deterministic", "token_accounting": "estimated_whitespace",
                                    "commands_executed": False, "synthetic_support_tool": True}
        if self.name == "retriever":
            records = [BOOKS[name].to_dict() for name in sorted(_relevant_books(values))]
            metadata["runbooks"] = records
            content = _json({"runbooks": records})
        elif self.name == "researcher":
            books = _retrieved_books(state.previous_agent_outputs)
            if "rb-triage" not in books:
                content = "Need retrieved public runbooks before synthesizing a supported diagnosis."
            else:
                content = _json(_assessment(values, ids, books))
                metadata["answer"] = content
        elif self.name == "tool_executor":
            content = _json({"observations": ticket["observations"], "scope": "supplied synthetic snapshots only", "commands_executed": False})
        elif self.name == "math":
            difference = (values["requested_mib"] - values["free_mib"]
                          if _number(values.get("requested_mib")) and _number(values.get("free_mib")) else None)
            content = _json({"request_minus_observed_free_mib": difference, "snapshot_at_failure": values.get("memory_snapshot_at_failure")})
        elif self.name == "planner":
            content = "Plan: retrieve the public runbook, then synthesize from supplied evidence; escalate missing or conflicting observations."
        else:
            candidate = _candidate(state.previous_agent_outputs)
            check = check_support_answer(Task("", state.user_task, state.task_type), candidate or "", state.previous_agent_outputs)
            content = _json({"candidate_present": candidate is not None, "checks": check})
            if self.name in {"coder", "verifier"} and check["contract_valid"]:
                metadata["answer"] = _json(json.loads(candidate))
                content = metadata["answer"]
        tokens = estimated_tokens(_json(state.to_dict())) + estimated_tokens(content)
        return AgentOutput(self.name, content, tokens, time.perf_counter() - started, 0.0, metadata)


def build_support_agents() -> dict[str, Agent]:
    return {name: SupportAgent(name) for name in AGENT_NAMES}


class SupportRulePolicy:
    name = "support_rule_based"
    last_tokens = 0
    last_cost_usd = 0.0

    def route(self, state: ExecutionState, k: int) -> RoutingDecision:
        if _candidate(state.previous_agent_outputs) is not None:
            return RoutingDecision([], terminate=True).validate(k)
        if "rb-triage" not in _retrieved_books(state.previous_agent_outputs):
            selected = ["retriever", "researcher"] if k >= 2 else ["retriever"]
            return RoutingDecision(selected, "sequential" if k >= 2 else "parallel").validate(k)
        return RoutingDecision(["researcher"]).validate(k)


def support_tasks() -> list[Task]:
    """Twelve fixed openly synthetic test fixtures; labels authored separately."""
    cases = [
        ("No devices after an empty process visibility setting.",
         {"runtime_error": "no_cuda_devices", "cuda_visible_devices": "", "visible_device_count": 0},
         "process_gpu_visibility_empty", ["runtime_error", "cuda_visible_devices", "visible_device_count"], ["rb-device-visibility"]),
        ("A host GPU index was passed as a process-local ordinal after masking.",
         {"runtime_error": "invalid_device_ordinal", "cuda_visible_devices": "3", "visible_device_count": 1, "requested_device_ordinal": 3},
         "process_device_ordinal_out_of_range", ["runtime_error", "visible_device_count", "requested_device_ordinal"], ["rb-device-visibility"]),
        ("An application reports that its runtime is incompatible with the driver.",
         {"runtime_error": "cuda_driver_incompatible", "framework_cuda_version": "13.0", "driver_version": "550.54.15"},
         "driver_below_runtime_minimum", ["runtime_error", "framework_cuda_version", "driver_version"], ["rb-driver-runtime"]),
        ("The nvidia-smi CUDA banner differs from the framework runtime, but a current validation passed.",
         {"runtime_error": "none", "current_validation_success": True, "nvidia_smi_cuda_version": "12.8", "framework_cuda_version": "12.6"},
         "driver_banner_is_not_runtime_version", ["runtime_error", "current_validation_success", "nvidia_smi_cuda_version", "framework_cuda_version"], ["rb-driver-runtime"]),
        ("The host has GPUs; this container intentionally excludes GPU devices.",
         {"runtime_error": "no_cuda_devices", "host_gpu_count": 4, "container_gpu_count": 0, "nvidia_visible_devices": "none"},
         "container_gpu_excluded", ["runtime_error", "host_gpu_count", "container_gpu_count", "nvidia_visible_devices"], ["rb-container-visibility"]),
        ("The supplied snapshot was taken at the failed allocation.",
         {"runtime_error": "cuda_out_of_memory", "requested_mib": 2048, "free_mib": 256, "memory_snapshot_at_failure": True},
         "insufficient_free_memory_at_failure", ["runtime_error", "requested_mib", "free_mib", "memory_snapshot_at_failure"], ["rb-memory-observation"]),
        ("Out-of-memory occurred earlier; the only memory snapshot is from after the failed process exited.",
         {"runtime_error": "cuda_out_of_memory", "requested_mib": 1024, "free_mib": 24000, "memory_snapshot_at_failure": False},
         "unknown_escalate", ["runtime_error", "requested_mib", "free_mib", "memory_snapshot_at_failure"], ["rb-memory-observation", "rb-triage"]),
        ("Invalid ordinal was reported, but no process-visible count was collected.",
         {"runtime_error": "invalid_device_ordinal", "requested_device_ordinal": 3},
         "unknown_escalate", ["runtime_error", "requested_device_ordinal"], ["rb-device-visibility", "rb-triage"]),
        ("No CUDA devices were observed; framework build, driver and allocation details are missing.",
         {"runtime_error": "no_cuda_devices", "visible_device_count": 0},
         "unknown_escalate", ["runtime_error", "visible_device_count"], ["rb-device-visibility", "rb-triage"]),
        ("A collective timed out. The ticket contains no communicator, network or per-rank diagnostic logs.",
         {"runtime_error": "nccl_timeout", "visible_device_count": 2},
         "unknown_escalate", ["runtime_error", "visible_device_count"], ["rb-device-visibility", "rb-triage"]),
        ("Historical notes say CUDA out of memory and invalid ordinal; those incidents were closed and the current check passed.",
         {"runtime_error": "none", "current_validation_success": True},
         "no_current_gpu_failure_observed", ["runtime_error", "current_validation_success"], ["rb-triage"]),
        ("Older notes blamed container GPU visibility; the current container check passed with an exposed GPU.",
         {"runtime_error": "none", "current_validation_success": True, "host_gpu_count": 4, "container_gpu_count": 1, "nvidia_visible_devices": "all"},
         "no_current_gpu_failure_observed", ["runtime_error", "current_validation_success"], ["rb-triage"]),
    ]
    tasks = []
    for index, (summary, facts, diagnosis, evidence_kinds, book_ids) in enumerate(cases, 1):
        observations = [{"id": f"obs-{kind}", "kind": kind, "value": value} for kind, value in facts.items()]
        public = {"synthetic": True, "read_only": True, "summary": summary, "observations": observations,
                  "request": "Return a diagnosis, a read-only next action and supporting observation/runbook IDs; escalate uncertainty."}
        label = _json(_response(diagnosis, [*book_ids, *(f"obs-{kind}" for kind in evidence_kinds)]))
        tasks.append(Task(f"support-fixture-{index:02d}", _json(public), "synthetic_hpc_ai_support", "test", label,
                          {"scope": "openly synthetic fixtures; rules and labels designed together"}))
    return tasks
