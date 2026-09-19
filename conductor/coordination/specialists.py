"""Frozen typed specialists with observable public-artifact dependencies.

The implementations are deterministic synthetic tools, not downstream language
models. Their controlled capability boundaries must not be generalized to real
multi-agent necessity. No specialist receives a Task or grading reference.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conductor.agents.base import estimated_tokens
from conductor.datasets.integrity import digest
from conductor.schema import AGENT_NAMES, AgentOutput, ExecutionState
from conductor.coordination.workloads import PublicStores, parse_request

CAPABILITIES = {
    "planner": "Describe public dependencies and missing artifact kinds without producing a task answer.",
    "retriever": "Read numeric source records or a public error's fallback record; never synthesize research or calculate.",
    "researcher": "Reconcile approved research records against retrieved criteria, select rates, or resolve a public stale handle.",
    "coder": "Transform a supplied value or compile a bounded recipe from supplied evidence; never fetch sources or execute code.",
    "tool_executor": "Invoke a public fixture or execute an allowlisted compiled recipe; tool failures are state-dependent.",
    "critic": "Inspect observed artifacts for blocked handoffs, conflicting values and retryable errors; never fetch or solve.",
    "verifier": "Independently recompute source grounding and a supplied candidate's public specification; never create a missing candidate.",
    "math": "Compute from retrieved numeric artifacts and researched rates; never retrieve, reverse strings or repair tools.",
}


def artifact_digest(record: Mapping[str, Any]) -> str:
    return digest({key: value for key, value in record.items() if key != "id"})


def _artifact(request: dict[str, Any], kind: str, value: Any, *, evidence: list[str] | None = None,
              inputs: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    record = {"request_digest": digest(request), "kind": kind, "value": value,
              "evidence": sorted(set(evidence or [])), "inputs": list(inputs or []), **extra}
    return {"id": artifact_digest(record), **record}


def artifacts(state: ExecutionState, kind: str | None = None) -> list[dict[str, Any]]:
    """Only current-request, content-addressed artifacts are consumable."""
    fingerprint = digest(parse_request(state))
    result, seen = [], set()
    for output in state.previous_agent_outputs:
        for item in output.get("metadata", {}).get("artifacts", []):
            if (isinstance(item, dict) and item.get("request_digest") == fingerprint
                    and item.get("id") == artifact_digest(item) and item["id"] not in seen
                    and (kind is None or item.get("kind") == kind)):
                result.append(item)
                seen.add(item["id"])
    return result


def _last(records: list[dict[str, Any]], kind: str, purpose: str | None = None) -> dict[str, Any] | None:
    return next((item for item in reversed(records) if item["kind"] == kind
                 and (purpose is None or item.get("purpose") == purpose)), None)


def _facts(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    values, evidence, inputs = {}, [], []
    for item in records:
        if item["kind"] == "facts":
            values.update(item["value"])
            evidence.extend(item["evidence"])
            inputs.append(item["id"])
    return values, sorted(set(evidence)), inputs


def _approved(record: Mapping[str, Any], threshold: int | None = None) -> str:
    eligible = [row for row in record["entries"] if row["approved"] is True
                and (threshold is None or row["revision"] > threshold)]
    if not eligible:
        raise ValueError("no eligible approved research record")
    return str(max(eligible, key=lambda row: row["revision"])["value"])


def expected_public_result(request: dict[str, Any], stores: PublicStores) -> str:
    """Independent public semantic specification; no Task or private answer."""
    refs, parameters, operation = request["source_refs"], request["parameters"], request["operation"]
    fact_refs = refs.get("fact_refs", [])
    value = int(stores.numeric_facts[fact_refs[0]]["value"]) if fact_refs else None
    if operation == "calculate":
        result = str(value * int(parameters["factor"]))
    elif operation == "transform":
        result = str(value * int(parameters.get("pre_scale", 1)))[::-1]
    elif operation == "research":
        result = _approved(stores.research_records[refs["research_refs"][0]], value + int(parameters.get("threshold_offset", 0)))
    elif operation == "combine":
        record = stores.research_records[refs["research_refs"][0]]
        selector = stores.tool_fixtures[refs["tool_ref"]]["value"] if parameters.get("rate_selector_from_tool") else record["selected"]
        result = str(value * int(record["rates"][selector]))
    elif operation == "resolve_conflict":
        result = _approved(stores.research_records[refs["research_refs"][0]])
    else:
        result = str(stores.tool_fixtures[refs["tool_ref"]]["value"])
    return result[::-1] if parameters.get("post_transform") else result


def check_grounded_candidate(request: dict[str, Any], candidate: dict[str, Any],
                             records: list[dict[str, Any]], stores: PublicStores) -> tuple[bool, dict[str, bool]]:
    """Recompute an artifact's public support, ignoring producer names/routes."""
    checks = {"source_grounded": False, "result_matches_request": False, "dependencies_valid": False}
    try:
        by_id = {item["id"]: item for item in records}
        fingerprint = digest(request)
        if (candidate.get("kind") != "candidate" or candidate.get("purpose") != "result"
                or candidate.get("id") not in by_id):
            return False, checks
        visiting, checked, closure = set(), set(), []

        def validate(item: dict[str, Any]) -> bool:
            identity = item["id"]
            if identity in checked:
                return True
            if (identity in visiting or item.get("request_digest") != fingerprint or artifact_digest(item) != identity
                    or not isinstance(item.get("inputs"), list) or not isinstance(item.get("evidence"), list)
                    or any(reference not in stores.source_ids for reference in item["evidence"])):
                return False
            visiting.add(identity)
            if any(reference not in by_id or not validate(by_id[reference]) for reference in item["inputs"]):
                return False
            kind, value = item["kind"], item["value"]
            if kind == "facts":
                if (not isinstance(value, dict) or set(value) != set(item["evidence"])
                        or any(reference not in stores.numeric_facts or stored != stores.numeric_facts[reference]["value"]
                               for reference, stored in value.items())):
                    return False
                # A fallback is learned through an observed tool error, not an implicit source fetch.
                if request["operation"] == "read_tool" and not item["inputs"]:
                    return False
            elif kind == "research":
                record_ref = value.get("record_ref") if isinstance(value, dict) else None
                if record_ref not in stores.research_records:
                    return False
                record = stores.research_records[record_ref]
                if "rate" in value:
                    selector = value["selector"]
                    if value["rate"] != record["rates"].get(selector):
                        return False
                elif "resolved" in value:
                    threshold = None
                    if request["operation"] == "research":
                        reference = request["source_refs"]["fact_refs"][0]
                        threshold = int(stores.numeric_facts[reference]["value"]) + int(request["parameters"].get("threshold_offset", 0))
                    if value["resolved"] != _approved(record, threshold):
                        return False
                else:
                    return False
            elif kind == "tool_failure":
                if (not isinstance(value, dict) or value.get("resource_ref") not in stores.tool_fixtures
                        or value.get("error") != stores.tool_fixtures[value["resource_ref"]].get("failure")
                        or not value.get("error")):
                    return False
            elif kind == "tool_result":
                resource = item.get("resource_ref")
                if resource not in stores.tool_fixtures or str(value) != str(stores.tool_fixtures[resource]["value"]):
                    return False
                if (resource != request["source_refs"].get("tool_ref")
                        or stores.tool_fixtures[resource].get("failure")) and not item["inputs"]:
                    return False
            elif kind == "recovery":
                if not item["inputs"] or not isinstance(value, dict):
                    return False
                if "replacement_tool_ref" in value:
                    record = value.get("record_ref")
                    if record not in stores.research_records or value["replacement_tool_ref"] != stores.research_records[record].get("replacement_tool_ref"):
                        return False
            elif kind == "program":
                if value != {"operation": request["operation"], "parameters": request["parameters"]} or not item["inputs"]:
                    return False
            elif kind == "critique":
                if (not item["inputs"] or not isinstance(value, dict)
                        or value.get("unconfirmed_source") is not True):
                    return False
            elif kind == "candidate":
                if not item["inputs"]:
                    return False
                if item.get("purpose") == "arithmetic":
                    reference = request["source_refs"]["fact_refs"][0]
                    scale = request["parameters"].get("factor", request["parameters"].get("pre_scale", 1))
                    if str(value) != str(int(stores.numeric_facts[reference]["value"]) * int(scale)):
                        return False
                elif item.get("purpose") == "threshold":
                    reference = request["source_refs"]["fact_refs"][0]
                    if str(value) != str(int(stores.numeric_facts[reference]["value"]) + int(request["parameters"]["threshold_offset"])):
                        return False
            else:
                return False
            visiting.remove(identity)
            checked.add(identity)
            closure.append(item)
            return True

        checks["dependencies_valid"] = validate(candidate)
        if not checks["dependencies_valid"]:
            return False, checks
        source_support = {reference for item in closure if item["kind"] in {"facts", "research", "tool_failure", "tool_result", "recovery"}
                          for reference in item["evidence"]}
        required = set(request["source_refs"].get("fact_refs", [])) | set(request["source_refs"].get("research_refs", []))
        if request["source_refs"].get("tool_ref"):
            required.add(request["source_refs"]["tool_ref"])
        checks["source_grounded"] = required <= source_support and set(candidate["evidence"]) == source_support
        checks["result_matches_request"] = str(candidate["value"]) == expected_public_result(request, stores)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, checks
    return all(checks.values()), checks


def assemble_workflow_answer(state: ExecutionState) -> str:
    request, records = parse_request(state), artifacts(state)
    candidates = {item["id"]: item for item in records if item["kind"] == "candidate" and item.get("purpose") == "result"}
    for item in reversed(records):
        if item["kind"] in {"verification", "execution"} and item.get("verified") is True:
            if any(reference in candidates for reference in item["inputs"]):
                return json.dumps({"result": str(item["value"]), "evidence": sorted(item["evidence"]), "verified": True},
                                  sort_keys=True, separators=(",", ":"))
    if not request["verification"] and candidates:
        candidate = list(candidates.values())[-1]
        return json.dumps({"result": str(candidate["value"]), "evidence": sorted(candidate["evidence"]), "verified": False},
                          sort_keys=True, separators=(",", ":"))
    return ""


@dataclass(frozen=True)
class WorkflowAgent:
    name: str
    stores: PublicStores
    frozen: bool = True

    def __post_init__(self) -> None:
        if self.name not in AGENT_NAMES or self.frozen is not True:
            raise ValueError("workflow specialists must be known and frozen")

    @property
    def capability(self) -> str:
        return CAPABILITIES[self.name]

    @property
    def config(self) -> dict[str, Any]:
        return {"backend": "workflow", "corpus_sha256": self.stores.corpus_sha256,
                "implementation_version": "controlled_workflow_v2_1", "capability": self.capability,
                "scope": "frozen deterministic synthetic artifact contracts; no language model parameters"}

    async def execute(self, state: ExecutionState) -> AgentOutput:
        started = time.perf_counter()
        await asyncio.sleep(0)
        request, records = parse_request(state), artifacts(state)
        emitted, status, reason = self._execute(request, records)
        content = json.dumps({"status": status, "reason": reason, "artifacts": emitted}, sort_keys=True, separators=(",", ":"))
        metadata: dict[str, Any] = {"status": status, "reason": reason, "artifacts": emitted,
            "answer": None, "backend": "workflow", "token_accounting": "estimated_whitespace",
            "commands_executed": False, "arbitrary_code_executed": False}
        supplied = ExecutionState(state.user_task, state.task_type, previous_agent_outputs=[
            {"agent": self.name, "metadata": {"artifacts": records + emitted}}])
        answer = assemble_workflow_answer(supplied)
        if answer and emitted and emitted[-1]["kind"] in {"candidate", "verification", "execution"}:
            metadata["answer"] = answer
        tokens = estimated_tokens(json.dumps(state.to_dict(), sort_keys=True)) + estimated_tokens(content)
        return AgentOutput(self.name, content, tokens, time.perf_counter() - started, metadata=metadata)

    def _execute(self, request: dict[str, Any], records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
        try:
            result = self._produce(request, records)
            return result, "ok", "public_artifact_contract_satisfied"
        except (KeyError, ValueError, TypeError) as error:
            return [], "blocked", str(error)

    def _produce(self, request: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        operation, refs, parameters = request["operation"], request["source_refs"], request["parameters"]
        values, evidence, inputs = _facts(records)
        if self.name == "planner":
            return [_artifact(request, "plan", {"operation": operation, "available_artifact_kinds": sorted({item["kind"] for item in records}),
                                                "verification_requested": request["verification"]})]
        if self.name == "retriever":
            references, predecessors = refs.get("fact_refs", []), []
            if operation == "read_tool":
                failure = _last(records, "tool_failure")
                if failure is None or not failure["value"].get("fallback_fact_ref"):
                    raise ValueError("missing public fallback reference from an observed tool failure")
                references = [failure["value"]["fallback_fact_ref"]]
                predecessors = [failure["id"]]
            if not references:
                raise ValueError("no public numeric reference requested")
            facts = _artifact(request, "facts", {reference: self.stores.numeric_facts[reference]["value"] for reference in references},
                              evidence=references, inputs=predecessors)
            if operation == "read_tool":
                support = sorted(set(references + [refs["tool_ref"]]))
                return [facts, _artifact(request, "candidate", str(facts["value"][references[0]]), evidence=support,
                                         inputs=[facts["id"]], purpose="result")]
            return [facts]
        if self.name == "researcher":
            if operation == "read_tool":
                failure = _last(records, "tool_failure")
                if failure is None or not failure["value"].get("repair_record_ref"):
                    raise ValueError("no observed stale handle with a public repair record")
                record_ref = failure["value"]["repair_record_ref"]
                replacement = self.stores.research_records[record_ref]["replacement_tool_ref"]
                return [_artifact(request, "recovery", {"record_ref": record_ref, "replacement_tool_ref": replacement},
                                  evidence=[refs["tool_ref"], record_ref], inputs=[failure["id"]])]
            if not refs.get("research_refs"):
                raise ValueError("no public research record requested")
            record_ref = refs["research_refs"][0]
            record = self.stores.research_records[record_ref]
            if operation == "combine":
                selector = record["selected"]
                dependencies, support = [], [record_ref]
                if parameters.get("rate_selector_from_tool"):
                    selected = _last(records, "tool_result")
                    if selected is None:
                        raise ValueError("missing tool-provided rate selector")
                    selector = selected["value"]
                    dependencies.append(selected["id"])
                    support.extend(selected["evidence"])
                return [_artifact(request, "research", {"record_ref": record_ref, "selector": selector, "rate": record["rates"][selector]},
                                  evidence=support, inputs=dependencies)]
            reference = refs["fact_refs"][0]
            if reference not in values:
                raise ValueError("missing retrieved threshold or estimate")
            threshold = None
            if operation == "research":
                threshold = int(values[reference])
                if parameters.get("threshold_offset"):
                    adjusted = _last(records, "candidate", "threshold")
                    if adjusted is None:
                        raise ValueError("missing adjusted-threshold handoff")
                    threshold = int(adjusted["value"])
                    inputs.append(adjusted["id"])
            elif operation == "resolve_conflict":
                critique = _last(records, "critique")
                if critique is None or critique.get("value", {}).get("unconfirmed_source") is not True:
                    raise ValueError("missing critique of the unconfirmed estimate")
                inputs.append(critique["id"])
            resolved = _approved(record, threshold)
            support = sorted(set(evidence + [record_ref]))
            research = _artifact(request, "research", {"record_ref": record_ref, "resolved": resolved}, evidence=support, inputs=inputs)
            purpose = "research_value" if parameters.get("post_transform") else "result"
            return [research, _artifact(request, "candidate", resolved, evidence=support, inputs=[research["id"]], purpose=purpose)]
        if self.name == "math":
            reference = refs.get("fact_refs", [None])[0]
            if reference not in values:
                raise ValueError("missing retrieved numeric operand")
            quantity = int(values[reference])
            purpose = "result"
            if operation == "calculate":
                result = quantity * int(parameters["factor"])
                if parameters.get("post_transform"):
                    purpose = "arithmetic"
            elif operation == "transform" and parameters.get("pre_scale"):
                result, purpose = quantity * int(parameters["pre_scale"]), "arithmetic"
            elif operation == "research" and parameters.get("threshold_offset"):
                result, purpose = quantity + int(parameters["threshold_offset"]), "threshold"
            elif operation == "combine":
                rate = _last(records, "research")
                if rate is None or "rate" not in rate["value"]:
                    raise ValueError("missing independent researched-rate handoff")
                result = quantity * int(rate["value"]["rate"])
                evidence = sorted(set(evidence + rate["evidence"]))
                inputs.append(rate["id"])
            else:
                raise ValueError("operation is outside numeric calculation capability")
            return [_artifact(request, "candidate", str(result), evidence=evidence, inputs=inputs, purpose=purpose)]
        if self.name == "coder":
            if operation == "transform" or parameters.get("post_transform"):
                if parameters.get("pre_scale") or operation == "calculate":
                    predecessor = _last(records, "candidate", "arithmetic")
                elif operation == "resolve_conflict":
                    predecessor = _last(records, "candidate", "research_value")
                else:
                    predecessor = None
                if parameters.get("pre_scale") or parameters.get("post_transform"):
                    if predecessor is None:
                        raise ValueError("missing numeric or researched artifact for the requested transformation")
                    text = str(predecessor["value"])
                    evidence, inputs = predecessor["evidence"], [predecessor["id"]]
                else:
                    reference = refs["fact_refs"][0]
                    if reference not in values:
                        raise ValueError("missing retrieved value for coding transformation")
                    text = str(values[reference])
                return [_artifact(request, "candidate", text[::-1], evidence=evidence, inputs=inputs, purpose="result")]
            if operation == "calculate":
                if refs["fact_refs"][0] not in values:
                    raise ValueError("missing inputs for bounded program compilation")
                return [_artifact(request, "program", {"operation": operation, "parameters": parameters}, evidence=evidence, inputs=inputs)]
            raise ValueError("no bounded transformation or program compilation requested")
        if self.name == "critic":
            failure = _last(records, "tool_failure")
            if failure and failure["value"].get("error") == "temporary_unavailable":
                return [_artifact(request, "recovery", {"retry_tool_ref": failure["value"]["resource_ref"], "retry_allowed": True},
                                  evidence=failure["evidence"], inputs=[failure["id"]])]
            if operation == "resolve_conflict" and values:
                return [_artifact(request, "critique", {"unconfirmed_source": True,
                                  "candidate_present": _last(records, "candidate") is not None},
                                  evidence=evidence, inputs=inputs)]
            return [_artifact(request, "critique", {"candidate_present": _last(records, "candidate") is not None,
                                                    "observed_failure": failure is not None,
                                                    "verification_present": _last(records, "verification") is not None})]
        if self.name == "tool_executor":
            program = _last(records, "program")
            if program:
                # Bounded recipe interpreter; no eval, subprocess, arbitrary source or generated-code execution.
                reference = refs.get("fact_refs", [None])[0]
                if operation != "calculate" or reference not in values:
                    raise ValueError("unsupported recipe or missing execution inputs")
                result = str(int(values[reference]) * int(parameters["factor"]))
                if parameters.get("post_transform"):
                    result = result[::-1]
                candidate = _artifact(request, "candidate", result, evidence=program["evidence"], inputs=[program["id"]], purpose="result")
                good, checks = check_grounded_candidate(request, candidate, records + [candidate], self.stores)
                if not good:
                    raise ValueError("compiled recipe does not have valid public grounding")
                execution = _artifact(request, "execution", result, evidence=candidate["evidence"],
                                      inputs=[candidate["id"], program["id"]], verified=True,
                                      checks={**checks, "verification_performed": True, "program_executed": True, "assertions_passed": True})
                return [candidate, execution]
            resource = refs.get("tool_ref")
            if resource is None:
                raise ValueError("no public tool reference or compiled program requested")
            recovery = _last(records, "recovery")
            if recovery and recovery["value"].get("replacement_tool_ref"):
                resource = recovery["value"]["replacement_tool_ref"]
            fixture = self.stores.tool_fixtures[resource]
            can_retry = recovery and recovery["value"].get("retry_tool_ref") == resource and recovery["value"].get("retry_allowed") is True
            if fixture.get("failure") and not can_retry:
                details = {"resource_ref": resource, "error": fixture["failure"]}
                for field in ("fallback_fact_ref", "repair_record_ref"):
                    if fixture.get(field):
                        details[field] = fixture[field]
                return [_artifact(request, "tool_failure", details, evidence=[resource])]
            dependencies = [recovery["id"]] if recovery else []
            support = sorted(set([resource] + (recovery["evidence"] if recovery else [])))
            result = _artifact(request, "tool_result", str(fixture["value"]), evidence=support, inputs=dependencies, resource_ref=resource)
            if operation == "read_tool":
                return [result, _artifact(request, "candidate", str(fixture["value"]), evidence=support, inputs=[result["id"]], purpose="result")]
            return [result]
        if self.name == "verifier":
            candidate = _last(records, "candidate", "result")
            if candidate is None:
                raise ValueError("missing candidate; verifier cannot independently solve an unattempted workflow")
            good, checks = check_grounded_candidate(request, candidate, records, self.stores)
            if not good:
                raise ValueError("candidate failed independent public grounding or dependency checks")
            return [_artifact(request, "verification", candidate["value"], evidence=candidate["evidence"],
                              inputs=[candidate["id"]], verified=True, checks={**checks, "verification_performed": True})]
        raise ValueError("unknown frozen capability")


def build_workflow_agents(stores: PublicStores) -> dict[str, WorkflowAgent]:
    return {name: WorkflowAgent(name, stores) for name in AGENT_NAMES}
