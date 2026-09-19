"""Private outcome grading backed by independently checked public evidence.

The answer key is used only here.  Specialist execution and public grounding
checks operate on the request, execution artifacts, and frozen public stores.
"""
from __future__ import annotations

import json
from typing import Any

from conductor.schema import ExecutionState, Task


_GROUNDING_CHECKS = ("source_grounded", "result_matches_request", "dependencies_valid")


def assemble_workflow_answer(*args: Any, **kwargs: Any) -> str:
    """Expose the public answer assembler without passing it a private Task."""
    from .specialists import assemble_workflow_answer as assemble

    return assemble(*args, **kwargs)


def _source_ids(value: Any) -> set[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return set(value)


def _supporting_ids(candidate: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> set[str]:
    """Follow public artifact dependencies without consulting producer names."""
    pending = list(candidate.get("inputs", []))
    found: set[str] = set()
    while pending:
        identifier = pending.pop()
        if not isinstance(identifier, str) or identifier in found:
            continue
        record = by_id.get(identifier)
        if record is None:
            continue
        found.add(identifier)
        inputs = record.get("inputs", [])
        if isinstance(inputs, list):
            pending.extend(inputs)
    return found


def _matching_proof(
    candidate: dict[str, Any],
    records: list[dict[str, Any]],
    public_checks: dict[str, Any],
) -> dict[str, Any] | None:
    """Require a proof linked to a candidate whose claims were recomputed."""
    by_id = {record["id"]: record for record in records}
    supporting = _supporting_ids(candidate, by_id)
    sources = _source_ids(candidate.get("evidence"))
    for proof in records:
        if proof.get("kind") not in {"verification", "execution"}:
            continue
        if proof.get("verified") is not True or proof.get("value") != candidate.get("value"):
            continue
        inputs = proof.get("inputs")
        if not isinstance(inputs, list) or candidate["id"] not in inputs:
            continue
        if _source_ids(proof.get("evidence")) != sources:
            continue
        checks = proof.get("checks")
        if not isinstance(checks, dict) or checks.get("verification_performed") is not True:
            continue
        if any(
            public_checks.get(name) is not True or checks.get(name) is not public_checks.get(name)
            for name in _GROUNDING_CHECKS
        ):
            continue
        if proof["kind"] == "execution":
            if checks.get("program_executed") is not True or checks.get("assertions_passed") is not True:
                continue
            if not any(
                isinstance(identifier, str)
                and identifier in supporting
                and by_id[identifier].get("kind") == "program"
                for identifier in inputs
            ):
                continue
        return proof
    return None


def grade_workflow(
    task: Task, state: ExecutionState, final_answer: str
) -> tuple[bool, float, dict[str, Any]]:
    """Grade answer correctness and its public workflow proof, not its route."""
    from .specialists import artifacts, check_grounded_candidate
    from .workloads import PublicStores, parse_request

    details: dict[str, Any] = {
        "answer_correct": False,
        "grounded_candidate": False,
        "evidence_matches": False,
        "verification_required": False,
        "verification_present": False,
    }

    def reject(reason: str) -> tuple[bool, float, dict[str, Any]]:
        details["reason"] = reason
        return False, 0.0, details

    if state.user_task != task.user_task or state.task_type != task.task_type:
        return reject("execution_request_mismatch")
    try:
        final = json.loads(final_answer)
    except (TypeError, ValueError):
        return reject("invalid_final_json")
    if (
        not isinstance(final, dict)
        or not isinstance(final.get("result"), str)
        or type(final.get("verified")) is not bool
        or _source_ids(final.get("evidence")) is None
    ):
        return reject("invalid_final_schema")

    details["answer_correct"] = final["result"] == task.expected_answer
    if not details["answer_correct"]:
        return reject("answer_mismatch")
    try:
        grading = task.metadata["grading"]
        stores = PublicStores.from_dict(grading["public_stores"])
        request = parse_request(state)
        records = artifacts(state)
    except (AttributeError, KeyError, TypeError, ValueError):
        return reject("invalid_grading_snapshot_or_public_state")
    if not isinstance(request, dict):
        return reject("invalid_public_request")

    details["verification_required"] = request.get("verification") is True
    if details["verification_required"] and not final["verified"]:
        return reject("required_verification_not_claimed")
    need_proof = details["verification_required"] or final["verified"]
    final_sources = _source_ids(final["evidence"])
    for candidate in records:
        if candidate.get("kind") != "candidate" or candidate.get("value") != final["result"]:
            continue
        try:
            grounded, public_checks = check_grounded_candidate(request, candidate, records, stores)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if not grounded or not isinstance(public_checks, dict):
            continue
        if any(public_checks.get(name) is not True for name in _GROUNDING_CHECKS):
            continue
        details["grounded_candidate"] = True
        if _source_ids(candidate.get("evidence")) != final_sources:
            continue
        details["evidence_matches"] = True
        proof = _matching_proof(candidate, records, public_checks)
        if need_proof and proof is None:
            continue
        details["verification_present"] = proof is not None
        details["candidate_id"] = candidate["id"]
        if proof is not None:
            details["verification_id"] = proof["id"]
        details["reason"] = "correct_grounded_workflow"
        return True, 1.0, details

    if not details["grounded_candidate"]:
        return reject("no_grounded_candidate")
    if not details["evidence_matches"]:
        return reject("final_source_evidence_mismatch")
    return reject("missing_valid_verification_proof")
