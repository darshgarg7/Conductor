from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from conductor.coordination.coverage import (audit_partitions, audit_preferences, audit_sft_records,
                                             preference_pair_sha256, write_immutable_audit)
from conductor.schema import ExecutionState, RoutingDecision


def record(task: str, group: str, family: str = "handoff", decision: RoutingDecision | None = None,
           step: int = 0, kinds: list[str] | None = None) -> dict:
    state = ExecutionState(f"request {task}", "coordination_workflow", current_step=step)
    return {"task_id": task, "source_group": group, "family": family, "state": state.to_dict(),
            "decision": (decision or RoutingDecision(["retriever"])).to_dict(), "state_kinds": kinds or []}


def test_coverage_counts_chosen_labels_not_candidate_availability() -> None:
    records = [record("a", "a"), record("a", "a", step=1, decision=RoutingDecision(["math", "coder"], "sequential"),
                                     kinds=["handoff", "partial"]),
               record("a", "a", step=2, decision=RoutingDecision([], terminate=True))]
    report = audit_sft_records(records)
    assert report["counts"]["count:0"] == 1 and report["counts"]["count:1"] == 1
    assert report["counts"]["count:2"] == 1 and "count:3" not in report["counts"]
    assert report["counts"]["state:handoff"] == 1 and report["counts"]["state:partial"] == 1
    assert report["within_action_handoffs"] == {"math>coder": 1}
    assert report["families"]["handoff"]["tasks"] == 1


def test_group_disjoint_partition_and_family_floor() -> None:
    report = audit_partitions([record("a", "source-a")], [record("b", "source-b")], required_cells=["count:1"])
    assert report["coverage_gate_passed"]
    with pytest.raises(ValueError, match="source-group overlap"):
        audit_partitions([record("a", "same")], [record("b", "same")])
    with pytest.raises(ValueError, match="task overlap"):
        audit_partitions([record("a", "source-a")], [record("a", "source-b")])
    with pytest.raises(ValueError, match="missing required families"):
        audit_partitions([record("a", "source-a")], [record("b", "source-b", family="other")])
    with pytest.raises(ValueError, match="too few distinct source groups"):
        audit_partitions([record("a", "source-a")], [record("b", "source-b")], minimum_dev_groups=2)


def test_empty_multi_agent_label_cell_blocks_claimed_coverage() -> None:
    with pytest.raises(ValueError, match="count:3"):
        audit_partitions([record("a", "a")], [record("b", "b")], required_cells=["count:3"])


def test_identical_public_state_conflicting_targets_are_rejected() -> None:
    first = record("a", "a")
    second = copy.deepcopy(first)
    second["decision"] = RoutingDecision(["math"]).to_dict()
    report = audit_sft_records([first, second])
    assert report["contradictory_public_state_labels"] == 1
    with pytest.raises(ValueError, match="contradictory"):
        audit_partitions([first, second], [record("b", "b")])


def preference() -> dict:
    value = record("a", "a")
    value.pop("decision")
    value.update({"chosen": RoutingDecision(["retriever", "math"], "sequential").to_dict(),
                  "rejected": RoutingDecision(["math", "retriever"], "sequential").to_dict(),
                  "continuation_sha256": "1" * 64, "preference_source": "on_policy_mistake"})
    return value


def test_ordered_preferences_and_duplicates_are_disclosed() -> None:
    value = preference()
    report = audit_preferences([value, copy.deepcopy(value)])
    assert report["canonical_distinct_actions_validated"] and report["duplicate_pairs"] == 1
    assert report["sources"] == {"on_policy_mistake": 2}
    assert report["chosen_agent_counts"] == {"2": 2}
    assert report["chosen_multiagent_modes"] == {"sequential": 2}


def test_stored_preference_identity_uses_serialized_public_state() -> None:
    value = preference()
    value["pair_sha256"] = preference_pair_sha256(
        value["state"], RoutingDecision(**value["chosen"]), RoutingDecision(**value["rejected"]),
        value["continuation_sha256"])
    assert audit_preferences([value])["unique_pairs"] == 1
    value["pair_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stored preference pair identity"):
        audit_preferences([value])


def test_parallel_reordering_is_not_a_real_preference() -> None:
    value = preference()
    value["chosen"]["execution_mode"] = "parallel"
    value["rejected"]["execution_mode"] = "parallel"
    with pytest.raises(ValueError, match="same canonical"):
        audit_preferences([value])


def test_preference_branches_must_share_public_state_and_continuation() -> None:
    value = preference()
    value["rejected_state"] = copy.deepcopy(value["state"])
    value["rejected_state"]["current_step"] += 1
    with pytest.raises(ValueError, match="same public state"):
        audit_preferences([value])
    value.pop("rejected_state")
    value["rejected_continuation_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="different continuation"):
        audit_preferences([value])


def test_illegal_supervision_rejected_before_training() -> None:
    value = record("a", "a", decision=RoutingDecision(["retriever", "math"], "sequential"))
    value["state"]["remaining_budget"]["agent_calls"] = 1
    with pytest.raises(ValueError, match="budget support"):
        audit_sft_records([value])


def test_offline_identity_is_mandatory() -> None:
    value = record("a", "a")
    del value["source_group"]
    with pytest.raises(ValueError, match="offline task_id"):
        audit_sft_records([value])


def test_audit_bytes_sealed_and_existing_file_preserved(tmp_path: Path) -> None:
    destination = tmp_path / "coverage.json"
    digest = write_immutable_audit(destination, audit_sft_records([record("a", "a")]))
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        write_immutable_audit(destination, {})
    assert destination.read_bytes() == before
