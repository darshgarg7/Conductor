"""Descriptive routing concentration with explicit task, run and event units."""
from __future__ import annotations

import copy
import json

import pytest

from conductor.evaluation.diagnostics import routing_diagnostics
from conductor.schema import AgentOutput, ExecutionState, RoutingDecision, StepRecord, Task, Trajectory


def trajectory(task_id: str, category: str, agents: list[str] | None = None, *,
               success: bool = True, mode: str = "parallel", policy: str = "candidate") -> dict:
    agents = agents or ["math"]
    task = Task(task_id, f"Independent public fixture {task_id}", category, "eval", "correct")
    outputs = [AgentOutput(agent, "observed evidence", 1, .001).to_dict() for agent in agents]
    steps = [StepRecord(ExecutionState(task.user_task, category, current_step=0).to_dict(),
                        RoutingDecision(agents, mode).to_dict(), outputs),
             StepRecord(ExecutionState(task.user_task, category, current_step=1).to_dict(),
                        RoutingDecision([], terminate=True).to_dict(), [])]
    return Trajectory(task.__dict__, policy, steps, "correct" if success else "incorrect",
                      success, float(success), .01).to_dict()


def summary(items: list[dict], policy: str = "candidate") -> dict:
    return routing_diagnostics(items)["policies"][policy]


def test_one_justified_shared_route_can_solve_every_task_without_an_automatic_failure_gate() -> None:
    data = [trajectory("a", "math"), trajectory("b", "reasoning")]
    result = summary(data)
    assert result["constant_first_action"] and result["unique_first_actions"] == 1
    assert result["exact_success_rate"] == result["recorded_success_rate"] == 1
    assert result["first_action_entropy_bits"] == 0
    assert result["task_type_first_action_mutual_information_bits"] == 0
    warning = next(item for item in result["warnings"] if item["code"] == "constant_first_action")
    assert warning["severity"] == "descriptive" and "justified shared route" in warning["message"]
    assert "model_failed" not in result and "collapse_failure" not in result


def test_low_success_is_reported_separately_from_constant_first_action() -> None:
    result = summary([trajectory("a", "math", ["coder"], success=False),
                      trajectory("b", "retrieval", ["coder"], success=False)])
    assert result["constant_first_action"] and result["exact_success_rate"] == 0
    assert result["termination_by_step"][1]["requested_termination_frequency_given_fresh_route"] == 1
    assert result["categories"]["math"]["exact_success_rate"] == 0
    assert result["agent_selection_distribution"]["coder"]["task_called_probability"] == 1


def test_repeated_runs_do_not_masquerade_as_extra_independent_tasks() -> None:
    first = trajectory("a", "math", ["math"])
    second = trajectory("b", "retrieval", ["retriever"], success=False)
    original = summary([first, second])
    repeated = summary([copy.deepcopy(first) for _ in range(50)] + [second])
    assert repeated["trajectory_count"] == 51 and repeated["independent_task_count"] == 2
    assert repeated["runs_per_task"] == {"a": 50, "b": 1}
    for field in ("recorded_success_rate", "exact_success_rate", "first_action_entropy_bits",
                  "task_type_first_action_mutual_information_bits", "action_entropy_bits",
                  "mean_actual_agent_activations"):
        assert repeated[field] == pytest.approx(original[field])
    assert repeated["task_type_first_action_mutual_information_bits"] == pytest.approx(1)
    assert repeated["normalized_task_type_first_action_mutual_information"] == pytest.approx(1)
    assert repeated["agent_selection_distribution"]["math"]["selection_share"] == pytest.approx(.5)
    assert repeated["categories"]["math"]["independent_task_count"] == 1
    assert repeated["categories"]["retrieval"]["exact_success_rate"] == 0


def test_within_task_randomness_is_averaged_in_first_action_mutual_information() -> None:
    result = summary([trajectory("a", "math", ["math"]), trajectory("a", "math", ["retriever"]),
                      trajectory("b", "retrieval", ["math"]), trajectory("b", "retrieval", ["retriever"])])
    assert result["independent_task_count"] == 2 and result["trajectory_count"] == 4
    assert result["first_action_entropy_bits"] == pytest.approx(1)
    assert result["task_type_first_action_mutual_information_bits"] == pytest.approx(0)
    assert not result["constant_first_action"]


def test_unique_arbitrary_actions_do_not_turn_maximal_plugin_mi_into_specialization_evidence() -> None:
    items = [trajectory(str(index), ("math" if index < 3 else "retrieval"), [agent])
             for index, agent in enumerate(("math", "planner", "coder", "retriever", "critic", "verifier"))]
    result = summary(items)
    null = result["first_action_association_permutation_null"]
    assert result["task_type_first_action_mutual_information_bits"] == pytest.approx(1)
    assert null["status"] == "measured" and null["independent_task_count"] == 6
    assert null["mean_permuted_mutual_information_bits"] == pytest.approx(1)
    assert null["excess_over_permuted_mean_bits"] == pytest.approx(0)
    assert null["p_value_ge_observed"] == 1
    assert "does not establish learning" in null["scope"]


def test_permutation_null_preserves_independent_task_units_and_is_seed_reproducible() -> None:
    first = trajectory("a", "math", ["math"])
    other = trajectory("b", "retrieval", ["retriever"])
    original = summary([first, other])["first_action_association_permutation_null"]
    repeated = summary([copy.deepcopy(first) for _ in range(30)] + [other])["first_action_association_permutation_null"]
    assert original["independent_task_count"] == repeated["independent_task_count"] == 2
    assert original["mean_permuted_mutual_information_bits"] == pytest.approx(repeated["mean_permuted_mutual_information_bits"])
    assert original["p_value_ge_observed"] == repeated["p_value_ge_observed"]
    assert summary([first, other])["first_action_association_permutation_null"] == original


def test_permutation_association_remains_descriptive_when_type_actions_are_consistent() -> None:
    items = [trajectory(str(index), "math" if index < 4 else "retrieval", ["math" if index < 4 else "retriever"])
             for index in range(8)]
    null = summary(items)["first_action_association_permutation_null"]
    assert null["mean_permuted_mutual_information_bits"] < null["observed_mutual_information_bits"]
    assert 0 < null["p_value_ge_observed"] < 1
    assert "Exchangeability is an assumption" in null["scope"]


def test_permutation_work_can_be_disabled_and_invalid_controls_fail_closed() -> None:
    item = trajectory("a", "math")
    assert routing_diagnostics([item], permutation_samples=0)["policies"]["candidate"][
        "first_action_association_permutation_null"]["status"] == "unavailable"
    for arguments in ({"seed": -1}, {"seed": True}, {"permutation_samples": -1}, {"permutation_samples": 1.5}):
        with pytest.raises(ValueError, match="nonnegative integers"):
            routing_diagnostics([item], **arguments)


def test_parallel_order_is_canonical_but_sequential_order_changes_the_action() -> None:
    first = trajectory("a", "math", ["math", "verifier"])
    second = trajectory("b", "retrieval", ["verifier", "math"])
    parallel = summary([first, second])
    assert parallel["unique_first_actions"] == 1 and parallel["constant_first_action"]
    first["steps"][0]["decision"]["execution_mode"] = "sequential"
    second["steps"][0]["decision"]["execution_mode"] = "sequential"
    sequential = summary([first, second])
    assert sequential["unique_first_actions"] == 2 and not sequential["constant_first_action"]
    assert sequential["task_type_first_action_mutual_information_bits"] == pytest.approx(1)


def test_requested_actions_and_actual_dispatch_are_not_conflated_after_budget_clipping() -> None:
    item = trajectory("a", "math", success=False)
    item["steps"] = [item["steps"][0]]
    item["metadata"]["requested_routing_decisions"] = [RoutingDecision(["math"]).to_dict()]
    item["steps"][0]["decision"] = RoutingDecision([], terminate=True).to_dict()
    item["steps"][0]["agent_outputs"] = []
    result = summary([item])
    assert result["first_action_distribution"][0]["action"]["selected_agents"] == ["math"]
    assert result["mean_actual_agent_activations"] == 0
    assert result["requested_action_run_coverage"] == 1
    stop = result["termination_by_step"][0]
    assert stop["requested_termination_frequency_given_fresh_route"] == 0
    assert stop["execution_termination_frequency_given_visit"] == 1


def test_reused_dispatches_count_as_activations_but_not_new_routing_decisions() -> None:
    item = trajectory("a", "math")
    repeated = copy.deepcopy(item["steps"][0])
    repeated["routing_reused"] = True
    repeated["state"]["current_step"] = 1
    item["steps"][1]["state"]["current_step"] = 2
    item["steps"].insert(1, repeated)
    result = summary([item])
    assert result["raw_fresh_routing_decisions"] == 2 and result["raw_reused_decisions"] == 1
    assert result["unique_routing_actions"] == 2 and result["mean_actual_agent_activations"] == 2
    assert result["termination_by_step"][1]["requested_termination_frequency_given_fresh_route"] is None
    assert result["agent_selection_distribution"]["math"]["task_normalized_route_selection_probability"] == .5


def test_long_failed_trajectories_do_not_gain_extra_action_frequency_weight() -> None:
    short = trajectory("a", "math", ["math"])
    long = trajectory("b", "retrieval", ["retriever"], success=False)
    first = long["steps"][0]
    long["steps"] = [copy.deepcopy(first) for _ in range(9)] + [long["steps"][1]]
    for index, step in enumerate(long["steps"]):
        step["state"]["current_step"] = index
    result = summary([short, long])
    mass = {tuple(row["action"]["selected_agents"]): row["task_weighted_mass"] for row in result["action_distribution"]}
    assert mass[("math",)] == pytest.approx(.5)
    assert mass[("retriever",)] == pytest.approx(.9)
    assert mass[()] == pytest.approx(.6)
    assert sum(mass.values()) == pytest.approx(2)
    stop = result["termination_by_step"][1]
    assert stop["requested_termination_frequency_given_fresh_route"] == .5
    assert result["termination_by_step"][9]["visitation_rate"] == .5
    assert result["mean_actual_agent_activations"] == 5


def test_missing_routes_are_visible_and_prevent_unsupported_mutual_information() -> None:
    first = trajectory("a", "math")
    missing = copy.deepcopy(first)
    missing["steps"] = []
    missing["final_answer"] = "incorrect"
    missing["task_success"] = False
    result = summary([first, missing])
    assert result["missing_first_action_task_fraction"] == .5
    assert result["task_type_first_action_mutual_information_bits"] is None
    assert not result["constant_first_action"]
    assert any(row["missing_routing_decision"] for row in result["first_action_distribution"])
    assert result["exact_success_rate"] == .5


def test_private_labels_only_verify_offline_success_and_disagreements_are_visible() -> None:
    item = trajectory("a", "math", success=False)
    item["task_success"] = True
    result = summary([item])
    assert result["recorded_success_rate"] == 1 and result["exact_success_rate"] == 0
    assert result["reported_success_disagreements"] == 1
    assert any(row["code"] == "success_record_disagreement" for row in result["warnings"])
    assert "expected_answer" not in json.dumps(result)


def test_missing_grading_evidence_does_not_become_verified_exact_success() -> None:
    item = trajectory("a", "math")
    del item["task"]["expected_answer"]
    result = summary([item])
    assert result["recorded_success_rate"] == 1 and result["exact_success_rate"] is None
    assert result["exact_grading_coverage"] == 0
    assert result["categories"]["math"]["exact_success_rate"] is None


@pytest.mark.parametrize("field,value", [("task_type", "changed"), ("user_task", "changed"),
                                        ("expected_answer", "changed"), ("split", "train")])
def test_inconsistent_repeated_task_identity_is_rejected(field: str, value: str) -> None:
    first = trajectory("a", "math")
    second = copy.deepcopy(first)
    second["task"][field] = value
    with pytest.raises(ValueError, match="inconsistent"):
        routing_diagnostics([first, second])


def test_renaming_identical_prompts_cannot_inflate_independent_task_units() -> None:
    first = trajectory("a", "math")
    second = copy.deepcopy(first)
    second["task"]["id"] = "renamed"
    with pytest.raises(ValueError, match="extra independent"):
        routing_diagnostics([first, second])


def test_policy_groups_do_not_pool_outcomes_and_mixed_conditions_are_flagged() -> None:
    first = trajectory("a", "math")
    second = trajectory("b", "retrieval")
    first["metadata"]["checkpoint_sha256"] = "checkpoint-a"
    second["metadata"]["checkpoint_sha256"] = "checkpoint-b"
    other = trajectory("a", "math", success=False, policy="baseline")
    result = routing_diagnostics([first, second, other])
    assert result["policies"]["candidate"]["exact_success_rate"] == 1
    assert result["policies"]["baseline"]["exact_success_rate"] == 0
    assert result["policies"]["candidate"]["mixed_conditions"]
    assert any(row["code"] == "mixed_conditions" for row in result["policies"]["candidate"]["warnings"])


def test_noninitial_routes_are_not_advertised_as_matched_initial_state_probes() -> None:
    item = trajectory("a", "math")
    for index, step in enumerate(item["steps"]):
        step["state"]["current_step"] = index + 5
    result = summary([item])
    assert result["first_route_step_distribution"] == [{"step": 5, "task_weighted_mass": 1}]
    assert any(row["code"] == "noninitial_first_routes" for row in result["warnings"])


@pytest.mark.parametrize("mutation", ["duplicate_steps", "boolean_step", "unknown_agent", "unaligned_requests", "nonboolean_success"])
def test_corrupt_units_or_decisions_are_rejected(mutation: str) -> None:
    item = trajectory("a", "math")
    if mutation == "duplicate_steps":
        item["steps"][1]["state"]["current_step"] = 0
    elif mutation == "boolean_step":
        item["steps"][0]["state"]["current_step"] = True
    elif mutation == "unknown_agent":
        item["steps"][0]["agent_outputs"][0]["agent"] = "not-a-specialist"
    elif mutation == "unaligned_requests":
        item["metadata"]["requested_routing_decisions"] = []
    else:
        item["task_success"] = 1
    with pytest.raises(ValueError):
        routing_diagnostics([item])


def test_empty_data_and_dataclass_records_remain_json_serializable_without_fake_metrics() -> None:
    empty = routing_diagnostics([])
    assert empty["policies"] == {}
    record = Trajectory.from_dict(trajectory("a", "math"))
    result = routing_diagnostics([record])
    assert result["policies"]["candidate"]["exact_success_rate"] == 1
    json.dumps(result, allow_nan=False)


def test_evaluation_and_analysis_emit_matching_diagnostics_from_actual_baseline_trajectories(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from dataclasses import asdict
    from conductor.analyze import analyze
    from conductor.datasets.io import write_jsonl
    from conductor.datasets.tasks import make_tasks
    from conductor.evaluate import evaluate

    data = tmp_path / "tasks.jsonl"
    write_jsonl(data, [asdict(task) for task in make_tasks(train_count=4, eval_count=6)])
    evaluation = tmp_path / "evaluation"
    config = {"data": str(data), "output": str(evaluation), "seed": 71,
              "policies": ["all_agent", "rule_based"], "agents": {"backend": "deterministic"},
              "routing_diagnostics": {"permutation_samples": 23}, "bootstrap_samples": 100}
    result = asyncio.run(evaluate(config))
    saved = json.loads((evaluation / "routing_diagnostics.json").read_text())
    assert saved == result["routing_diagnostics"]
    assert saved["policies"]["rule_based"]["independent_task_count"] == 6
    assert saved["policies"]["rule_based"]["exact_success_rate"] == 1
    assert saved["policies"]["all_agent"]["constant_first_action"]
    permutation = saved["policies"]["rule_based"]["first_action_association_permutation_null"]
    assert permutation["seed"] == 71 and permutation["permutation_samples"] == 23
    monkeypatch.setattr("conductor.analyze.plot_results", lambda *args: [])
    target = tmp_path / "analysis"
    analyzed = analyze(evaluation, target)
    assert analyzed["routing_diagnostics"] == saved
    assert json.loads((target / "routing_diagnostics.json").read_text()) == saved
