import pytest

from conductor.metrics.statistics import conservative_claim_gate, paired_bootstrap


def test_paired_bootstrap_uses_tasks_and_preserves_no_difference():
    measured = paired_bootstrap([0, 1, 1], [0, 1, 1], samples=200, seed=7)
    assert measured["paired_tasks"] == 3
    assert measured["mean_delta"] == 0 and measured["delta_ci95"] == [0, 0]
    assert measured["ratio_ci95"] == [1, 1]
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap([float("nan")], [1])


def test_cost_claim_requires_quality_fingerprints_and_real_billing():
    evidence = {"paired_tasks": 40, "comparable_fingerprints": True,
                "success_delta_ci95": [0.0, 0.1], "cost_ratio_ci95": [0.7, 0.9],
                "unknown_cost_exclusions": 0}
    assert conservative_claim_gate(evidence)["status"] == "eligible"
    evidence["unknown_cost_exclusions"] = 1
    assert conservative_claim_gate(evidence)["status"] == "insufficient_evidence"
    evidence.update(unknown_cost_exclusions=0, success_delta_ci95=[-.1, .1])
    assert conservative_claim_gate(evidence)["status"] == "insufficient_evidence"


def test_nonfinite_or_reversed_intervals_cannot_pass_claim_gate():
    evidence = {"paired_tasks": 40, "comparable_fingerprints": True,
                "success_delta_ci95": [0.0, 0.1], "cost_ratio_ci95": [0.7, 0.9],
                "unknown_cost_exclusions": 0}
    for interval in ([float("nan"), 0.9], [0.7, float("inf")], [0.9, 0.7]):
        assert conservative_claim_gate({**evidence, "cost_ratio_ci95": interval})["status"] == "insufficient_evidence"
    assert conservative_claim_gate({**evidence, "success_delta_ci95": [float("nan"), .1]})["status"] == "insufficient_evidence"
    assert conservative_claim_gate({**evidence, "candidate_only_tasks": 1})["status"] == "insufficient_evidence"
    with pytest.raises(ValueError, match="thresholds"):
        conservative_claim_gate(evidence, noninferiority_margin=-.1)


def test_duplicate_task_observations_are_not_silently_selected():
    from conductor.metrics.aggregate import paired_differences
    row = {"task_id": "same", "policy": "baseline"}
    with pytest.raises(ValueError, match="Duplicate paired task"):
        paired_differences([row, row], "baseline", "candidate")


def test_unknown_retry_billing_overrides_provider_token_accounting():
    from conductor.metrics.aggregate import trajectory_metrics
    item = {"task": {"id": "one"}, "policy": "baseline", "metadata": {}, "steps": [
        {"controller_tokens": 0, "agent_outputs": [{"agent": "math", "tokens": 10,
         "metadata": {"token_accounting": "provider_usage", "token_usage_unknown": True}}]}]}
    assert trajectory_metrics(item)["cost_comparison_valid"] is False
    item["steps"][0]["agent_outputs"][0]["metadata"]["token_usage_unknown"] = False
    assert trajectory_metrics(item)["cost_comparison_valid"] is True
    item["metadata"]["inference_cost_known"] = False
    assert trajectory_metrics(item)["cost_comparison_valid"] is False


def test_checkpoint_digest_resolves_pointer_and_excludes_training_history(tmp_path):
    import json
    from conductor.evaluation.provenance import checkpoint_sha256
    root = tmp_path / "checkpoint"
    selected = root / "resume" / "step-1"
    selected.mkdir(parents=True)
    (root / "checkpoint_pointer.json").write_text(json.dumps({"path": "resume/step-1"}))
    (selected / "controller.json").write_text(json.dumps({"backend": "tiny", "stage": "sft"}))
    (selected / "model.pt").write_bytes(b"inference weights")
    before = checkpoint_sha256(root)
    (selected / "training_state.pt").write_bytes(b"RNG and optimizer state")
    (root / "optimizer.pt").write_bytes(b"unrelated history")
    older = root / "resume" / "step-0"
    older.mkdir()
    (older / "model.pt").write_bytes(b"older weights")
    assert checkpoint_sha256(root) == before == checkpoint_sha256(selected)
    (selected / "model.pt").write_bytes(b"different inference weights")
    assert checkpoint_sha256(root) != before


def test_zero_baseline_draws_cannot_be_discarded_to_claim_cost_savings():
    measured = paired_bootstrap([0.0] * 29 + [1.0], [0.01] * 29 + [0.5], samples=200, seed=7)
    assert measured["mean_ratio"] < 1
    assert measured["undefined_ratio_samples"] > 0
    assert measured["ratio_ci95"] is None
