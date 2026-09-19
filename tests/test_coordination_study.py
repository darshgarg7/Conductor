from __future__ import annotations

from conductor.coordination.study import select_primary_baseline


def metric(success: float, calls: float, latency: float) -> dict[str, float]:
    return {"success_rate": success, "family_balanced_success": success,
            "minimum_family_success": success, "minimum_dependency_stage_success": success,
            "mean_agent_calls": calls, "mean_wall_clock_seconds": latency}


def test_primary_baseline_selection_uses_frozen_quality_cost_latency_order() -> None:
    per_seed = {seed: {
        "rules": metric(1, 4, .04),
        "linear": metric(.96, 3, .02),
        "mlp": metric(1, 4, .03),
    } for seed in (42, 137, 2027)}
    result = select_primary_baseline(per_seed, .95)
    assert result["status"] == "locked" and result["selected"] == "mlp"
    per_seed[137]["mlp"] = metric(.94, 2, .01)
    result = select_primary_baseline(per_seed, .95)
    assert result["selected"] == "rules"


def test_primary_baseline_selection_blocks_when_floor_is_unmet() -> None:
    result = select_primary_baseline({42: {"linear": metric(.8, 2, .01)}}, .95)
    assert result["status"] == "blocked" and "selected" not in result
