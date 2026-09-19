from __future__ import annotations

import pytest
import torch

from conductor.coordination.study import _adapter_update_report, select_primary_baseline


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


def test_adapter_update_report_requires_each_declared_target(tmp_path) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file
    before, after = tmp_path / "before" / "adapter", tmp_path / "after" / "adapter"
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    names = [f"base.{target}.lora_B.weight" for target in ("q_proj", "v_proj", "router")]
    save_file({name: torch.zeros(2, 2) for name in names}, before / "adapter_model.safetensors")
    save_file({name: torch.ones(2, 2) for name in names}, after / "adapter_model.safetensors")
    report = _adapter_update_report(before.parent, after.parent)
    assert report["passed"] and report["nonzero_lora_b_tensors"] == 3
