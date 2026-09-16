"""Empirical task-paired bootstrap intervals and conservative claim eligibility."""
from __future__ import annotations

from typing import Any
import math

import numpy as np


def paired_bootstrap(left: list[float], right: list[float], *, seed: int = 42, samples: int = 2000,
                     confidence: float = 0.95) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"status": "unanswered", "reason": "No equal-length paired observations", "delta_ci95": None, "ratio_ci95": None}
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError("Bootstrap requires at least 100 samples and confidence in (0,1)")
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bootstrap observations must be finite")
    rng = np.random.default_rng(seed)
    # Chunk to bound bootstrap storage on research-scale task corpora.
    differences, ratios = [], []
    undefined_ratios = 0
    for offset in range(0, samples, 128):
        ids = rng.integers(0, len(a), size=(min(128, samples - offset), len(a)))
        means_a, means_b = a[ids].mean(1), b[ids].mean(1)
        differences.extend((means_b - means_a).tolist())
        valid = means_a > 0
        ratios.extend((means_b[valid] / means_a[valid]).tolist())
        # Discarding zero-denominator draws conditions the interval on trials
        # favorable to a cost reduction. Equal zero costs are neutral; a zero
        # baseline with nonzero candidate cost makes a finite ratio undefined.
        neutral = (means_a == 0) & (means_b == 0)
        ratios.extend(np.ones(int(neutral.sum())).tolist())
        undefined_ratios += int((~valid & ~neutral).sum())
    alpha = (1 - confidence) / 2
    interval = lambda values: [float(value) for value in np.quantile(values, [alpha, 1 - alpha])] if values else None
    return {"status": "measured", "paired_tasks": len(a), "mean_delta": float((b - a).mean()),
            "mean_ratio": float(b.mean() / a.mean()) if a.mean() > 0 else None,
            "delta_ci95": interval(differences), "ratio_ci95": None if undefined_ratios else interval(ratios),
            "undefined_ratio_samples": undefined_ratios, "bootstrap_samples": samples,
            "bootstrap_seed": seed, "confidence": confidence,
            "scope": "Empirical task-paired bootstrap; repeated samples of one task are not independent tasks."}


def conservative_claim_gate(comparison: dict[str, Any], *, noninferiority_margin: float = 0.0,
                            minimum_tasks: int = 30, cost_metric: str = "cost_usd") -> dict[str, Any]:
    if not math.isfinite(noninferiority_margin) or noninferiority_margin < 0 or minimum_tasks < 1:
        raise ValueError("Claim thresholds require a finite nonnegative margin and positive minimum task count")
    reasons = []
    quality = comparison.get("success_delta_ci95")
    interval_key = {"cost_usd": "cost_ratio_ci95", "total_tokens": "token_ratio_ci95", "agent_calls": "agent_calls_ratio_ci95",
                    "wall_clock_seconds": "latency_ratio_ci95"}.get(cost_metric)
    cost = comparison.get(interval_key) if interval_key else None
    def valid_interval(value: Any) -> bool:
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(bound, (int, float)) and not isinstance(bound, bool)
                        and math.isfinite(bound) for bound in value) and value[0] <= value[1])
    if comparison.get("paired_tasks", 0) < minimum_tasks:
        reasons.append(f"Fewer than {minimum_tasks} independent paired task IDs")
    if not comparison.get("comparable_fingerprints", False):
        reasons.append("Corpus, specialist, and experiment fingerprints are missing or inconsistent")
    if comparison.get("baseline_only_tasks", 0) or comparison.get("candidate_only_tasks", 0):
        reasons.append("Unmatched task coverage prevents endorsement of the complete held-out comparison")
    if not valid_interval(quality) or quality[0] < -noninferiority_margin:
        reasons.append("Quality noninferiority is not established by the lower confidence bound")
    if not valid_interval(cost) or cost[0] < 0 or cost[1] >= 1:
        reasons.append("Cost reduction is not established by the upper confidence bound")
    if cost_metric in {"cost_usd", "total_tokens"} and comparison.get("unknown_cost_exclusions") != 0:
        reasons.append("Unknown/proxy/heterogeneous token billing invalidates this cost comparison")
    return {"status": "eligible" if not reasons else "insufficient_evidence", "reasons": reasons,
            "cost_metric": cost_metric, "minimum_paired_tasks": minimum_tasks,
            "quality_noninferiority_margin": noninferiority_margin,
            "interpretation": "Eligibility supports only this predefined measured comparison, not a broader improvement claim."}
