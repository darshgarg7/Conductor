from __future__ import annotations

import pytest

from conductor.preference.reward import reward
from conductor.schema import StepRecord, Trajectory


def trajectory(success: bool, tokens: int = 100, calls: int = 1, latency: float = 0.1) -> Trajectory:
    return Trajectory({}, "test", [StepRecord({}, {}, [{"tokens": tokens} for _ in range(calls)], controller_tokens=10)],
                      "answer", success, float(success), latency, [{"source": "a", "target": "b"}])


def test_reward_respects_success_and_normalized_cost_tradeoff() -> None:
    good, costly = trajectory(True), trajectory(True, 500, 4, 1.0)
    assert reward(good) > reward(costly) > reward(trajectory(False))
    expected = 1 - .05 * 110 / 1000 - .02 * .1 - .04 / 12 - .01 / 20
    assert reward(good) == pytest.approx(expected)
    assert reward(good, {"token_weight": 0, "latency_weight": 0, "agent_call_weight": 0, "communication_weight": 0}) == 1
    with pytest.raises(ValueError, match="positive"):
        reward(good, {"token_scale": 0})
