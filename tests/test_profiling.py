import pytest

from conductor.inference.profiling import profile_stages
from conductor.inference.timing import timed_call
from conductor.inference.workloads import replay_states
from conductor.schema import ExecutionState


def test_cpu_stage_has_no_cuda_event_claim():
    value, timing = timed_call(lambda: 7, "cpu", gpu_stage=True)
    assert value == 7 and timing["host_seconds"] >= 0
    assert timing["cuda_event_seconds"] is None


def test_replay_uses_only_public_states_and_content_identity(tmp_path):
    import json
    source = tmp_path / "trajectories.jsonl"
    state = ExecutionState("Calculate 2+3", "math").to_dict()
    source.write_text(json.dumps({"task": {"id": "a", "expected_answer": "5"}, "steps": [{"state": state}]}) + "\n")
    states, identities, provenance = replay_states(source)
    assert states[0].to_dict() == state
    assert identities[0]["task_id"] == "a" and len(identities[0]["state_sha256"]) == 64
    assert provenance["kind"] == "trajectory_replay"
    assert "expected_answer" not in states[0].to_dict()
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="No trajectory"):
        replay_states(source)


def test_unsupported_stages_remain_unmeasured():
    assert profile_stages(object(), [ExecutionState("x", "math")], 1)["status"] == "unsupported"
