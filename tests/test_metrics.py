from conductor.metrics.aggregate import aggregate_metrics, paired_differences, trajectory_metrics


def trajectory(policy="first", success=True):
    return {"task": {"id": "one", "task_type": "math", "split": "eval", "metadata": {}},
            "policy": policy, "steps": [
                {"agent_outputs": [{"agent": "math", "tokens": 7, "cost_usd": 0.2}],
                 "controller_tokens": 3, "controller_latency_seconds": 0.1, "controller_cost_usd": 0.05},
                {"agent_outputs": [], "controller_tokens": 2, "controller_latency_seconds": 0.05}],
            "wall_clock_latency": 0.5, "task_success": success, "grader_score": float(success),
            "communication_graph": [{"source": "controller", "target": "math", "message_bytes": 11,
                                     "estimated_message_tokens": 3}]}


def test_observed_cost_tokens_and_termination_are_separate():
    row = trajectory_metrics(trajectory())
    assert row["controller_tokens"] == 5
    assert row["downstream_tokens"] == 7
    assert row["total_tokens"] == 12
    assert row["agent_calls"] == 1
    assert row["rounds"] == 1
    assert row["routing_calls"] == 2
    assert abs(row["cost_usd"] - 0.25) < 1e-9
    assert abs(row["controller_overhead_fraction"] - 0.3) < 1e-9
    assert row["activation_sparsity"] == 7 / 8
    assert row["communication_message_bytes"] == 11
    assert 0 <= row["communication_sparsity"] <= 1


def test_paired_comparison_excludes_unmatched_tasks():
    first = trajectory_metrics(trajectory())
    second = trajectory_metrics(trajectory("second", False))
    unmatched = {**second, "task_id": "two", "total_tokens": 999}
    result = paired_differences([first, second, unmatched], "first", "second")
    assert result["paired_tasks"] == 1
    assert result["candidate_only_tasks"] == 1
    assert result["mean_delta_task_success"] == -1
    assert result["mean_delta_total_tokens"] == 0
    assert paired_differences([], "first", "second")["status"] == "unanswered"


def test_missing_overhead_stays_null():
    item = trajectory()
    item["wall_clock_latency"] = 0
    rows = [trajectory_metrics(item)]
    assert rows[0]["controller_overhead_fraction"] is None
    assert aggregate_metrics(rows)[0]["mean_controller_overhead_fraction"] is None
