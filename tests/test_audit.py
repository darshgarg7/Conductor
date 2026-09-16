"""Evidence checks reject damaged journals and unsupported engineering claims."""
from __future__ import annotations

import json
from pathlib import Path

from conductor.audit import RESUME_CLAIMS, audit, audit_checkpoint, audit_dataset, audit_run
from conductor.datasets.integrity import seal


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def trajectory(identifier: str = "task-1") -> dict[str, object]:
    return {"task": {"id": identifier, "split": "train"}, "policy": "rule_based",
            "task_success": True, "metadata": {"source_kind": "synthetic_development_templates"}}


def test_checksums_and_training_counts_are_observed(tmp_path: Path) -> None:
    write(tmp_path / "trajectories.jsonl", seal(trajectory(), "record-1"))
    result = audit_dataset(tmp_path)
    assert result["status"] == "valid"
    assert result["counts"]["trajectories"] == 1
    assert result["verified_training_trajectories"] == 1
    assert result["successes"] == 1
    assert result["source_kinds"] == ["synthetic_development_templates"]


def test_corrupt_checksum_and_incomplete_tail_are_never_repaired(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    record = seal(trajectory(), "record-1")
    record["task_success"] = False
    write(path, record)
    before = path.read_bytes()
    assert audit_dataset(tmp_path)["status"] == "invalid"
    assert path.read_bytes() == before
    path.write_bytes(json.dumps(seal(trajectory(), "record-1")).encode())
    before = path.read_bytes()
    assert audit_dataset(tmp_path)["status"] == "invalid"
    assert path.read_bytes() == before


def test_duplicate_journals_across_shards_invalidate_count_claim(tmp_path: Path) -> None:
    for shard in ("a", "b"):
        write(tmp_path / shard / "trajectories.jsonl", seal(trajectory(), "record-1"))
    result = audit(datasets=[str(tmp_path)], claims=[{"kind": "training_trajectory_count", "minimum": 1}])
    assert result["invalid_artifacts"] == 1
    assert result["unsupported_claims"] == 1


def test_random_tiny_checkpoint_cannot_support_pretrained_claim(tmp_path: Path) -> None:
    write(tmp_path / "controller.json", {"backend": "tiny", "stage": "sft", "pretrained": False})
    (tmp_path / "model.pt").write_bytes(b"fixture weight existence only")
    result = audit(checkpoints=[str(tmp_path)], claims=[{"kind": "pretrained_post_training", "stages": ["sft"]}])
    assert result["checkpoints"][0]["pretrained"] is False
    assert result["unsupported_claims"] == 1


def test_pretrained_checkpoint_requires_completed_training_and_pinned_identity(tmp_path: Path) -> None:
    write(tmp_path / "controller.json", {"backend": "hf", "stage": "sft", "pretrained": True,
          "resolved_revision": "a" * 40, "base_model": "fixture-pretrained-moe"})
    (tmp_path / "head.pt").write_bytes(b"fixture weight existence only")
    assert not audit_checkpoint(tmp_path)["completed_pretrained_training_observed"]
    write(tmp_path / "run.json", {"git_commit": "b" * 40, "git_dirty": False, "seed": 42,
          "runtime_seconds": 1.0, "hardware": {"torch": "fixture"}, "configuration": {"model": {"backend": "hf"}},
          "metrics": {"completed": True, "optimizer_windows": 1, "dataset_sha256": "c" * 64,
          "pretrained": True, "specialists_updated": False, "stage": "sft"}})
    assert audit_checkpoint(tmp_path)["completed_pretrained_training_observed"]
    metadata = json.loads((tmp_path / "controller.json").read_text())
    metadata["resolved_revision"] = "main"
    write(tmp_path / "controller.json", metadata)
    assert not audit_checkpoint(tmp_path)["completed_pretrained_training_observed"]


def test_cuda_probe_cannot_be_mistaken_for_model_validation(tmp_path: Path) -> None:
    path = tmp_path / "doctor.json"
    write(path, {"status": "compatible", "validation": {"device": "cuda:0"}, "probe": {"finite": True}})
    result = audit(doctors=[str(path)], claims=list(RESUME_CLAIMS))
    assert result["nvidia_model_validation"] == "absent"
    assert result["unsupported_claims"] == 4


def test_independent_inventories_are_not_summed_to_manufacture_scale(tmp_path: Path) -> None:
    for directory in ("a", "b"):
        write(tmp_path / directory / "trajectories.jsonl", seal(trajectory(), "record-1"))
    result = audit(datasets=[str(tmp_path / "a"), str(tmp_path / "b")],
                   claims=[{"kind": "training_trajectory_count", "minimum": 2}])
    assert result["claims"][0]["observed"] == 1
    assert result["unsupported_claims"] == 1


def test_malformed_run_metrics_and_nonfinite_data_fail_closed(tmp_path: Path) -> None:
    write(tmp_path / "run.json", {"metrics": [], "runtime_seconds": 1})
    assert audit_run(tmp_path)["status"] == "invalid"
    (tmp_path / "trajectories.jsonl").write_text(' {"task": {"id": "x"}, "task_success": true, "score": NaN}\n')
    assert audit_dataset(tmp_path)["status"] == "invalid"
