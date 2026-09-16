"""Audit artifact consistency and claim support without creating experiment metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from conductor.controller.artifacts import resolve_checkpoint
from conductor.datasets.integrity import file_digest, record_checksum
from conductor.utils.runs import write_json

def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _error(path: Path, error: Exception) -> dict[str, Any]:
    return {"path": str(path.resolve()), "status": "invalid", "reason": str(error)}


def audit_dataset(path: str | Path) -> dict[str, Any]:
    """Read journals without acquiring write locks or repairing a damaged tail."""
    directory = Path(path)
    names = {"tasks.jsonl", "trajectories.jsonl", "sft.jsonl", "preferences.jsonl"}
    files = [directory] if directory.is_file() else sorted(p for p in directory.rglob("*.jsonl") if p.name in names)
    result: dict[str, Any] = {"path": str(directory.resolve()), "status": "valid", "files": [],
                              "counts": {}, "training_trajectories": 0, "verified_training_trajectories": 0,
                              "successes": 0, "failures": 0, "source_kinds": []}
    if not files:
        return {**result, "status": "invalid", "reason": "No task/trajectory/SFT/preference JSONL artifacts found"}
    counts: Counter[str] = Counter()
    seen: dict[str, set[str]] = {}
    task_ids: set[str] = set()
    public_texts: set[str] = set()
    sources: set[str] = set()
    for source in files:
        item: dict[str, Any] = {"path": str(source.resolve()), "status": "valid", "records": 0,
                                "checksum_verified_records": 0, "legacy_unsealed_records": 0}
        try:
            item["sha256"] = file_digest(source)
            kind = source.stem
            known = seen.setdefault(kind, set())
            with source.open("rb") as handle:
                for number, raw in enumerate(handle, 1):
                    if not raw.strip():
                        continue
                    if not raw.endswith(b"\n"):
                        raise ValueError(f"line {number}: incomplete journal tail; audit never repairs data")
                    record = json.loads(raw, parse_constant=_reject_constant)
                    if not isinstance(record, dict):
                        raise ValueError(f"line {number}: expected JSON object")
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise ValueError(f"line {number}: metadata must be an object")
                    checksum, record_id = metadata.get("record_checksum_sha256"), metadata.get("record_id")
                    sealed = checksum is not None or record_id is not None
                    if sealed:
                        if not isinstance(record_id, str) or not record_id or checksum != record_checksum(record):
                            raise ValueError(f"line {number}: journal checksum/record ID is invalid")
                        item["checksum_verified_records"] += 1
                    else:
                        item["legacy_unsealed_records"] += 1
                    if kind == "tasks":
                        task_id = record.get("id")
                        public_text = record.get("user_task")
                        if not isinstance(task_id, str) or not isinstance(public_text, str):
                            raise ValueError(f"line {number}: task requires ID and public text")
                        normalized = " ".join(public_text.casefold().split())
                        if task_id in task_ids or normalized in public_texts:
                            raise ValueError(f"line {number}: duplicate task ID/public text across supplied shards")
                        task_ids.add(task_id)
                        public_texts.add(normalized)
                        identity = task_id
                    elif kind == "trajectories":
                        task = record.get("task", {})
                        if not isinstance(task, dict) or not task.get("id") or not isinstance(record.get("task_success"), bool):
                            raise ValueError(f"line {number}: trajectory task ID and boolean success are required")
                        identity = record_id or json.dumps([task["id"], record.get("policy")])
                        if task.get("split") == "train":
                            result["training_trajectories"] += 1
                            result["verified_training_trajectories"] += int(sealed)
                        result["successes" if record["task_success"] else "failures"] += 1
                    else:
                        identity = record_id or hashlib.sha256(raw).hexdigest()
                    if identity in known:
                        raise ValueError(f"line {number}: duplicate {kind} record across supplied shards")
                    known.add(identity)
                    if metadata.get("source_kind"):
                        sources.add(str(metadata["source_kind"]))
                    item["records"] += 1
                    counts[kind] += 1
        except (OSError, ValueError, TypeError, KeyError) as error:
            item.update(status="invalid", reason=str(error))
            result["status"] = "invalid"
        result["files"].append(item)
    result["counts"], result["source_kinds"] = dict(counts), sorted(sources)
    result["counts_scope"] = "Read records; counts are incomplete and unusable as evidence when status is invalid"
    return result


def audit_checkpoint(path: str | Path) -> dict[str, Any]:
    original = Path(path)
    try:
        directory = resolve_checkpoint(original)
        metadata = _read(directory / "controller.json")
        backend, stage = metadata.get("backend"), metadata.get("stage")
        weights = [p for p in directory.rglob("*") if p.is_file() and p.suffix in {".pt", ".bin", ".safetensors"}
                   and p.name not in {"training_state.pt", "optimizer.pt"}]
        revision = metadata.get("resolved_revision") or metadata.get("original_resolved_revision")
        local_digest = metadata.get("local_checkpoint_sha256")
        pinned = bool(isinstance(revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", revision)) or bool(
            isinstance(local_digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", local_digest))
        pretrained = backend == "hf" and metadata.get("pretrained") is True
        training_run = original / "run.json"
        training = _read(training_run) if training_run.exists() else {}
        metrics = training.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("training metrics must be an object")
        run_evidence = audit_run(training_run) if training else {}
        skipped = metrics.get("skipped_amp_updates", 0)
        completed = (run_evidence.get("completed") is True and run_evidence.get("provenance_present") is True
                     and metrics.get("completed", True) is True and isinstance(metrics.get("optimizer_windows"), int)
                     and isinstance(skipped, int) and metrics["optimizer_windows"] > skipped
                     and metrics.get("pretrained") is True and bool(metrics.get("dataset_sha256"))
                     and metrics.get("specialists_updated") is False)
        expected = "dpo" if stage == "preference" else stage
        completed = completed and metrics.get("stage") == expected
        if completed and expected == "dpo":
            completed = bool(metrics.get("reference_checkpoint_sha256"))
        return {"path": str(original.resolve()), "resolved_path": str(directory.resolve()), "status": "valid" if weights else "invalid",
                "backend": backend, "stage": stage, "pretrained": pretrained,
                "base_model": metadata.get("base_model"), "resolved_revision": revision,
                "identity_pinned": pinned, "controller_metadata_sha256": file_digest(directory / "controller.json"),
                "weight_files": [str(p.relative_to(directory)) for p in sorted(weights)],
                "completed_pretrained_training_observed": bool(pretrained and pinned and weights and completed),
                "training_run": str(training_run.resolve()) if training else None,
                "training_run_sha256": file_digest(training_run) if training else None,
                "training_metrics": metrics,
                "scope": "Metadata and local artifact consistency; weights alone do not demonstrate completed training"}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _error(original, error)


def audit_run(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    source = directory if directory.is_file() else directory / "run.json"
    try:
        record = _read(source)
        if not isinstance(record.get("configuration", {}), dict) or not isinstance(record.get("hardware", {}), dict):
            raise ValueError("run configuration and hardware must be objects")
        if "metrics" in record and not isinstance(record["metrics"], dict):
            raise ValueError("run metrics must be an object")
        runtime = record.get("runtime_seconds")
        complete = isinstance(runtime, (int, float)) and not isinstance(runtime, bool) and math.isfinite(runtime) and runtime >= 0 and "metrics" in record
        commit = record.get("git_commit")
        provenance = bool(isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit)
                          and isinstance(record.get("git_dirty"), bool) and isinstance(record.get("seed"), int)
                          and not isinstance(record.get("seed"), bool) and record.get("configuration") and record.get("hardware"))
        configuration = record.get("configuration", {})
        requested_device = configuration.get("model", {}).get("device", "")
        hardware = record.get("hardware", {})
        cuda_model_run = complete and provenance and str(requested_device).startswith("cuda") and hardware.get("cuda_available") is True and bool(hardware.get("gpu"))
        return {"path": str(source.resolve()), "status": "valid", "sha256": file_digest(source),
                "completed": complete, "provenance_present": provenance, "git_commit": record.get("git_commit"),
                "git_dirty": record.get("git_dirty"), "seed": record.get("seed"), "hardware": hardware,
                "runtime_seconds": runtime, "metrics": record.get("metrics"),
                "cuda_model_run_observed": bool(cuda_model_run),
                "scope": "Recorded observations; this audit does not rerun or independently authenticate the experiment"}
    except (OSError, ValueError, TypeError) as error:
        return _error(source, error)


def audit(*, datasets: list[str] | None = None, checkpoints: list[str] | None = None,
          runs: list[str] | None = None, doctors: list[str] | None = None,
          claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    inventories = [audit_dataset(path) for path in datasets or []]
    models = [audit_checkpoint(path) for path in checkpoints or []]
    experiments = [audit_run(path) for path in runs or []]
    device_probes = []
    for path in doctors or []:
        try:
            record = _read(Path(path))
            device_probes.append({"path": str(Path(path).resolve()), "status": record.get("status"),
                                  "validation": record.get("validation"), "probe": record.get("probe"),
                                  "scope": "An arithmetic compatibility probe is not model performance validation"})
        except (OSError, ValueError) as error:
            device_probes.append(_error(Path(path), error))
    decisions = []
    for claim in claims or []:
        verdict: dict[str, Any] = {"claim": claim, "status": "unsupported"}
        if claim.get("kind") == "training_trajectory_count":
            minimum = claim.get("minimum")
            # Never sum potentially overlapping independent inventories to manufacture scale.
            available = max((item["training_trajectories"] for item in inventories if item["status"] == "valid"), default=0)
            verdict.update(observed=available, reason="Requires a valid supplied inventory containing the declared training trajectory count")
            if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0 and available >= minimum:
                verdict.update(status="supported_local_inventory", reason="Count is observed; dataset/model quality and training usage require separate evidence")
        elif claim.get("kind") == "pretrained_post_training":
            stages = claim.get("stages", ["sft", "preference"])
            observed = sorted({item.get("stage") for item in models if item.get("completed_pretrained_training_observed")})
            verdict.update(observed_stages=observed, reason="Requires pinned pretrained checkpoints, positive completed training updates, frozen specialists and a DPO reference")
            if isinstance(stages, list) and stages and all(stage in observed for stage in stages):
                verdict["status"] = "supported_local_artifacts"
        else:
            verdict["reason"] = ("No automated endorsement: this claim requires linked raw measurements, identical held-out tasks/specialists, "
                                 "appropriate token billing, repeated matched timing and uncertainty analysis. Inspect the reported measurements and claim gates.")
        decisions.append(verdict)
    return {"scope": "Local artifact consistency audit, not independent authenticity verification or NVIDIA production certification",
            "datasets": inventories, "checkpoints": models, "runs": experiments, "device_probes": device_probes,
            "nvidia_model_validation": "recorded_cuda_run_present" if any(item.get("cuda_model_run_observed") for item in experiments) else "absent",
            "claims": decisions,
            "unsupported_claims": sum(item["status"] == "unsupported" for item in decisions),
            "invalid_artifacts": sum(item.get("status") == "invalid" for item in inventories + models + experiments + device_probes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=[], help="Dataset directory or JSONL; repeat for separate inventories")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--run", action="append", default=[], help="Run directory or run.json")
    parser.add_argument("--doctor", action="append", default=[])
    parser.add_argument("--claims", help="JSON list of declarative claims; no measurement values are synthesized")
    parser.add_argument("--output", default="outputs/evidence/audit.json")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero for invalid artifacts or unsupported claims")
    args = parser.parse_args()
    claims = json.loads(Path(args.claims).read_text()) if args.claims else []
    if not isinstance(claims, list) or any(not isinstance(value, dict) for value in claims):
        parser.error("claims must be a JSON list of objects")
    result = audit(datasets=args.dataset, checkpoints=args.checkpoint, runs=args.run, doctors=args.doctor, claims=claims)
    write_json(args.output, result)
    print(json.dumps({"output": str(Path(args.output).resolve()), "unsupported_claims": result["unsupported_claims"],
                      "invalid_artifacts": result["invalid_artifacts"], "nvidia_model_validation": result["nvidia_model_validation"]}))
    raise SystemExit(int(args.strict and (result["unsupported_claims"] > 0 or result["invalid_artifacts"] > 0)))


if __name__ == "__main__":
    main()
