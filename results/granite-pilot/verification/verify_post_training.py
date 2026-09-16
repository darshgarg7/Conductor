"""Read-only adapter/head verification; never instantiate or load the full base.

Run: .venv/bin/python outputs/research/granite-pilot/verify_post_training.py --wait-seconds 600
Writes evidence beside this script. No base-tensor frozen-weight audit is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file
from huggingface_hub import try_to_load_from_cache

# Running a file directly puts this ignored directory, not the repository, on
# sys.path. Import only pure checkpoint helpers; no model construction occurs.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from conductor.controller.artifacts import resolve_checkpoint
from conductor.training.runner import checkpoint_digest

PIN = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_stage(directory: Path) -> dict:
    checkpoint = resolve_checkpoint(directory)
    metadata = json.loads((checkpoint / "controller.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    configuration = metadata["configuration"]["model"]
    adapter_configuration = json.loads((checkpoint / "adapter" / "adapter_config.json").read_text())
    assert metadata["pretrained"] and metrics["pretrained"], "development random model is not pretrained evidence"
    assert metadata["resolved_revision"] == PIN, "resolved pretrained revision mismatch"
    assert metadata["base_model"] == MODEL, "wrong base model"
    assert configuration.get("resolved_revision") == PIN, "checkpoint replay is not pinned"
    assert metrics["completed"], "training did not complete"
    assert adapter_configuration["init_lora_weights"] is True, "nonzero B only proves an update relative to documented zero-B initialization"
    assert adapter_configuration["r"] == 4 and adapter_configuration["lora_alpha"] == 8
    assert set(adapter_configuration["target_modules"]) == {"q_proj", "v_proj", "router.layer"}
    tensors = {}
    files = {}
    for artifact in sorted((checkpoint / "adapter").glob("*.safetensors")):
        tensors.update(load_file(str(artifact), device="cpu"))
        files[str(artifact.relative_to(checkpoint))] = file_sha256(artifact)
    assert tensors, "missing LoRA adapter weights"
    heads = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    files["head.pt"] = file_sha256(checkpoint / "head.pt")
    assert all(torch.isfinite(tensor).all().item() for tensor in [*tensors.values(), *heads.values()])
    assert all(tensor.dtype == torch.float32 for tensor in [*tensors.values(), *heads.values()])
    assert sum(tensor.numel() for tensor in [*tensors.values(), *heads.values()]) == metrics["trainable_parameters"]
    return {"checkpoint": checkpoint, "metadata": metadata, "metrics": metrics,
            "tensors": tensors, "head": heads, "files": files, "adapter_configuration": adapter_configuration, "run": run}


def weight_summary(tensors: dict) -> dict:
    b_weights = {key: tensor for key, tensor in tensors.items() if "lora_B" in key}
    router_b = {key: tensor for key, tensor in b_weights.items() if "router.layer" in key}
    return {"adapter_tensor_count": len(tensors), "adapter_parameter_count": sum(tensor.numel() for tensor in tensors.values()),
            "lora_B_tensor_count": len(b_weights),
            "nonzero_lora_B_tensor_count": sum(bool(torch.count_nonzero(tensor)) for tensor in b_weights.values()),
            "router_lora_B_tensor_count": len(router_b),
            "nonzero_router_lora_B_tensor_count": sum(bool(torch.count_nonzero(tensor)) for tensor in router_b.values()),
            "router_lora_B_tensors": [{"name": key, "shape": list(tensor.shape), "nonzero_elements": int(torch.count_nonzero(tensor)),
                                       "l2_norm": float(tensor.float().norm()), "absolute_sum": float(tensor.float().abs().sum())}
                                      for key, tensor in sorted(router_b.items())]}


def compare_weights(before: dict, after: dict) -> dict:
    assert before.keys() == after.keys(), "SFT/DPO artifact tensor catalogs differ"
    changed = []
    total_squared = 0.0
    total_changed = 0
    for name, previous in before.items():
        current = after[name]
        assert previous.shape == current.shape
        delta = current.float() - previous.float()
        count = int(torch.count_nonzero(delta))
        if count:
            changed.append({"name": name, "changed_elements": count, "l2_delta": float(delta.norm()), "max_absolute_delta": float(delta.abs().max())})
            total_squared += float(delta.square().sum())
            total_changed += count
    return {"tensor_count": len(before), "changed_tensor_count": len(changed), "changed_elements": total_changed,
            "aggregate_l2_delta": total_squared ** .5, "changes": changed}


def verify(directory: Path) -> dict:
    sft, preference = read_stage(directory / "sft"), read_stage(directory / "preference")
    assert sft["metadata"]["stage"] == "sft" and preference["metadata"]["stage"] == "preference"
    reference = json.loads((directory / "preference" / "reference_log_probabilities.json").read_text())
    actual_sft_digest = checkpoint_digest(directory / "sft")
    assert actual_sft_digest == reference["sft_checkpoint_sha256"] == preference["metrics"]["reference_checkpoint_sha256"], "DPO did not use this exact frozen SFT artifact"
    config_path = try_to_load_from_cache(MODEL, "config.json", revision=PIN)
    assert isinstance(config_path, str), "missing cached pinned pretrained architecture config"
    base_config = json.loads(Path(config_path).read_text())
    layers = base_config["num_hidden_layers"]
    summaries = {"sft": weight_summary(sft["tensors"]), "preference": weight_summary(preference["tensors"])}
    for summary in summaries.values():
        assert summary["router_lora_B_tensor_count"] == layers
        assert summary["nonzero_router_lora_B_tensor_count"] == layers, "one or more router adapters stayed at initial zero B"
    deltas = {"adapters": compare_weights(sft["tensors"], preference["tensors"]), "head": compare_weights(sft["head"], preference["head"])}
    assert deltas["adapters"]["changed_tensor_count"] and deltas["head"]["changed_tensor_count"], "DPO did not update both adapters and routing head"
    router_changes = [change for change in deltas["adapters"]["changes"] if "router.layer" in change["name"]]
    evidence = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL, "pretrained_revision": PIN,
                "base_architecture": {"model_type": base_config["model_type"], "num_hidden_layers": layers,
                                      "num_local_experts": base_config["num_local_experts"], "num_experts_per_tok": base_config["num_experts_per_tok"]},
                "cached_base_config_sha256": file_sha256(Path(config_path)),
                "git_commit_at_verification": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_clean_at_verification": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "stage_metrics": {"sft": sft["metrics"], "preference": preference["metrics"]},
                "stage_provenance": {stage: {key: record["run"].get(key) for key in ("git_commit", "git_dirty", "seed", "hardware", "library_versions", "runtime_seconds")}
                                     for stage, record in (("sft", sft), ("preference", preference))},
                "stage_head_parameter_counts": {"sft": sum(tensor.numel() for tensor in sft["head"].values()),
                                                "preference": sum(tensor.numel() for tensor in preference["head"].values())},
                "adapter_initialization": "PEFT init_lora_weights=true initializes LoRA B to zero; all saved B tensors are nonzero.",
                "adapter_configurations": {"sft": sft["adapter_configuration"], "preference": preference["adapter_configuration"]},
                "stage_weight_summaries": summaries, "sft_to_preference_weight_deltas": deltas,
                "changed_router_adapter_tensor_count": len(router_changes),
                "exact_sft_reference_sha256": actual_sft_digest,
                "reference_probability_record_count": len(reference["records"]),
                "artifact_files_sha256": {"sft": sft["files"], "preference": preference["files"]},
                "verification_script_sha256": file_sha256(Path(__file__)),
                "verification_scope": "Actual saved FP32 coordinator adapter/head updates and exact frozen SFT reference identity; no full base tensors or specialist tensors loaded.",
                "frozen_base_weight_audit_performed": False,
                "frozen_specialist_weight_audit_performed": False,
                "freeze_evidence_scope": "Coordinator optimizer parameter selection, serving/training source, and regression tests establish freeze policy; file immutability is not a full frozen-base tensor audit.",
                "nvidia_execution_verified": False}
    destination = directory / "post_training_verification.json"
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=float, default=0)
    arguments = parser.parse_args()
    directory = Path(__file__).resolve().parent
    deadline = time.monotonic() + arguments.wait_seconds
    while not all((directory / stage / "metrics.json").exists() for stage in ("sft", "preference")):
        if time.monotonic() >= deadline:
            raise SystemExit("SFT/preference metrics not ready; retry after both stages complete")
        time.sleep(1)
    evidence = verify(directory)
    print(json.dumps({"verified": True, "output": str(directory / "post_training_verification.json"),
                      "stage_weight_summaries": {stage: {key: value for key, value in summary.items() if key != "router_lora_B_tensors"}
                                                 for stage, summary in evidence["stage_weight_summaries"].items()},
                      "changed_adapter_tensors": evidence["sft_to_preference_weight_deltas"]["adapters"]["changed_tensor_count"],
                      "changed_head_tensors": evidence["sft_to_preference_weight_deltas"]["head"]["changed_tensor_count"],
                      "changed_router_adapter_tensors": evidence["changed_router_adapter_tensor_count"]}, indent=2))


if __name__ == "__main__":
    main()
