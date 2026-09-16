"""Verify genuine pinned Granite pilot adapter/head updates without a base load.

Run after SFT and preference training complete:
python scripts/verify_pilot_adapters.py --run-root outputs/research/granite-pilot

The pinned base architecture config must already be cached; no model weights are
loaded or downloaded. Output is a new immutable directory, leaving previous
pilot evidence unchanged. No full frozen-base tensor audit is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file
from huggingface_hub import try_to_load_from_cache

from conductor.controller.artifacts import atomic_directory, atomic_json, resolve_checkpoint
from conductor.controller.actions import ActionCatalog
from conductor.schema import RoutingDecision
from conductor.training.runner import checkpoint_digest

ROOT = Path(__file__).resolve().parents[1]
PIN = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
MODEL = "ibm-granite/granite-3.1-1b-a400m-base"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    """Evidence checks remain active when Python is run with optimization."""
    if not condition:
        raise ValueError(message)


def read_stage(directory: Path) -> dict:
    checkpoint = resolve_checkpoint(directory)
    metadata = json.loads((checkpoint / "controller.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    configuration = metadata["configuration"]["model"]
    adapter_configuration = json.loads((checkpoint / "adapter" / "adapter_config.json").read_text())
    require(metadata["pretrained"] and metrics["pretrained"], "development random model is not pretrained evidence")
    require(metadata["resolved_revision"] == PIN, "resolved pretrained revision mismatch")
    require(metadata["base_model"] == MODEL, "wrong base model")
    require(configuration.get("resolved_revision") == PIN, "checkpoint replay is not pinned")
    require(metrics["completed"], "training did not complete")
    require(adapter_configuration["init_lora_weights"] is True,
            "nonzero B only proves an update relative to documented zero-B initialization")
    require(adapter_configuration["r"] == 4 and adapter_configuration["lora_alpha"] == 8,
            "pilot adapter rank/alpha differ from r=4/alpha=8")
    require(set(adapter_configuration["target_modules"]) == {"q_proj", "v_proj", "router.layer"},
            "pilot adapter target modules differ from the saved experimental configuration")
    tensors = {}
    files = {}
    for artifact in sorted((checkpoint / "adapter").glob("*.safetensors")):
        part = load_file(str(artifact), device="cpu")
        require(not tensors.keys() & part.keys(), "duplicate adapter tensor keys across safetensors files")
        tensors.update(part)
        files[str(artifact.relative_to(checkpoint))] = file_sha256(artifact)
    require(bool(tensors), "missing LoRA adapter weights")
    heads = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    files["head.pt"] = file_sha256(checkpoint / "head.pt")
    require(all(torch.isfinite(tensor).all().item() for tensor in [*tensors.values(), *heads.values()]),
            "adapter/head contains non-finite saved weights")
    require(all(tensor.dtype == torch.float32 for tensor in [*tensors.values(), *heads.values()]),
            "pilot trainable adapter/head weights are not saved in FP32")
    require(sum(tensor.numel() for tensor in [*tensors.values(), *heads.values()]) == metrics["trainable_parameters"],
            "saved adapter/head parameter count differs from the recorded optimizer parameter count")
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
    require(before.keys() == after.keys(), "SFT/DPO artifact tensor catalogs differ")
    changed = []
    total_squared = 0.0
    total_changed = 0
    for name, previous in before.items():
        current = after[name]
        require(previous.shape == current.shape, f"SFT/DPO tensor shapes differ: {name}")
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
    require(sft["metadata"]["stage"] == "sft" and preference["metadata"]["stage"] == "preference",
            "checkpoints do not represent completed SFT and preference stages")
    reference = json.loads((directory / "preference" / "reference_log_probabilities.json").read_text())
    actual_sft_digest = checkpoint_digest(directory / "sft")
    require(actual_sft_digest == reference["sft_checkpoint_sha256"] == preference["metrics"]["reference_checkpoint_sha256"],
            "DPO did not use this exact frozen SFT artifact")
    config_path = try_to_load_from_cache(MODEL, "config.json", revision=PIN)
    require(isinstance(config_path, str), "missing cached pinned pretrained architecture config; complete the pilot download first")
    base_config = json.loads(Path(config_path).read_text())
    layers = base_config["num_hidden_layers"]
    summaries = {"sft": weight_summary(sft["tensors"]), "preference": weight_summary(preference["tensors"])}
    for summary in summaries.values():
        require(summary["router_lora_B_tensor_count"] == layers, "router adapter count differs from pretrained layer count")
        require(summary["nonzero_router_lora_B_tensor_count"] == layers, "one or more router adapters stayed at initial zero B")
        require(summary["nonzero_lora_B_tensor_count"] == summary["lora_B_tensor_count"],
                "one or more LoRA B tensors stayed at initial zero")
    deltas = {"adapters": compare_weights(sft["tensors"], preference["tensors"]), "head": compare_weights(sft["head"], preference["head"])}
    require(bool(deltas["adapters"]["changed_tensor_count"] and deltas["head"]["changed_tensor_count"]),
            "DPO did not update both adapters and routing head")
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
    return evidence


def supervision_diagnostics(directory: Path, data_root: Path | None) -> dict:
    """Count distinct successful first-action labels without altering the dataset."""
    sft_run = json.loads((directory / "sft" / "run.json").read_text())
    configured = Path(sft_run["configuration"]["training"]["data"])
    source = data_root / configured.name if data_root is not None else configured
    if not source.is_file() and data_root is None:
        source = Path("data/generated/granite-pilot") / configured.name
    require(source.is_file(), "SFT data missing; use --data-root or --skip-supervision-diagnostics")
    metrics = json.loads((directory / "sft" / "metrics.json").read_text())
    require(file_sha256(source) == metrics["dataset_sha256"], "supervision diagnostic dataset differs from actual SFT data")
    targets: dict[str, set[tuple]] = defaultdict(set)
    states: dict[str, set[str]] = defaultdict(set)
    with source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["state"]["current_step"]) != 0:
                continue
            decision = RoutingDecision(**record["decision"])
            if len(decision.selected_agents) <= metrics["train_k"]:
                decision.validate(metrics["train_k"])
                task_id = str(record["task_id"])
                states[task_id].add(json.dumps(record["state"], sort_keys=True, separators=(",", ":")))
                targets[task_id].add(ActionCatalog.key(decision))
    require(all(len(values) == 1 for values in states.values()),
            "a task has different initial execution states; this diagnostic only compares identical task/state inputs")
    counts = Counter(len(decisions) for decisions in targets.values())
    return {"initial_state_task_count": len(targets),
            "initial_states_with_multiple_distinct_successful_targets": sum(count for size, count in counts.items() if size > 1),
            "distinct_target_counts": {str(size): count for size, count in sorted(counts.items())},
            "interpretation": "Successful reference trajectories can provide different first actions for the same task; this does not identify a unique cost-optimal SFT label."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("outputs/research/granite-pilot"))
    parser.add_argument("--data-root", type=Path,
                        help="SFT dataset directory; default: saved training.data path, falling back to data/generated/granite-pilot")
    parser.add_argument("--output-dir", type=Path,
                        help="new output directory; default: RUN_ROOT/adapter-verification; refuses existing outputs")
    parser.add_argument("--skip-supervision-diagnostics", action="store_true",
                        help="verify adapters without reading SFT trajectory labels")
    arguments = parser.parse_args()
    directory = arguments.run_root.resolve()
    output = (arguments.output_dir or directory / "adapter-verification").resolve()
    if output.exists():
        raise FileExistsError(f"verification output already exists: {output}; select a fresh --output-dir")
    for stage in ("sft", "preference"):
        source_checkpoint = resolve_checkpoint(directory / stage).resolve()
        require(source_checkpoint not in output.parents,
                "verification must not publish inside an immutable source checkpoint")
    evidence = verify(directory)
    diagnostics = None if arguments.skip_supervision_diagnostics else supervision_diagnostics(directory, arguments.data_root)
    with atomic_directory(output) as temporary:
        atomic_json(temporary / "post_training_verification.json", evidence)
        if diagnostics is not None:
            atomic_json(temporary / "supervision_diagnostics.json", diagnostics)
    print(json.dumps({"verified": True, "output": str(output), "supervision_diagnostics": diagnostics,
                      "stage_weight_summaries": {stage: {key: value for key, value in summary.items() if key != "router_lora_B_tensors"}
                                                 for stage, summary in evidence["stage_weight_summaries"].items()},
                      "changed_adapter_tensors": evidence["sft_to_preference_weight_deltas"]["adapters"]["changed_tensor_count"],
                      "changed_head_tensors": evidence["sft_to_preference_weight_deltas"]["head"]["changed_tensor_count"],
                      "changed_router_adapter_tensors": evidence["changed_router_adapter_tensor_count"]}, indent=2))


if __name__ == "__main__":
    main()
