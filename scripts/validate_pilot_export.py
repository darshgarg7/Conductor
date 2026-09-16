"""Verify the pilot export against all held-out public initial states on CPU."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch

from conductor.controller.artifacts import resolve_checkpoint
from conductor.controller.factory import build_controller
from conductor.controller.hf import HFController
from conductor.datasets.io import read_jsonl
from conductor.evaluation.provenance import checkpoint_sha256
from conductor.schema import ExecutionState
from conductor.utils.runs import Run, seed_everything


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    base = Path("outputs/research/granite-pilot")
    checkpoint = base / "preference"
    metadata = json.loads((resolve_checkpoint(checkpoint) / "controller.json").read_text())
    config = metadata["configuration"]
    config["model"].update(device="cpu", dtype="float32", require_cuda=False)
    config["inference"]["instrument_experts"] = False
    config["export_validation"] = {"preserve_model": False,
                                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    seed_everything(config.get("seed", 42))
    run = Run(base / "export-verification", config, str(checkpoint))
    tasks = [task for task in read_jsonl("data/generated/granite-pilot/tasks.jsonl") if task["split"] == "eval"]
    states = [ExecutionState(task["user_task"], task["task_type"]) for task in tasks]
    controller = build_controller(config, str(checkpoint))
    source_hash = checkpoint_sha256(checkpoint)
    with torch.inference_mode():
        before = controller.forward_encoded(controller.encode_states(states)).cpu()
    decisions = {str(k): [decision.to_dict() for decision in controller.batch_route(states, k)] for k in (1, 2, 3)}
    manifest = controller.export_merged(base / "export", validation_states=states, preserve_model=False)
    require(checkpoint_sha256(checkpoint) == source_hash, "source checkpoint changed during export")
    del controller
    gc.collect()
    restored = HFController.load(base / "export", {"model": {"device": "cpu", "dtype": "float32", "require_cuda": False}})
    with torch.inference_mode():
        after = restored.forward_encoded(restored.encode_states(states)).cpu()
    torch.testing.assert_close(before, after, **manifest["validation"]["validation_tolerances"]["logits"])
    probability_difference = 0.0
    for k in (1, 2, 3):
        mask = restored.catalog.mask(k)
        original_probability = (before.masked_fill(~mask, float("-inf")) / restored.temperature).softmax(-1)
        restored_probability = (after.masked_fill(~mask, float("-inf")) / restored.temperature).softmax(-1)
        torch.testing.assert_close(original_probability, restored_probability,
                                   **manifest["validation"]["validation_tolerances"]["probabilities"])
        require(torch.equal(original_probability.argmax(-1), restored_probability.argmax(-1)),
                f"reloaded probability argmax changed at k={k}")
        probability_difference = max(probability_difference, float((original_probability - restored_probability).abs().max()))
        actual = [decision.to_dict() for decision in restored.batch_route(states, k)]
        for original, reloaded in zip(decisions[str(k)], actual):
            require(original["selected_agents"] == reloaded["selected_agents"], f"reloaded agents changed at k={k}")
            require(original["execution_mode"] == reloaded["execution_mode"], f"reloaded mode changed at k={k}")
            require(original["terminate"] == reloaded["terminate"], f"reloaded termination changed at k={k}")
    run.finish({"status": "measured", "device": "cpu", "probe_task_ids": [task["id"] for task in tasks],
                "probe_count": len(states), "k_values": [1, 2, 3], "reload_actions_equal": True,
                "maximum_reload_logit_difference": float((before - after).abs().max()),
                "maximum_reload_probability_difference": probability_difference,
                "maximum_merge_probability_difference": manifest["validation"]["maximum_probability_difference"],
                "source_checkpoint_unchanged": True, "source_checkpoint_sha256": source_hash,
                "validation_tolerances": manifest["validation"]["validation_tolerances"],
                "maximum_merge_logit_difference": manifest["validation"]["maximum_logit_difference"],
                "export_file_count": len(manifest["files_sha256"]),
                "scope": "Six synthetic public initial-state probes; numerical equivalence, not task accuracy or GPU performance."})


if __name__ == "__main__":
    main()
