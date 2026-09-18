"""Select a routing-head representation using task-disjoint linear probes.

The pretrained backbone and zero-initialized LoRA updates stay frozen here.
One backbone pass supplies four pooling/normalization ablations. The selected
head is an SFT warm start; this experiment does not establish backbone
post-training, online task success, or an inference optimization gain.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from conductor.controller.actions import ActionCatalog
from conductor.controller.factory import build_controller
from conductor.datasets.integrity import digest, file_digest
from conductor.schema import ExecutionState
from conductor.training.data import batches, read_records, split_by_task
from conductor.utils.config import load_config
from conductor.utils.runs import Run, seed_everything, write_json


def fit_head(features: torch.Tensor, labels: torch.Tensor, train_indices: list[int],
             validation_indices: list[int], catalog: ActionCatalog, k: int,
             *, epochs: int, learning_rate: float, seed: int
             ) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    """Full-catalog cross entropy, selecting epochs on validation only."""
    if (not train_indices or not validation_indices or set(train_indices) & set(validation_indices)
            or len(set(train_indices + validation_indices)) != len(features)):
        raise ValueError("probe requires complete, disjoint, nonempty train/validation partitions")
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("probe epochs and learning_rate must be positive")
    if features.ndim != 2 or labels.shape != (len(features),) or not torch.isfinite(features).all():
        raise ValueError("probe features/labels must be finite and aligned")
    seed_everything(seed)
    head = nn.Linear(features.shape[1], len(catalog)).to(features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=.01)
    mask = catalog.mask(k, features.device)
    if not bool(mask[labels].all()):
        raise ValueError("probe targets exceed agent budget")
    train, validation = features[train_indices], features[validation_indices]
    train_labels, validation_labels = labels[train_indices], labels[validation_indices]
    trace, best, best_weights = [], None, None
    with torch.no_grad():
        initial = head(features).masked_fill(~mask, float("-inf"))
        initial_loss = float(F.cross_entropy(initial[train_indices], train_labels))
        initial_logit_std = float(initial[:, mask].std())
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train).masked_fill(~mask, float("-inf"))
        loss = F.cross_entropy(logits, train_labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            predicted = head(validation).masked_fill(~mask, float("-inf"))
            validation_loss = float(F.cross_entropy(predicted, validation_labels))
            accuracy = float((predicted.argmax(-1) == validation_labels).float().mean())
        row = {"epoch": epoch, "train_loss": float(loss.detach()),
               "validation_loss": validation_loss, "validation_accuracy": accuracy}
        trace.append(row)
        if best is None or (-accuracy, validation_loss) < (-best["validation_accuracy"], best["validation_loss"]):
            best = row
            best_weights = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best is not None and best_weights is not None
    head.load_state_dict(best_weights)
    with torch.no_grad():
        predictions = head(features).masked_fill(~mask, float("-inf")).argmax(-1)
    summary = {**best, "initial_train_loss": initial_loss, "initial_logit_std": initial_logit_std,
               "feature_rms": float(features.square().mean().sqrt()),
               "train_examples": len(train_indices), "validation_examples": len(validation_indices),
               "validation_predictions": predictions[validation_indices].tolist(),
               "validation_targets": labels[validation_indices].tolist()}
    return best_weights, summary, trace


def probe(config: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    options = effective["probe"]
    output = Path(options["output"])
    if output.exists():
        raise FileExistsError(f"use a fresh probe directory: {output}")
    if effective["model"].get("backend") != "hf" or not effective["model"].get("pretrained", True):
        raise ValueError("representation probes require an actual pretrained HF MoE")
    source_hash = file_digest(options["data"])
    records = read_records(options["data"], "sft")
    train, validation = split_by_task(records, float(options.get("validation_fraction", .25)), effective.get("seed", 42))
    if not validation:
        raise ValueError("representation selection requires task-disjoint validation")
    train_ids = {record["task_id"] for record in train}
    validation_ids = {record["task_id"] for record in validation}
    if train_ids & validation_ids:
        raise ValueError("probe task leakage")
    train_indices = [index for index, record in enumerate(records) if record["task_id"] in train_ids]
    validation_indices = [index for index, record in enumerate(records) if record["task_id"] in validation_ids]
    run = Run(output, effective)
    controller = build_controller(effective)
    representations: dict[str, list[torch.Tensor]] = {f"{pooling}/{normalization}": []
        for pooling in ("last", "mean") for normalization in ("none", "layer_norm")}
    token_audit = []
    with torch.inference_mode():
        for batch in batches(records, int(options.get("batch_size", 2))):
            states = [ExecutionState(**record["state"]) for record in batch]
            inputs = controller.encode_states(states)
            hidden = controller.model.backbone(**inputs, use_cache=False, return_dict=True,
                                                output_router_logits=False).last_hidden_state
            for pooling in ("last", "mean"):
                for normalization in ("none", "layer_norm"):
                    controller.model.pooling = pooling
                    controller.model.head_input_normalization = normalization
                    representations[f"{pooling}/{normalization}"].append(
                        controller.model.pooled_hidden(hidden, inputs["attention_mask"]).float().cpu())
            for record, ids, attention in zip(batch, inputs["input_ids"], inputs["attention_mask"]):
                token_audit.append({"task_id": record["task_id"], "step": record["state"]["current_step"],
                                    "state_sha256": digest(record["state"]), "input_tokens": int(attention.sum()),
                                    "decoded_input": controller.tokenizer.decode(ids[attention.bool()].tolist())})
    write_json(output / "data_partition.json", {"train_task_ids": sorted(train_ids), "validation_task_ids": sorted(validation_ids)})
    write_json(output / "token_audit.json", token_audit)
    labels = torch.tensor([controller.catalog.index(record["decision"]) for record in records])
    summaries, weights, traces, cached = {}, {}, {}, {}
    for name, chunks in representations.items():
        features = torch.cat(chunks)
        cached[name] = features
        weights[name], summaries[name], traces[name] = fit_head(
            features, labels, train_indices, validation_indices, controller.catalog,
            int(effective.get("routing", {}).get("k", 2)), epochs=int(options.get("epochs", 200)),
            learning_rate=float(options.get("learning_rate", .01)), seed=int(effective.get("seed", 42)))
    if file_digest(options["data"]) != source_hash:
        raise ValueError("training source changed during feature extraction")
    selected = min(summaries, key=lambda name: (-summaries[name]["validation_accuracy"], summaries[name]["validation_loss"], name))
    pooling, normalization = selected.split("/")
    controller.model.pooling = pooling
    controller.model.head_input_normalization = normalization
    controller.pooling = pooling
    controller.head_input_normalization = normalization
    controller.config["model"].update(pooling=pooling, head_input_normalization=normalization)
    controller.model.head.load_state_dict(weights[selected])
    checkpoint = output / "selected"
    controller.save(checkpoint, "sft")
    metadata = json.loads((checkpoint / "controller.json").read_text())
    metadata.update(training_scope="frozen_backbone_head_only", backbone_adapters_updated=False,
                    source_dataset_sha256=source_hash, selected_on="task_disjoint_internal_validation")
    write_json(checkpoint / "controller.json", metadata)
    torch.save({"features": cached, "labels": labels, "dataset_sha256": source_hash,
                "model": controller.config["model"], "state_sha256": [digest(record["state"]) for record in records]},
               output / "representations.pt")
    write_json(output / "fit_trace.json", traces)
    write_json(output / "selected_config.json", controller.config)
    result = {"selected_representation": selected, "ablations": summaries, "checkpoint": str(checkpoint),
              "dataset_sha256": source_hash, "training_tasks": len(train_ids), "validation_tasks": len(validation_ids),
              "pretrained": True, "backbone_updated": False, "specialists_updated": False,
              "selection_uses_heldout_tasks": False,
              "scope": "Frozen pretrained representations; fitted action heads. Online task success requires separate rollout evaluation."}
    run.finish(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    probe(load_config(args.config))


if __name__ == "__main__":
    main()
