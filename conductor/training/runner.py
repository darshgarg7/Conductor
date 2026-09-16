"""Genuine SFT and preference updates for the coordinator only."""
from __future__ import annotations

import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from conductor.controller.factory import build_controller
from conductor.preference.dpo import dpo_loss
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.training.data import batches, read_records, split_by_task
from conductor.utils.runs import Run, log_event, seed_everything, write_json


def checkpoint_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(path).rglob("*")):
        if file.is_file() and file.name != "run.json":
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


def _log_probabilities(controller: Any, records: list[dict[str, Any]], k: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    states = [ExecutionState(**record["state"]) for record in records]
    probabilities = controller.forward_states(states, k=k).float().log_softmax(-1)
    chosen = torch.tensor([controller.catalog.index(record["chosen"]) for record in records], device=controller.device)
    rejected = torch.tensor([controller.catalog.index(record["rejected"]) for record in records], device=controller.device)
    index = torch.arange(len(records), device=controller.device)
    return probabilities[index, chosen], probabilities[index, rejected]


def _reference_cache(config: dict[str, Any], checkpoint: str, records: list[dict[str, Any]],
                     batch_size: int, k: int, output: Path) -> str:
    reference = build_controller(config, checkpoint)
    if reference.stage != "sft":
        raise ValueError("DPO requires a saved SFT checkpoint as the reference, not the pretrained/random base")
    reference.model.eval()
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    digest = checkpoint_digest(checkpoint)
    cache = []
    with torch.inference_mode():
        for batch in batches(records, batch_size):
            chosen, rejected = _log_probabilities(reference, batch, k)
            for record, good, bad in zip(batch, chosen.cpu().tolist(), rejected.cpu().tolist()):
                record["reference_chosen_logp"] = good
                record["reference_rejected_logp"] = bad
                cache.append({"task_id": record.get("task_id"),
                              "state_sha256": hashlib.sha256(json.dumps(record["state"], sort_keys=True).encode()).hexdigest(),
                              "chosen_logp": good, "rejected_logp": bad})
    write_json(output / "reference_log_probabilities.json", {"sft_checkpoint_sha256": digest, "records": cache})
    del reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return digest


def _validate(controller: Any, records: list[dict[str, Any]], stage: str, batch_size: int,
              k: int, beta: float) -> dict[str, float | None]:
    if not records:
        return {"loss": None, "accuracy": None}
    controller.model.eval()
    total_loss, total_accuracy = 0.0, 0.0
    with torch.inference_mode():
        for batch in batches(records, batch_size):
            if stage == "sft":
                logits = controller.forward_states([ExecutionState(**record["state"]) for record in batch], k=k).float()
                labels = torch.tensor([controller.catalog.index(record["decision"]) for record in batch], device=controller.device)
                loss = F.cross_entropy(logits, labels)
                accuracy = float((logits.argmax(-1) == labels).float().mean())
            else:
                chosen, rejected = _log_probabilities(controller, batch, k)
                ref_chosen = torch.tensor([record["reference_chosen_logp"] for record in batch], device=controller.device)
                ref_rejected = torch.tensor([record["reference_rejected_logp"] for record in batch], device=controller.device)
                loss, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta)
                accuracy = float((chosen > rejected).float().mean())
            total_loss += float(loss) * len(batch)
            total_accuracy += accuracy * len(batch)
    return {"loss": total_loss / len(records), "accuracy": total_accuracy / len(records)}


def train(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    training = config["training"]
    stage = training.get("stage", "sft")
    if stage not in {"sft", "dpo"}:
        raise ValueError("training.stage must be sft or dpo")
    checkpoint = checkpoint or training.get("checkpoint")
    if stage == "dpo" and checkpoint is None:
        raise ValueError("DPO requires --checkpoint pointing to an SFT controller")
    output = Path(training.get("output", f"outputs/checkpoints/conductor-{stage}"))
    if checkpoint is not None and Path(checkpoint).resolve() == output.resolve():
        raise ValueError("training output must not overwrite its input/reference checkpoint")
    run = Run(output, config, checkpoint)
    records = read_records(training["data"], stage)
    batch_size = int(training.get("batch_size", 32))
    k = int(config.get("routing", {}).get("k", 2))
    beta = float(training.get("beta", 0.1))
    # A k=1 sweep cannot imitate k=2 activations. Explicitly exclude such
    # examples and report the count; never take CE/DPO on a masked -inf label.
    original_examples = len(records)
    legal = []
    for record in records:
        fields = ("decision",) if stage == "sft" else ("chosen", "rejected")
        decisions = [RoutingDecision(**record[field]) for field in fields]
        for decision in decisions:
            decision.validate(len(AGENT_NAMES))
        if all(len(decision.selected_agents) <= k for decision in decisions):
            legal.append(record)
    records = legal
    if not records:
        raise ValueError(f"no legal {stage} records for configured k={k}")
    excluded = original_examples - len(records)
    if excluded:
        log_event("training_labels_excluded", reason="action exceeds configured k", count=excluded, k=k)
    reference_digest = None
    if stage == "dpo":
        reference_digest = _reference_cache(config, checkpoint, records, batch_size, k, output)
    controller = build_controller(config, checkpoint)
    # Model architecture/identity comes from its checkpoint; the experiment's
    # training stage, data, seed and external k come from the current launch.
    controller.config["training"] = dict(training)
    controller.config["seed"] = seed
    controller.config["routing"] = dict(config.get("routing", {"k": k}))
    run.record["resolved_model"] = dict(controller.config["model"])
    write_json(output / "run.json", run.record)
    # Check every target before any update; sparse labels cannot activate > k.
    for record in records:
        for field in (("decision",) if stage == "sft" else ("chosen", "rejected")):
            RoutingDecision(**record[field]).validate(k)
            controller.catalog.index(record[field])
        if stage == "dpo" and controller.catalog.index(record["chosen"]) == controller.catalog.index(record["rejected"]):
            raise ValueError("preference actions must differ")
    train_records, validation = split_by_task(records, float(training.get("validation_fraction", 0.2)), seed)
    for batch in batches(records[:min(len(records), 128)], batch_size):
        controller.batch_route([ExecutionState(**record["state"]) for record in batch], k)
    write_json(output / "expert_statistics_before.json", controller.expert_stats())
    controller.reset_expert_stats()
    initial = _validate(controller, train_records, stage, batch_size, k, beta)
    parameters = [parameter for parameter in controller.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("no trainable coordinator parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(training.get("learning_rate", 0.003)),
                                  weight_decay=float(training.get("weight_decay", 0.01)))
    scaler = torch.amp.GradScaler("cuda", enabled=controller.device.type == "cuda" and controller.dtype == torch.float16)
    epochs = int(training.get("epochs", 40))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    metrics_path = output / "training_metrics.jsonl"
    history = []
    generator = random.Random(seed)
    save_every = int(training.get("save_every_epochs", 0))
    with metrics_path.open("w") as handle:
        for epoch in range(epochs):
            generator.shuffle(train_records)
            controller.model.train()
            total, loss_sum, margin_sum = 0, 0.0, 0.0
            for batch in batches(train_records, batch_size):
                optimizer.zero_grad(set_to_none=True)
                if stage == "sft":
                    states = [ExecutionState(**record["state"]) for record in batch]
                    logits = controller.forward_states(states, k=k).float()
                    labels = torch.tensor([controller.catalog.index(record["decision"]) for record in batch], device=controller.device)
                    loss = F.cross_entropy(logits, labels)
                    margin = None
                else:
                    chosen, rejected = _log_probabilities(controller, batch, k)
                    ref_chosen = torch.tensor([record["reference_chosen_logp"] for record in batch], device=controller.device)
                    ref_rejected = torch.tensor([record["reference_rejected_logp"] for record in batch], device=controller.device)
                    loss, margin = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite training loss; check precision and labels")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(training.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(batch)
                if margin is not None:
                    margin_sum += float(margin.mean()) * len(batch)
                total += len(batch)
            validation_metrics = _validate(controller, validation, stage, batch_size, k, beta)
            record = {"epoch": epoch + 1, "train_loss": loss_sum / total,
                      "validation_loss": validation_metrics["loss"], "validation_accuracy": validation_metrics["accuracy"]}
            if stage == "dpo":
                record["mean_dpo_margin"] = margin_sum / total
            history.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
                log_event("training_epoch", stage=stage, **record)
            if save_every and (epoch + 1) % save_every == 0:
                controller.save(output / f"epoch-{epoch + 1}", "sft" if stage == "sft" else "preference")
    controller.model.eval()
    final_train = _validate(controller, train_records, stage, batch_size, k, beta)
    final_validation = _validate(controller, validation, stage, batch_size, k, beta)
    controller.save(output, "sft" if stage == "sft" else "preference")
    torch.save({"optimizer": optimizer.state_dict(), "epoch": epochs, "seed": seed}, output / "optimizer.pt")
    write_json(output / "data_partition.json", {"train_task_ids": sorted({str(record.get("task_id", record["state"]["user_task"])) for record in train_records}),
               "validation_task_ids": sorted({str(record.get("task_id", record["state"]["user_task"])) for record in validation})})
    controller.reset_expert_stats()
    for batch in batches(records[:min(len(records), 128)], batch_size):
        controller.batch_route([ExecutionState(**record["state"]) for record in batch], k)
    write_json(output / "expert_statistics.json", controller.expert_stats())
    metrics = {"stage": stage, "epochs": epochs, "train_examples": len(train_records), "validation_examples": len(validation),
               "source_examples": original_examples, "excluded_actions_exceeding_k": excluded, "train_k": k,
               "initial_train": initial, "final_train": final_train, "final_validation": final_validation,
               "trainable_parameters": sum(parameter.numel() for parameter in parameters),
               "total_controller_parameters": sum(parameter.numel() for parameter in controller.model.parameters()),
               "reference_checkpoint_sha256": reference_digest,
               "dataset_sha256": hashlib.sha256(Path(training["data"]).read_bytes()).hexdigest(),
               "pretrained": controller.config.get("model", {}).get("backend") == "hf" and controller.pretrained,
               "checkpoint": str(output), "specialists_updated": False}
    run.finish(metrics)
    return metrics
