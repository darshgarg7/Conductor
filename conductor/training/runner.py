"""Coordinator-only SFT/DPO with exact accumulation, resume and torchrun DDP.

Checkpoints commit complete optimizer-update windows. Single-node DDP shards
examples without dropping or repeating their contributions: empty rank shards
execute zero-weight dummy samples, so every rank participates in collectives.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import platform
from importlib.metadata import version
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from conductor.controller.actions import ActionCatalog
from conductor.controller.artifacts import atomic_json, resolve_checkpoint
from conductor.controller.factory import build_controller
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.training.data import batches, read_records, split_by_task
from conductor.training.state import (canonical_hash, load_training_state, require_fresh_training_output,
                                      restore_rng, save_training_checkpoint)
from conductor.utils.runs import Run, log_event, seed_everything, write_json


def checkpoint_digest(path: str | Path) -> str:
    directory = resolve_checkpoint(path)
    digest = hashlib.sha256()
    for file in sorted(directory.rglob("*")):
        if file.is_file() and (file.name in {"model.pt", "head.pt", "controller.json", "config.json"}
                               or file.suffix == ".safetensors" or "adapter" in file.parts):
            digest.update(str(file.relative_to(directory)).encode())
            with file.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
    return digest.hexdigest()


def _distributed(config: dict[str, Any]) -> tuple[int, int, bool]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    initialized = False
    if world > 1:
        device = torch.device(config.get("model", {}).get("device", "cpu"))
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            local = int(os.environ.get("LOCAL_RANK", "0"))
            config["model"]["device"] = f"cuda:{local}"
            from conductor.utils.hardware import validate_device
            validate_device(f"cuda:{local}", config["model"].get("dtype", "float32"), require_cuda=True)
            torch.cuda.set_device(local)
        elif device.type != "cpu":
            raise ValueError("torchrun training supports CUDA/NCCL or CPU/Gloo, not MPS")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=300))
            initialized = True
    return rank, world, initialized


def _log_probabilities(controller: Any, records: list[dict[str, Any]], k: int
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    logits = controller.forward_states([ExecutionState(**record["state"]) for record in records], k=k).float()
    probabilities = logits.log_softmax(-1)
    index = torch.arange(len(records), device=controller.device)
    chosen = torch.tensor([controller.catalog.index(record["chosen"]) for record in records], device=controller.device)
    rejected = torch.tensor([controller.catalog.index(record["rejected"]) for record in records], device=controller.device)
    return probabilities[index, chosen], probabilities[index, rejected]


def _reference_cache(config: dict[str, Any], checkpoint: str, records: list[dict[str, Any]],
                     batch_size: int, k: int, output: Path, rank: int, world: int,
                     resumed: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    digest = checkpoint_digest(checkpoint)
    expected_records = canonical_hash(records)
    cache = None
    if resumed is not None:
        cache = resumed.get("reference_cache")
        if cache is None or cache["sft_checkpoint_sha256"] != digest or cache["records_sha256"] != expected_records or cache["k"] != k:
            raise ValueError("DPO resume reference/cache/state identity changed")
    elif rank == 0:
        reference = build_controller(config, checkpoint)
        if reference.stage != "sft":
            raise ValueError("DPO requires the exact SFT checkpoint, not a pretrained/random/preference model")
        head_metadata = (reference.model.head.metadata() if hasattr(reference.model.head, "metadata") else
                         {"head_type": "catalog", "action_catalog_size": len(reference.catalog)})
        support_identity = {"head": head_metadata, "k": k,
                            "state_serialization": getattr(reference, "state_serialization", "public_feature_v1"),
                            "feature_version": getattr(reference, "feature_version", None)}
        entries = []
        with torch.inference_mode():
            for batch in batches(records, batch_size):
                good, bad = _log_probabilities(reference, batch, k)
                for record, chosen, rejected in zip(batch, good.cpu().tolist(), bad.cpu().tolist()):
                    entries.append({"task_id": record.get("task_id"), "state_sha256": canonical_hash(record["state"]),
                                    "chosen_action_sha256": canonical_hash(ActionCatalog.key(
                                        RoutingDecision(**record["chosen"]))),
                                    "rejected_action_sha256": canonical_hash(ActionCatalog.key(
                                        RoutingDecision(**record["rejected"]))),
                                    "chosen_logp": chosen, "rejected_logp": rejected})
        cache = {"sft_checkpoint_sha256": digest, "records_sha256": expected_records, "k": k,
                 "policy_support": support_identity, "policy_support_sha256": canonical_hash(support_identity),
                 "records": entries}
        del reference
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if world > 1:
        payload = [cache]
        dist.broadcast_object_list(payload, src=0)
        cache = payload[0]
    assert cache is not None
    if rank == 0:
        atomic_json(output / "reference_log_probabilities.json", cache)
    for record, entry in zip(records, cache["records"]):
        if canonical_hash(record["state"]) != entry["state_sha256"]:
            raise ValueError("reference cache order/state mismatch")
        if (canonical_hash(ActionCatalog.key(RoutingDecision(**record["chosen"]))) != entry["chosen_action_sha256"]
                or canonical_hash(ActionCatalog.key(RoutingDecision(**record["rejected"])))
                != entry["rejected_action_sha256"]):
            raise ValueError("reference cache complete-action identity mismatch")
        record["reference_chosen_logp"] = entry["chosen_logp"]
        record["reference_rejected_logp"] = entry["rejected_logp"]
    return digest, cache


def _losses(controller: Any, records: list[dict[str, Any]], stage: str, k: int,
            beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if stage == "sft":
        logits = controller.forward_states([ExecutionState(**record["state"]) for record in records], k=k).float()
        labels = torch.tensor([controller.catalog.index(record["decision"]) for record in records], device=controller.device)
        losses = F.cross_entropy(logits, labels, reduction="none")
        if getattr(controller, "last_auxiliary_loss", None) is not None:
            losses = losses + float(controller.config["training"].get("router_auxiliary_weight", 0)) * controller.last_auxiliary_loss
        return losses, (logits.argmax(-1) == labels).float(), torch.zeros(len(records), device=controller.device)
    chosen, rejected = _log_probabilities(controller, records, k)
    reference_chosen = torch.tensor([record["reference_chosen_logp"] for record in records], device=controller.device)
    reference_rejected = torch.tensor([record["reference_rejected_logp"] for record in records], device=controller.device)
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    losses = -F.logsigmoid(margin)
    if getattr(controller, "last_auxiliary_loss", None) is not None:
        losses = losses + float(controller.config["training"].get("router_auxiliary_weight", 0)) * controller.last_auxiliary_loss
    return losses, (chosen > rejected).float(), margin.detach()


def _validate_public_action_support(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    state = ExecutionState(**record["state"])
    calls = state.remaining_budget.get("agent_calls", 0)
    tokens = state.remaining_budget.get("tokens", 0)
    if (isinstance(calls, bool) or isinstance(tokens, bool) or not isinstance(calls, (int, float))
            or not isinstance(tokens, (int, float)) or not math.isfinite(calls) or not math.isfinite(tokens)
            or calls < 0 or tokens < 0):
        raise ValueError("training states require finite nonnegative public budgets")
    for field in fields:
        decision = RoutingDecision(**record[field]).validate(len(AGENT_NAMES))
        if len(decision.selected_agents) > int(calls) or (tokens < 1 and not decision.terminate):
            raise ValueError(f"{field} action violates its public state budget support")


def _aggregate(values: list[float], device: torch.device, world: int) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _execution_identity(controller: Any, world: int) -> list[dict[str, Any]]:
    local = {"device_type": controller.device.type, "processor": platform.processor(), "machine": platform.machine(),
             "cpu_kernel_capability": torch.backends.cpu.get_cpu_capability(),
             "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
             "float32_matmul_precision": torch.get_float32_matmul_precision(),
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             "cuda_runtime": torch.version.cuda if controller.device.type == "cuda" else None}
    if controller.device.type == "cuda":
        local.update(gpu_name=controller.hardware_validation.get("name"),
                     gpu_compute_capability=controller.hardware_validation.get("compute_capability"),
                     cudnn_version=torch.backends.cudnn.version(),
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                     cudnn_benchmark=torch.backends.cudnn.benchmark,
                     nccl_version=list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None)
    per_rank: list[Any] = [None] * world
    if world > 1:
        dist.all_gather_object(per_rank, local)
    else:
        per_rank[0] = local
    return per_rank


def _validate(controller: Any, records: list[dict[str, Any]], stage: str, batch_size: int,
              k: int, beta: float, rank: int = 0, world: int = 1) -> dict[str, float | None]:
    # Bypass DDP forwards for uneven validation shards; only the final reduction
    # is collective. This prevents buffer-broadcast deadlocks on short shards.
    wrapped = controller.model
    controller.model = wrapped.module if isinstance(wrapped, DistributedDataParallel) else wrapped
    controller.model.eval()
    loss_sum = accuracy_sum = count = 0.0
    try:
        with torch.inference_mode():
            for batch in batches(records[rank::world], batch_size):
                losses, correct, _ = _losses(controller, batch, stage, k, beta)
                loss_sum += float(losses.sum())
                accuracy_sum += float(correct.sum())
                count += len(batch)
        loss_sum, accuracy_sum, count = _aggregate([loss_sum, accuracy_sum, count], controller.device, world)
    finally:
        controller.model = wrapped
    return {"loss": loss_sum / count if count else None, "accuracy": accuracy_sum / count if count else None}


def _scheduler(optimizer: Any, training: dict[str, Any], total_steps: int) -> Any:
    kind = training.get("scheduler", "constant")
    warmup = int(training.get("warmup_steps", 0))
    if kind not in {"constant", "linear", "cosine"} or warmup < 0:
        raise ValueError("scheduler must be constant/linear/cosine and warmup_steps nonnegative")
    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(max((step - warmup) / max(total_steps - warmup, 1), 0.0), 1.0)
        return 1.0 if kind == "constant" else (1 - progress if kind == "linear" else .5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(config: dict[str, Any], checkpoint: str | None = None, resume: str | None = None) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    rank, world, initialized = _distributed(effective)
    try:
        return _train(effective, checkpoint, resume, rank, world)
    finally:
        if initialized:
            dist.destroy_process_group()


def _train(config: dict[str, Any], checkpoint: str | None, resume: str | None, rank: int, world: int) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    training = config["training"]
    stage = training.get("stage", "sft")
    if stage not in {"sft", "dpo"}:
        raise ValueError("training.stage must be sft or dpo")
    checkpoint = checkpoint or training.get("checkpoint")
    resume = resume or training.get("resume")
    resumed = load_training_state(resume) if resume else None
    if stage == "dpo" and checkpoint is None:
        raise ValueError("DPO requires --checkpoint pointing to an SFT controller")
    output = Path(training.get("output", f"outputs/checkpoints/conductor-{stage}"))
    if checkpoint is not None and Path(checkpoint).resolve() == output.resolve():
        raise ValueError("training output must not overwrite its input/reference checkpoint")
    if not resume:
        require_fresh_training_output(output)
    output.mkdir(parents=True, exist_ok=True)
    run = Run(output, config, checkpoint) if rank == 0 else None
    records = read_records(training["data"], stage)
    source_hash = hashlib.sha256(Path(training["data"]).read_bytes()).hexdigest()
    batch_size = int(training.get("batch_size", 32))
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    epochs = int(training.get("epochs", 40))
    k = int(config.get("routing", {}).get("k", 2))
    beta = float(training.get("beta", .1))
    if batch_size < 1 or accumulation < 1 or epochs < 1 or beta <= 0:
        raise ValueError("batch size, accumulation, epochs and DPO beta must be positive")
    if float(training.get("learning_rate", .003)) <= 0 or float(training.get("max_grad_norm", 1)) <= 0 or float(training.get("weight_decay", .01)) < 0:
        raise ValueError("learning_rate/max_grad_norm must be positive and weight_decay nonnegative")
    if any(int(training.get(key, 0)) < 0 for key in ("save_every_updates", "save_every_epochs", "warmup_steps")):
        raise ValueError("checkpoint/warmup intervals must be nonnegative")
    if training.get("stop_after_updates") is not None and int(training["stop_after_updates"]) < 1:
        raise ValueError("stop_after_updates must be positive")
    original_examples = len(records)
    legal = []
    for record in records:
        fields = ("decision",) if stage == "sft" else ("chosen", "rejected")
        _validate_public_action_support(record, fields)
        decisions = [RoutingDecision(**record[field]) for field in fields]
        for decision in decisions:
            decision.validate(len(AGENT_NAMES))
        if all(len(decision.selected_agents) <= k for decision in decisions):
            legal.append(record)
    records = legal
    if not records:
        raise ValueError(f"no legal {stage} records for configured k={k}")
    filtered_hash = canonical_hash(records)
    excluded = original_examples - len(records)
    if excluded and rank == 0:
        log_event("training_labels_excluded", reason="action exceeds configured k", count=excluded, k=k)
    explicit_validation = None
    validation_hash = None
    if training.get("validation_data"):
        path = Path(training["validation_data"])
        validation_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        explicit_validation = read_records(path, stage, allowed_splits=("dev", "development", "validation"))
        fit_ids = {record["task_id"] for record in records}
        validation_ids = {record["task_id"] for record in explicit_validation}
        fit_groups = {record.get("source_group", record["task_id"]) for record in records}
        validation_groups = {record.get("source_group", record["task_id"]) for record in explicit_validation}
        if fit_ids & validation_ids or fit_groups & validation_groups:
            raise ValueError("explicit development validation overlaps fitting task/source groups")
        for record in explicit_validation:
            fields = ("decision",) if stage == "sft" else ("chosen", "rejected")
            _validate_public_action_support(record, fields)
            for field in fields:
                RoutingDecision(**record[field]).validate(k)
    reference_digest = None
    reference_cache = None
    if stage == "dpo":
        reference_digest, reference_cache = _reference_cache(config, checkpoint,
            records + (explicit_validation or []), batch_size, k, output, rank, world, resumed)
    controller = build_controller(config, resume or checkpoint, training=True)
    controller.config["training"] = dict(training)
    controller.config["seed"] = seed
    controller.config["routing"] = dict(config.get("routing", {"k": k}))
    for record in records:
        for field in (("decision",) if stage == "sft" else ("chosen", "rejected")):
            RoutingDecision(**record[field]).validate(k)
            controller.catalog.index(record[field])
        if stage == "dpo" and controller.catalog.index(record["chosen"]) == controller.catalog.index(record["rejected"]):
            raise ValueError("preference actions must differ")
    if explicit_validation is None:
        train_records, validation = split_by_task(records, float(training.get("validation_fraction", .2)), seed)
    else:
        train_records, validation = records, explicit_validation
    partitions = {"train_task_ids": sorted({str(record.get("task_id", record["state"]["user_task"])) for record in train_records}),
                  "validation_task_ids": sorted({str(record.get("task_id", record["state"]["user_task"])) for record in validation})}
    identity = {"stage": stage, "seed": seed, "world_size": world, "dataset_sha256": source_hash,
                "filtered_records_sha256": filtered_hash, "partition_sha256": canonical_hash(partitions),
                "reference_checkpoint_sha256": reference_digest, "k": k, "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation, "beta": beta,
                "learning_rate": float(training.get("learning_rate", .003)), "weight_decay": float(training.get("weight_decay", .01)),
                "max_grad_norm": float(training.get("max_grad_norm", 1)), "scheduler": training.get("scheduler", "constant"),
                "warmup_steps": int(training.get("warmup_steps", 0)), "dtype": str(controller.dtype),
                "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
                "router_auxiliary_weight": float(training.get("router_auxiliary_weight", 0)),
                "device_type": controller.device.type,
                "execution_device_type": controller.device.type,
                "per_rank_execution_hardware": _execution_identity(controller, world),
                "model_sha256": canonical_hash({key: value for key, value in controller.config["model"].items() if key not in {"device", "require_cuda"}})}
    if validation_hash is not None:
        identity["validation_dataset_sha256"] = validation_hash
    if controller.config["model"].get("backend") == "hf":
        identity["hf_versions"] = {"transformers": version("transformers"), "peft": version("peft"), "tokenizers": version("tokenizers")}
    if identity["router_auxiliary_weight"] < 0 or (identity["router_auxiliary_weight"] and controller.config["model"].get("backend") != "hf"):
        raise ValueError("router_auxiliary_weight must be nonnegative and is implemented only for the HF MoE backend")
    if identity["scheduler"] != "constant":
        identity["planned_epochs"] = epochs
    if resumed and resumed["identity"] != identity:
        changed = [key for key in identity if identity[key] != resumed["identity"].get(key)]
        raise ValueError(f"exact resume configuration/data/reference identity changed: {changed}")
    parameters = [parameter for parameter in controller.model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("training requires nonempty FP32 coordinator head/adapter parameters")
    optimizer = torch.optim.AdamW(parameters, lr=identity["learning_rate"], weight_decay=identity["weight_decay"])
    steps_per_epoch = math.ceil(len(train_records) / (batch_size * world * accumulation))
    scheduler = _scheduler(optimizer, training, steps_per_epoch * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=controller.device.type == "cuda" and controller.dtype == torch.float16)
    cursor = {"epoch": 0, "offset": 0, "step": 0, "epoch_loss": 0.0, "epoch_count": 0, "epoch_margin": 0.0, "skipped_updates": 0}
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        scaler.load_state_dict(resumed["scaler"])
        cursor = dict(resumed["rank_states"][rank]["cursor"])
    if rank == 0:
        assert run is not None
        run.record.update(resolved_model=dict(controller.config["model"]), distributed={"rank": rank, "world_size": world},
                          resumed_from=resume, data_identity=identity)
        write_json(output / "run.json", run.record)
        write_json(output / "data_partition.json", partitions)
        for batch in batches(records[:128], batch_size):
            controller.batch_route([ExecutionState(**record["state"]) for record in batch], k)
        write_json(output / ("expert_statistics_resume_start.json" if resume else "expert_statistics_before.json"), controller.expert_stats())
        controller.reset_expert_stats()
    initial = _validate(controller, train_records, stage, batch_size, k, beta, rank, world)
    raw_model = controller.model
    if world > 1:
        controller.model = DistributedDataParallel(raw_model, device_ids=[controller.device.index] if controller.device.type == "cuda" else None,
                                                  find_unused_parameters=True, broadcast_buffers=False)
    if resumed:
        restore_rng(resumed["rank_states"][rank]["rng"])
    else:
        seed_everything(seed + rank)
    save_steps = int(training.get("save_every_updates", 0))
    save_epochs = int(training.get("save_every_epochs", 0))
    stop_after = training.get("stop_after_updates")
    completed = True
    checkpoint_path = None
    metrics_path = output / "training_metrics.jsonl"
    handle = metrics_path.open("a" if resume else "w") if rank == 0 else None
    try:
        while cursor["epoch"] < epochs:
            order = list(range(len(train_records)))
            random.Random(seed + cursor["epoch"]).shuffle(order)
            controller.model.train()
            while cursor["offset"] < len(order):
                window_end = min(cursor["offset"] + batch_size * world * accumulation, len(order))
                window_count = window_end - cursor["offset"]
                optimizer.zero_grad(set_to_none=True)
                for begin in range(cursor["offset"], window_end, batch_size * world):
                    group = order[begin:min(begin + batch_size * world, window_end)]
                    local_ids = group[rank::world]
                    local = [train_records[index] for index in local_ids] if local_ids else [train_records[group[0]]]
                    weights = torch.ones(len(local), device=controller.device) if local_ids else torch.zeros(1, device=controller.device)
                    losses, _, margin = _losses(controller, local, stage, k, beta)
                    if not bool(torch.isfinite(losses).all()):
                        raise FloatingPointError("non-finite training loss")
                    # DDP averages gradients across ranks. Multiplying by world
                    # recovers the sum, then divides by actual window examples.
                    loss = (losses * weights).sum() * world / window_count
                    scaler.scale(loss).backward()
                    cursor["epoch_loss"] += float((losses.detach() * weights).sum())
                    cursor["epoch_margin"] += float((margin * weights).sum())
                    cursor["epoch_count"] += len(local_ids)
                scaler.unscale_(optimizer)
                finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in parameters if parameter.grad is not None)
                finite_flag = torch.tensor(int(finite), device=controller.device)
                if world > 1:
                    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                if not int(finite_flag):
                    if not scaler.is_enabled():
                        raise FloatingPointError("non-finite unscaled coordinator gradients")
                    cursor["skipped_updates"] += 1
                    if rank == 0:
                        log_event("amp_update_skipped", step=cursor["step"] + 1, reason="non-finite FP16 gradients")
                    scaler.step(optimizer)  # GradScaler recorded the overflow during unscale.
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(parameters, identity["max_grad_norm"], error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                cursor["offset"] = window_end
                cursor["step"] += 1
                stopping = stop_after is not None and cursor["step"] >= int(stop_after)
                if stopping or (save_steps and cursor["step"] % save_steps == 0):
                    wrapped = controller.model
                    controller.model = raw_model
                    checkpoint_path = save_training_checkpoint(output, controller, optimizer, scheduler, scaler, cursor, identity, rank, world, reference_cache)
                    controller.model = wrapped
                if stopping:
                    completed = False
                    break
            if not completed:
                break
            total_loss, total_margin, count = _aggregate([cursor["epoch_loss"], cursor["epoch_margin"], cursor["epoch_count"]], controller.device, world)
            validation_metrics = _validate(controller, validation, stage, batch_size, k, beta, rank, world)
            record = {"epoch": cursor["epoch"] + 1, "train_loss": total_loss / count,
                      "validation_loss": validation_metrics["loss"], "validation_accuracy": validation_metrics["accuracy"],
                      "optimizer_windows": cursor["step"], "learning_rate": optimizer.param_groups[0]["lr"]}
            if stage == "dpo":
                record["mean_dpo_margin"] = total_margin / count
            if rank == 0:
                assert handle is not None
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if cursor["epoch"] == 0 or (cursor["epoch"] + 1) % max(1, epochs // 10) == 0:
                    log_event("training_epoch", stage=stage, **record)
            cursor.update(epoch=cursor["epoch"] + 1, offset=0, epoch_loss=0.0, epoch_count=0, epoch_margin=0.0)
            if save_epochs and cursor["epoch"] % save_epochs == 0:
                wrapped = controller.model
                controller.model = raw_model
                checkpoint_path = save_training_checkpoint(output, controller, optimizer, scheduler, scaler, cursor, identity, rank, world, reference_cache)
                controller.model = wrapped
    finally:
        if handle is not None:
            handle.close()
        controller.model = raw_model
    final_train = _validate(controller, train_records, stage, batch_size, k, beta, rank, world)
    final_validation = _validate(controller, validation, stage, batch_size, k, beta, rank, world)
    checkpoint_path = save_training_checkpoint(output, controller, optimizer, scheduler, scaler, cursor, identity, rank, world, reference_cache)
    metrics = {"stage": stage, "epochs": epochs, "completed": completed, "optimizer_windows": cursor["step"],
               "skipped_amp_updates": cursor["skipped_updates"], "world_size": world, "gradient_accumulation_steps": accumulation,
               "train_examples": len(train_records), "validation_examples": len(validation), "source_examples": original_examples,
               "excluded_actions_exceeding_k": excluded, "train_k": k, "initial_train": initial, "final_train": final_train,
               "final_validation": final_validation, "trainable_parameters": sum(parameter.numel() for parameter in parameters),
               "total_controller_parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
               "reference_checkpoint_sha256": reference_digest, "dataset_sha256": source_hash,
               "pretrained": controller.config.get("model", {}).get("backend") == "hf" and controller.pretrained,
               "checkpoint": str(output), "resume_checkpoint": str(checkpoint_path), "specialists_updated": False}
    if validation_hash is not None:
        metrics["validation_dataset_sha256"] = validation_hash
    if rank == 0:
        # Compatibility root files are exports; normal loaders use the atomically
        # committed pointer, so concurrent readers never observe mixed weights.
        controller.save(output, "sft" if stage == "sft" else "preference")
        shutil.copy2(checkpoint_path / "training_state.pt", output / "optimizer.pt")
        atomic_json(output / "checkpoint_pointer.json", {"path": str(checkpoint_path.relative_to(output))})
        controller.reset_expert_stats()
        for batch in batches(records[:128], batch_size):
            controller.batch_route([ExecutionState(**record["state"]) for record in batch], k)
        write_json(output / "expert_statistics.json", controller.expert_stats())
        assert run is not None
        run.finish(metrics)
    return metrics
