"""Trusted local resume state, committed only at optimizer-update boundaries."""
from __future__ import annotations

import hashlib
import json
import random
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from conductor.controller.artifacts import atomic_directory, atomic_json


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_training_checkpoint(output: Path, controller: Any, optimizer: Any, scheduler: Any, scaler: Any,
                             cursor: dict[str, Any], identity: dict[str, Any], rank: int, world_size: int,
                             reference_cache: dict[str, Any] | None = None) -> Path:
    destination = output / "resume" / f"step-{cursor['step']:08d}-epoch-{cursor['epoch']:04d}"
    local = {"rng": capture_rng(), "cursor": dict(cursor),
             "execution_hardware": identity["per_rank_execution_hardware"][rank]}
    states: list[Any] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        if not destination.exists():
            with atomic_directory(destination) as temporary:
                controller.save(temporary, "sft" if identity["stage"] == "sft" else "preference")
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "rank_states": states, "identity": identity,
                            "identity_sha256": canonical_hash(identity), "torch_version": torch.__version__,
                            "reference_cache": reference_cache}, temporary / "training_state.pt")
                checksums = {str(file.relative_to(temporary)): _file_hash(file)
                             for file in sorted(temporary.rglob("*")) if file.is_file()}
                for file in temporary.rglob("*"):
                    if file.is_file():
                        with file.open("rb") as handle:
                            os.fsync(handle.fileno())
                atomic_json(temporary / "resume_manifest.json", {"identity": identity,
                            "identity_sha256": canonical_hash(identity), "cursor": cursor,
                            "world_size": world_size, "torch_version": torch.__version__,
                            "files_sha256": checksums,
                            "checkpoint_boundary": "after complete gradient-accumulation window; no partial gradients"})
        atomic_json(output / "latest_resume.json", {"path": str(destination.relative_to(output))})
    if world_size > 1:
        dist.barrier()
    return destination


def load_training_state(path: str | Path) -> dict[str, Any]:
    # Optimizer/RNG files intentionally contain Python/NumPy state. Only resume
    # trusted checkpoints produced by this project, never arbitrary downloads.
    directory = Path(path)
    manifest = json.loads((directory / "resume_manifest.json").read_text())
    for relative, expected in manifest.get("files_sha256", {}).items():
        if _file_hash(directory / relative) != expected:
            raise ValueError(f"resume checkpoint file checksum changed: {relative}")
    value = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if value["torch_version"] != torch.__version__:
        raise ValueError("exact resume requires the same PyTorch version")
    if canonical_hash(value["identity"]) != value["identity_sha256"]:
        raise ValueError("resume checkpoint identity hash is invalid")
    return value
