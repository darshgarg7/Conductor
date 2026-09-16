"""Content fingerprints used to reject incomparable research measurements."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from conductor.controller.artifacts import resolve_checkpoint


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    directory = resolve_checkpoint(path)
    if not (directory / "controller.json").is_file():
        raise FileNotFoundError(f"Inference checkpoint metadata missing: {directory}")
    # Optimizer/logging artifacts do not define inference weights.
    files = sorted(source for source in directory.rglob("*") if source.is_file()
                   and not {"resume", "source_adapter"}.intersection(source.relative_to(directory).parts)
                   and source.name not in {"optimizer.pt", "training_state.pt"} and (
                       source.name in {"controller.json", "adapter_config.json", "config.json"}
                       or source.suffix in {".pt", ".bin", ".safetensors"}))
    return canonical_hash({str(source.relative_to(directory)): file_sha256(source) for source in files})


def specialist_identity(agents: dict[str, Any], config: dict[str, Any], hash_weights: bool = False) -> dict[str, Any]:
    identities, model_hashes = {}, {}
    for name, agent in agents.items():
        implementations = []
        wrapped = agent
        seen: set[int] = set()
        while id(wrapped) not in seen:
            seen.add(id(wrapped))
            source = inspect.getsourcefile(type(wrapped))
            implementations.append({"class": f"{type(wrapped).__module__}.{type(wrapped).__qualname__}",
                                    "source_sha256": file_sha256(source) if source else None})
            if not hasattr(wrapped, "agent"):
                break
            wrapped = wrapped.agent
        item = {"class": f"{type(agent).__module__}.{type(agent).__qualname__}", "frozen": agent.frozen,
                "capability": agent.capability, "implementations": implementations,
                "configuration_sha256": canonical_hash(getattr(agent, "config", config.get("agents", {})))}
        model = getattr(agent, "model", None)
        if model is not None:
            identifier = id(model)
            if identifier not in model_hashes:
                model_config = getattr(model, "config", None)
                snapshot = {"resolved_revision": getattr(model_config, "_commit_hash", None),
                            "model_name": getattr(model_config, "_name_or_path", None),
                            "parameters": [(key, list(parameter.shape), str(parameter.dtype), parameter.requires_grad,
                                            parameter._version) for key, parameter in model.named_parameters()]}
                if hash_weights:
                    import torch
                    digest = hashlib.sha256()
                    for key, parameter in model.named_parameters():
                        digest.update(key.encode())
                        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    snapshot["weight_sha256"] = digest.hexdigest()
                model_hashes[identifier] = snapshot
            item["model"] = model_hashes[identifier]
        identities[name] = item
    return {"sha256": canonical_hash(identities), "identities": identities,
            "weight_hashing": "full_parameter_bytes" if hash_weights else "revision, shape, dtype, and mutation counters"}
