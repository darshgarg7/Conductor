"""Backend selection is configuration, independent of orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conductor.controller.tiny import TinyController
from conductor.controller.artifacts import resolve_checkpoint
from conductor.schema import Policy
from conductor.utils.runs import seed_everything


def build_controller(config: dict[str, Any], checkpoint: str | None = None, *, training: bool = False) -> Policy:
    seed_everything(int(config.get("seed", 42)))
    backend = config.get("model", {}).get("backend", "tiny")
    if checkpoint is not None:
        checkpoint = str(resolve_checkpoint(checkpoint))
        backend = json.loads((Path(checkpoint) / "controller.json").read_text())["backend"]
    if backend == "tiny":
        controller = TinyController.load(checkpoint, config) if checkpoint else TinyController(config)
    elif backend == "hf":
        from conductor.controller.hf import HFController
        controller = HFController.load(checkpoint, config) if checkpoint else HFController(config)
    elif backend == "cheap":
        from conductor.controller.cheap import CheapController
        controller = CheapController.load(checkpoint, config) if checkpoint else CheapController(config)
    else:
        raise ValueError(f"unsupported controller backend {backend!r}")
    if training:
        controller.enable_training(bool(config.get("training", {}).get("gradient_checkpointing", False)))
    else:
        for parameter in controller.model.parameters():
            parameter.requires_grad_(False)
        controller.model.eval()
        compile_config = config.get("inference", {}).get("compile", {})
        if compile_config.get("enabled", False):
            controller.compile_for_inference(backend=compile_config.get("backend", "inductor"),
                                             mode=compile_config.get("mode", "default"),
                                             fullgraph=bool(compile_config.get("fullgraph", False)))
    return controller
