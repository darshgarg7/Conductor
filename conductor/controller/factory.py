"""Backend selection is configuration, independent of orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conductor.controller.tiny import TinyController
from conductor.schema import Policy
from conductor.utils.runs import seed_everything


def build_controller(config: dict[str, Any], checkpoint: str | None = None) -> Policy:
    seed_everything(int(config.get("seed", 42)))
    backend = config.get("model", {}).get("backend", "tiny")
    if checkpoint is not None:
        backend = json.loads((Path(checkpoint) / "controller.json").read_text())["backend"]
    if backend == "tiny":
        return TinyController.load(checkpoint, config) if checkpoint else TinyController(config)
    if backend == "hf":
        from conductor.controller.hf import HFController
        return HFController.load(checkpoint, config) if checkpoint else HFController(config)
    raise ValueError(f"unsupported controller backend {backend!r}")
