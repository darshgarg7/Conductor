"""Run provenance and JSON logging. No synthetic experimental metrics."""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("CONDUCTOR_TORCH_THREADS", "1")))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def log_event(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


class Run:
    def __init__(self, output: str | Path, config: dict[str, Any], checkpoint: str | None = None) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            commit, dirty = None, None
        from conductor.utils.hardware import hardware_info
        library_versions = {}
        for package in ("torch", "transformers", "peft", "accelerate", "numpy", "PyYAML", "fastapi", "uvicorn"):
            try:
                library_versions[package] = version(package)
            except PackageNotFoundError:
                library_versions[package] = None
        self.record = {
            "configuration": config, "git_commit": commit, "git_dirty": dirty,
            "seed": config.get("seed", 42), "checkpoint": checkpoint,
            "hardware": {**hardware_info(), "cpu_count": os.cpu_count()},
            "library_versions": library_versions,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        self.tracker = None
        tracking = config.get("tracking", {})
        if tracking.get("enabled", False):
            import wandb
            self.tracker = wandb.init(project=tracking.get("project", "conductor"),
                                      name=tracking.get("name"), config=self.record,
                                      mode=tracking.get("mode", "offline"), dir=str(self.output))
        write_json(self.output / "run.json", self.record)

    def finish(self, metrics: dict[str, Any]) -> None:
        self.record.update(runtime_seconds=time.perf_counter() - self.started, metrics=metrics)
        write_json(self.output / "run.json", self.record)
        write_json(self.output / "metrics.json", metrics)
        if self.tracker is not None:
            self.tracker.log({"runtime_seconds": self.record["runtime_seconds"],
                              **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}})
            self.tracker.finish()
        log_event("run_completed", output=str(self.output), runtime_seconds=self.record["runtime_seconds"])
