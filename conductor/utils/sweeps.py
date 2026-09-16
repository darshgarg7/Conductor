"""Materialize seeded top-k sweeps with isolated experiment outputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml
from conductor.utils.config import load_config

MODULES = ("conductor.train", "conductor.evaluate", "conductor.benchmark")


def materialize_sweep(config_path: str | Path, output: str | Path, *, module: str = "conductor.train",
                      seeds: list[int], top_k: list[int], run_root: str | Path | None = None) -> list[Path]:
    """Write explicit consumer-specific k values and declare their launcher.

    Evaluation reads top-level k; inference benchmarks read k_values. Merely
    changing routing.k would silently leave those experiments unchanged.
    """
    if module not in MODULES:
        raise ValueError(f"sweep module must be one of {MODULES}")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    if not top_k or len(set(top_k)) != len(top_k) or any(type(k) is not int or not 1 <= k <= 3 for k in top_k):
        raise ValueError("top-k values must be unique integers from 1 through 3")
    config = load_config(config_path)
    if module == "conductor.train" and not isinstance(config.get("training"), dict):
        raise ValueError("training sweeps require a training configuration; select --module for evaluation/benchmarking")
    catalog_max = config.get("model", {}).get("max_agents", 3)
    available = len(config.get("agents", {}).get("names", range(8)))
    if max(top_k) > min(catalog_max, available):
        raise ValueError("top-k exceeds the configured action catalog or available specialist count")
    folder = Path(output)
    if "\n" in str(folder.resolve()):
        raise ValueError("manifest paths must not contain newlines")
    variant_paths = [folder / f"seed{seed}-k{k}.yaml" for seed in seeds for k in top_k]
    if any(path.exists() for path in [folder / "manifest.txt", folder / "manifest.json", *variant_paths]):
        raise FileExistsError("sweep configurations already exist; use a fresh output directory")
    namespace = Path(run_root) if run_root is not None else Path("outputs/sweeps") / folder.name / module.rsplit(".", 1)[1]
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        for k in top_k:
            variant: dict[str, Any] = copy.deepcopy(config)
            variant["seed"] = seed
            variant["sweep"] = {"version": 1, "module": module, "seed": seed, "agent_top_k": k}
            variant.setdefault("routing", {})["k"] = k
            tag = f"seed{seed}-k{k}"
            if module == "conductor.train":
                variant["training"]["output"] = str(namespace / tag / "checkpoint")
            elif module == "conductor.evaluate":
                variant["k"] = k
                variant["output"] = str(namespace / tag)
            else:
                variant["k_values"] = [k]
                variant["output"] = str(namespace / tag)
            path = folder / f"{tag}.yaml"
            path.write_text(yaml.safe_dump(variant, sort_keys=False))
            paths.append(path.resolve())
    (folder / "manifest.txt").write_text("\n".join(map(str, paths)) + "\n")
    (folder / "manifest.json").write_text(json.dumps({"version": 1, "module": module,
                                                    "paths": list(map(str, paths))}, indent=2) + "\n")
    return paths


