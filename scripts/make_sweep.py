"""Materialize seeded top-k sweeps with isolated experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from conductor.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for seed in args.seeds:
        for k in args.top_k:
            config = load_config(args.config)
            config["seed"] = seed
            config.setdefault("routing", {})["k"] = k
            config.setdefault("orchestration", {})["k"] = k
            tag = f"seed{seed}-k{k}"
            if "training" in config:
                config["training"]["output"] = f"outputs/sweeps/{tag}/checkpoint"
            else:
                config["output"] = f"outputs/sweeps/{tag}"
            path = folder / f"{tag}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            paths.append(str(path.resolve()))
    (folder / "manifest.txt").write_text("\n".join(paths) + "\n")
    print(f"Materialized {len(paths)} configurations: {folder / 'manifest.txt'}")


if __name__ == "__main__":
    main()
