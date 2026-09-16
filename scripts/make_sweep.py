"""Materialize seeded top-k sweeps with isolated experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from conductor.utils.sweeps import MODULES, materialize_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--module", choices=MODULES, default="conductor.train")
    parser.add_argument("--run-root", help="Root for distinct experiment artifacts, separate from the YAML directory")
    args = parser.parse_args()
    paths = materialize_sweep(args.config, args.output, module=args.module, seeds=args.seeds, top_k=args.top_k,
                              run_root=args.run_root)
    print(f"Materialized {len(paths)} {args.module} configurations: {Path(args.output) / 'manifest.txt'}")


if __name__ == "__main__":
    main()
