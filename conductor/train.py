"""python -m conductor.train --config ... [--checkpoint SFT_DIRECTORY]"""
from __future__ import annotations

import argparse

from conductor.training.runner import train
from conductor.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train only the learned Conductor coordinator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="SFT model/reference for DPO or initialization for SFT")
    args = parser.parse_args()
    train(load_config(args.config), args.checkpoint)


if __name__ == "__main__":
    main()
