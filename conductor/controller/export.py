"""Export a self-contained merged HF coordinator for serving."""
from __future__ import annotations

import argparse
import json

from conductor.controller.artifacts import resolve_checkpoint
from conductor.controller.factory import build_controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge coordinator LoRA, retain source adapter, verify and export")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()
    metadata = json.loads((resolve_checkpoint(args.checkpoint) / "controller.json").read_text())
    configuration = metadata["configuration"]
    configuration["model"].update(device=args.device, dtype=args.dtype, require_cuda=args.device.startswith("cuda"))
    controller = build_controller(configuration, args.checkpoint)
    if not hasattr(controller, "export_merged"):
        raise ValueError("merged LoRA export requires an HF coordinator")
    controller.export_merged(args.output)


if __name__ == "__main__":
    main()
