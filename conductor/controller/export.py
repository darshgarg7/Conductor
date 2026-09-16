"""Export a self-contained merged HF coordinator for serving.

--in-place avoids cloning the full backbone, consumes the loaded controller's
adapter capability, and retains the unchanged source adapter checkpoint on disk.
Safe merge still requires temporary memory per adapted layer.
"""
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
    parser.add_argument("--in-place", action="store_true",
                        help="merge without a full backbone copy; consumes only the loaded controller and preserves the on-disk source checkpoint")
    args = parser.parse_args()
    metadata = json.loads((resolve_checkpoint(args.checkpoint) / "controller.json").read_text())
    configuration = metadata["configuration"]
    configuration["model"].update(device=args.device, dtype=args.dtype, require_cuda=args.device.startswith("cuda"))
    controller = build_controller(configuration, args.checkpoint)
    if not hasattr(controller, "export_merged"):
        raise ValueError("merged LoRA export requires an HF coordinator")
    controller.export_merged(args.output, preserve_model=not args.in_place)


if __name__ == "__main__":
    main()
