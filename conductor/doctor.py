"""Machine-readable compatibility check and an optional real-device arithmetic probe."""
from __future__ import annotations

import argparse
import time
from typing import Any

import torch
from conductor.utils.hardware import hardware_info, validate_device
from conductor.utils.runs import write_json


def inspect(device: str, dtype: str, require_cuda: bool = False, probe: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"hardware": hardware_info(), "status": "compatible"}
    try:
        record["validation"] = validate_device(device, dtype, require_cuda)
        if probe:
            target = torch.device(record["validation"]["device"])
            with torch.inference_mode():
                tensor = torch.randn(64, 64, device=target, dtype=getattr(torch, dtype))
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                elif target.type == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                product = tensor @ tensor.T
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                elif target.type == "mps":
                    torch.mps.synchronize()
                record["probe"] = {"seconds": time.perf_counter() - started,
                                   "finite": bool(torch.isfinite(product).all()),
                                   "scope": "compatibility arithmetic probe, not a model performance benchmark"}
                if not record["probe"]["finite"]:
                    raise RuntimeError("device probe produced non-finite results")
    except (ValueError, RuntimeError) as error:
        record.update(status="incompatible", reason=str(error))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--output", default="outputs/doctor.json")
    args = parser.parse_args()
    record = inspect(args.device, args.dtype, args.require_cuda, args.probe)
    write_json(args.output, record)
    print(record)
    raise SystemExit(0 if record["status"] == "compatible" else 1)


if __name__ == "__main__":
    main()
