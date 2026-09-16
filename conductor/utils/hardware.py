"""Explicit execution-device validation; unavailable CUDA is never a CPU result."""
from __future__ import annotations

import platform
from typing import Any

import torch


def dtype_name(dtype: str | torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("supported compute dtypes: float32, float16, bfloat16")
    return name


def validate_device(device: str | torch.device, dtype: str | torch.dtype = "float32",
                    require_cuda: bool = False) -> dict[str, Any]:
    target = torch.device(device)
    precision = dtype_name(dtype)
    if require_cuda and target.type != "cuda":
        raise ValueError("this run requires a real NVIDIA CUDA device")
    if target.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported execution device: {target}")
    result: dict[str, Any] = {"requested_device": str(target), "compute_dtype": precision,
                              "validated": True, "nvidia_cuda": target.type == "cuda"}
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; install a CUDA PyTorch build on an NVIDIA host")
        index = target.index if target.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is outside the visible device count")
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            if precision == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("native bfloat16 is unsupported on the requested NVIDIA device; select float16/float32")
        result.update(device=f"cuda:{index}", name=props.name, total_memory_bytes=props.total_memory,
                      compute_capability=[props.major, props.minor], cuda_runtime=torch.version.cuda,
                      cudnn_version=torch.backends.cudnn.version())
    elif target.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        if precision != "float32":
            raise ValueError("validated MPS path uses float32; reduced precision requires a separate compatibility run")
        result.update(device="mps", name="Apple Metal", cuda_runtime=None)
    else:
        if precision == "float16":
            raise ValueError("float16 CPU execution is unsupported by Conductor; select float32 or bfloat16")
        result.update(device="cpu", name=platform.processor(), cuda_runtime=None)
    return result


def hardware_info() -> dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "memory_bytes": props.total_memory,
                            "compute_capability": [props.major, props.minor]})
    return {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "visible_nvidia_devices": devices,
            "mps_available": torch.backends.mps.is_available()}
