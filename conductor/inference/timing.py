"""Platform-aware timing and explicitly separated memory measurements."""
from __future__ import annotations

import resource
import sys
import time
from contextlib import contextmanager
from typing import Any


def synchronize(device: Any = None) -> None:
    import torch
    kind = getattr(device, "type", str(device).split(":")[0])
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: Any = None) -> None:
    import torch
    if getattr(device, "type", str(device).split(":")[0]) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_measurements(device: Any = None) -> dict[str, Any]:
    import torch
    try:
        import psutil
        rss, status = psutil.Process().memory_info().rss, "current_process_rss"
    except ImportError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss, status = int(peak * (1 if sys.platform == "darwin" else 1024)), "peak_process_rss"
    cuda = getattr(device, "type", str(device).split(":")[0]) == "cuda"
    return {"cpu_rss_bytes": rss, "cpu_rss_measurement": status,
            "measured_controller_device": str(device),
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if cuda else None,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if cuda else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}


@contextmanager
def trace_range(name: str, device: Any = None, nvtx: bool = False):
    import torch
    cuda = getattr(device, "type", str(device).split(":")[0]) == "cuda"
    if cuda and nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if cuda and nvtx:
            torch.cuda.nvtx.range_pop()


def timed_call(function, device: Any = None, *, gpu_stage: bool = False, name: str = "stage", nvtx: bool = False):
    """Explicit-device event elapsed time plus synchronized host elapsed time."""
    import torch
    cuda = getattr(device, "type", str(device).split(":")[0]) == "cuda"
    synchronize(device)
    start, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if cuda and gpu_stage else (None, None)
    started = time.perf_counter()
    with trace_range(name, device, nvtx):
        if start is not None:
            with torch.cuda.device(device):
                start.record(torch.cuda.current_stream(device))
        value = function()
        if end is not None:
            with torch.cuda.device(device):
                end.record(torch.cuda.current_stream(device))
            end.synchronize()
        else:
            synchronize(device)
    return value, {"host_seconds": time.perf_counter() - started,
                   "cuda_event_seconds": start.elapsed_time(end) / 1000 if start is not None else None,
                   "timing_backend": "cuda_event_and_synchronized_host" if start is not None else "synchronized_host_cpu_fallback"}


def nvml_measurements(device: Any = None, enabled: bool = False) -> dict[str, Any]:
    if not enabled or getattr(device, "type", str(device).split(":")[0]) != "cuda":
        return {"nvml_status": "disabled" if not enabled else "unavailable_non_cuda", "nvml_utilization_percent": None,
                "nvml_device_memory_used_bytes": None}
    try:
        import pynvml
        import torch
        pynvml.nvmlInit()
        try:
            # Resolve by UUID: visible/logical CUDA indices are not physical NVML indices.
            properties = torch.cuda.get_device_properties(device)
            uuid = getattr(properties, "uuid", None)
            if uuid is None:
                return {"nvml_status": "unavailable_cuda_uuid", "nvml_utilization_percent": None,
                        "nvml_device_memory_used_bytes": None}
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            utilization, memory = pynvml.nvmlDeviceGetUtilizationRates(handle), pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {"nvml_status": "sampled", "nvml_utilization_percent": utilization.gpu,
                    "nvml_device_memory_used_bytes": memory.used,
                    "nvml_scope": "Single device-wide NVML sample; includes other processes and is not per-request utilization."}
        finally:
            pynvml.nvmlShutdown()
    except Exception as error:
        return {"nvml_status": "unavailable", "nvml_reason": f"{type(error).__name__}: {error}",
                "nvml_utilization_percent": None, "nvml_device_memory_used_bytes": None}
