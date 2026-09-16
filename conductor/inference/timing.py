"""Platform-aware timing and explicitly separated memory measurements."""
from __future__ import annotations

import resource
import sys
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
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None}
