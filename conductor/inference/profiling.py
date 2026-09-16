"""Separate host preparation, device transfer, forward and decision measurements."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from conductor.inference.timing import memory_measurements, timed_call
from conductor.schema import ExecutionState


def profile_stages(controller: Any, states: list[ExecutionState], k: int, *, repeats: int = 3,
                   nvtx: bool = False, trace_path: str | Path | None = None) -> dict[str, Any]:
    required = ("serialize", "tokenize_serialized", "move_inputs", "forward_encoded", "decide")
    if not hasattr(controller, "tokenizer") or not all(callable(getattr(controller, key, None)) for key in required):
        return {"status": "unsupported", "reason": "Backend does not expose separated HF preparation/forward stages", "samples": []}
    if repeats < 1 or not states:
        raise ValueError("Stage profiling requires positive repeats and at least one state")
    device = controller.device
    controller.model.eval()
    controller.batch_route(states, k)  # warmup is outside every timing sample
    samples: list[dict[str, Any]] = []

    def measure(repeat: int) -> None:
        texts, serialization = timed_call(lambda: [controller.serialize(state) for state in states],
                                         name="conductor.serialize", nvtx=nvtx)
        encoded, tokenization = timed_call(lambda: controller.tokenize_serialized(texts),
                                          name="conductor.tokenize", nvtx=nvtx)
        inputs, transfer = timed_call(lambda: controller.move_inputs(encoded), device, gpu_stage=True,
                                     name="conductor.h2d", nvtx=nvtx)
        with torch.inference_mode():
            logits, forward = timed_call(lambda: controller.forward_encoded(inputs, k, track=False), device,
                                        gpu_stage=True, name="conductor.forward", nvtx=nvtx)
            decisions, decision = timed_call(lambda: controller.decide(logits, k), device,
                                            name="conductor.decide", nvtx=nvtx)
        samples.append({"repeat": repeat, "batch_size": len(states), "input_tokens": int(encoded["attention_mask"].sum()),
                        "serialization": serialization, "tokenization": tokenization, "transfer": transfer,
                        "forward": forward, "decision": decision,
                        "decisions": [item.to_dict() for item in decisions], **memory_measurements(device)})

    for repeat in range(repeats):
        measure(repeat)
    trace = {"status": "disabled"}
    if trace_path is not None:
        target = Path(trace_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            measure(-1)
        instrumented_sample = samples.pop()
        profiler.export_chrome_trace(str(target))
        trace = {"status": "measured", "path": str(target), "activities": [str(item) for item in activities],
                 "instrumented_sample": instrumented_sample}
    return {"status": "measured", "samples": samples, "trace": trace,
            "token_cache": controller.token_cache_stats() if hasattr(controller, "token_cache_stats") else None,
            "scope": "Separately synchronized warm stages. Stage sums are diagnostic; they do not equal asynchronous service latency. Instrumentation is excluded from request throughput measurements."}
