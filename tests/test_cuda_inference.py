"""Opt-in actual device regressions, never mocked into CUDA evidence."""
import asyncio

import pytest
import torch

from conductor.controller.factory import build_controller
from conductor.inference.profiling import profile_stages
from conductor.inference.timing import memory_measurements, timed_call
from conductor.schema import ExecutionState
from conductor.serving.engine import RoutingEngine

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual NVIDIA CUDA required")]


def test_cuda_events_memory_and_bounded_batched_routing():
    controller = build_controller({"seed": 3, "model": {"backend": "tiny", "device": "cuda:0", "dtype": "bfloat16",
                    "feature_dim": 32, "hidden_dim": 16, "num_experts": 4, "expert_top_k": 2, "max_agents": 3}})
    states = [ExecutionState(str(index), "math") for index in range(4)]
    routed, sample = timed_call(lambda: controller.batch_route(states, 2), controller.device, gpu_stage=True)
    assert len(routed) == 4 and all(len(item.selected_agents) <= 2 for item in routed)
    assert sample["cuda_event_seconds"] is not None and sample["cuda_event_seconds"] > 0
    memory = memory_measurements(controller.device)
    assert memory["cuda_allocated_bytes"] > 0 and memory["cuda_reserved_bytes"] >= memory["cuda_allocated_bytes"]
    async def scenario():
        engine = RoutingEngine(controller, max_batch_size=4, max_pending=8, batch_wait_seconds=.01)
        await engine.start()
        try:
            results = await asyncio.gather(*(engine.submit(state, 2) for state in states))
            assert len(results) == 4
            assert all(item["decision"]["selected_agents"] == routed[index].selected_agents
                       for index, item in enumerate(results))
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_hf_cuda_batched_stage_events(tmp_path):
    from transformers import GraniteMoeConfig, GraniteMoeModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    path = tmp_path / "model"
    model = GraniteMoeModel(GraniteMoeConfig(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2))
    model.save_pretrained(path)
    tokens = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "math": 2}, unk_token="[UNK]"))
    tokens.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokens, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)
    controller = build_controller({"model": {"backend": "hf", "pretrained": False, "name": str(path), "device": "cuda:0", "dtype": "bfloat16",
                    "require_cuda": True, "max_length": 32, "max_agents": 3, "lora": {"enabled": False}},
                    "inference": {"instrument_experts": False}})
    measured = profile_stages(controller, [ExecutionState("2+3", "math"), ExecutionState("1+1", "math")], 2)
    assert measured["status"] == "measured"
    assert all(sample["forward"]["cuda_event_seconds"] > 0 for sample in measured["samples"])
