"""CUDA execution checks: unavailable hardware is skipped, never emulated."""
from __future__ import annotations

import json

import pytest
import torch

from conductor.controller.factory import build_controller
from conductor.schema import ExecutionState, RoutingDecision
from conductor.training.runner import train

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual NVIDIA CUDA hardware")]


def _records(path) -> None:
    values = [{"task_id": f"task-{index}", "split": "train", "state": ExecutionState(f"Calculate {index}+2", "math").to_dict(),
               "decision": RoutingDecision(["math"]).to_dict()} for index in range(7)]
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_cuda_fp16_accumulation_scaler_and_atomic_resume(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    _records(path)
    configuration = {"seed": 42, "model": {"backend": "tiny", "device": "cuda:0", "require_cuda": True,
                                           "dtype": "float16", "max_agents": 2, "num_experts": 4, "expert_top_k": 2},
                     "routing": {"k": 2}, "training": {"stage": "sft", "data": str(path), "output": str(tmp_path / "trained"),
                                                       "epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 2,
                                                       "validation_fraction": 0, "learning_rate": .003}}
    metrics = train(configuration)
    torch.cuda.synchronize()
    checkpoint = torch.load(metrics["resume_checkpoint"] + "/training_state.pt", weights_only=False, map_location="cpu")
    assert "scale" in checkpoint["scaler"]
    controller = build_controller(configuration, metrics["checkpoint"])
    assert controller.device.type == "cuda" and controller.dtype == torch.float16
    assert bool(torch.isfinite(controller.forward_states([ExecutionState("Calculate 2+3", "math")])).all())


def test_cuda_hf_fp16_lora_checkpointing_preserves_fp32_trainable_parameters(tmp_path) -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import OlmoeConfig, OlmoeModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    base = tmp_path / "random-olmoe"
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "Calculate": 3, "math": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    wrapped.save_pretrained(base)
    architecture = OlmoeConfig(vocab_size=5, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=4, num_experts=4, num_experts_per_tok=2,
                              max_position_embeddings=64, pad_token_id=0, eos_token_id=2)
    OlmoeModel(architecture).save_pretrained(base)
    path = tmp_path / "sft.jsonl"
    _records(path)
    configuration = {"seed": 42, "model": {"backend": "hf", "name": str(base), "pretrained": False,
                                           "device": "cuda:0", "require_cuda": True, "dtype": "float16", "max_agents": 2,
                                           "max_length": 64, "attention_implementation": "sdpa", "local_files_only": True,
                                           "lora": {"enabled": True, "r": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj", "gate"]}},
                     "routing": {"k": 2}, "training": {"stage": "sft", "data": str(path), "output": str(tmp_path / "trained"),
                                                       "epochs": 2, "batch_size": 2, "gradient_accumulation_steps": 2,
                                                       "gradient_checkpointing": True, "validation_fraction": 0, "learning_rate": .001}}
    controller = build_controller(configuration, training=True)
    assert all(parameter.dtype == torch.float32 for parameter in controller.model.parameters() if parameter.requires_grad)
    assert all(not parameter.requires_grad for name, parameter in controller.model.backbone.named_parameters() if "lora_" not in name)
    metrics = train(configuration)
    torch.cuda.synchronize()
    restored = build_controller(configuration, metrics["checkpoint"])
    assert any(bool(parameter.abs().sum()) for name, parameter in restored.model.backbone.named_parameters() if "lora_B" in name)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in restored.model.parameters())
