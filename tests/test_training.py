from __future__ import annotations

import json
import copy
import os
import subprocess
import sys
import socket
import platform
from pathlib import Path

import pytest
import torch

from conductor.controller.factory import build_controller
from conductor.preference.dpo import dpo_loss
from conductor.schema import ExecutionState, RoutingDecision
from conductor.training.data import read_records, split_by_task
from conductor.training.runner import train


def config(path: Path, output: Path, stage: str = "sft", backend: str = "tiny") -> dict:
    return {"seed": 42, "model": {"backend": backend, "feature_dim": 128, "hidden_dim": 32,
                                  "num_experts": 4, "expert_top_k": 2, "max_agents": 2, "device": "cpu"},
            "routing": {"k": 2}, "training": {"stage": stage, "data": str(path), "output": str(output),
                                             "epochs": 16, "batch_size": 8, "validation_fraction": .2,
                                             "learning_rate": .01 if stage == "sft" else .003, "beta": .3}}


def records(stage: str) -> list[dict]:
    result = []
    for i in range(12):
        state = ExecutionState(f"Calculate {i}+2", "math")
        record = {"task_id": f"task-{i}", "split": "train", "state": state.to_dict()}
        if stage == "sft":
            record["decision"] = RoutingDecision(["math"]).to_dict()
        else:
            record.update(chosen=RoutingDecision(["math"]).to_dict(), rejected=RoutingDecision(["planner", "critic"]).to_dict())
        result.append(record)
    return result


def write_records(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def test_sft_updates_weights_and_dpo_uses_exact_frozen_sft_reference(tmp_path) -> None:
    sft_data, dpo_data = tmp_path / "sft.jsonl", tmp_path / "prefs.jsonl"
    write_records(sft_data, records("sft"))
    write_records(dpo_data, records("dpo"))
    sft_config = config(sft_data, tmp_path / "sft")
    before = build_controller(sft_config)
    initial = before.model.head.weight.detach().clone()
    metrics = train(sft_config)
    assert metrics["final_train"]["loss"] < metrics["initial_train"]["loss"]
    restored = build_controller(sft_config, str(tmp_path / "sft"))
    assert not torch.equal(initial, restored.model.head.weight)
    dpo_config = config(dpo_data, tmp_path / "dpo", "dpo")
    dpo_metrics = train(dpo_config, str(tmp_path / "sft"))
    assert dpo_metrics["initial_train"]["loss"] == pytest.approx(.693147, abs=1e-5)
    assert dpo_metrics["final_train"]["loss"] < dpo_metrics["initial_train"]["loss"]
    cache = json.loads((tmp_path / "dpo" / "reference_log_probabilities.json").read_text())
    assert cache["sft_checkpoint_sha256"] == dpo_metrics["reference_checkpoint_sha256"]
    assert not dpo_metrics["specialists_updated"]


def test_dpo_has_correct_reference_detachment_and_sign() -> None:
    chosen = torch.tensor([-.2], requires_grad=True)
    rejected = torch.tensor([-1.], requires_grad=True)
    reference = torch.tensor([-.3], requires_grad=True)
    loss, _ = dpo_loss(chosen, rejected, reference, torch.tensor([-1.]), beta=.1)
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0
    assert reference.grad is None


def test_task_disjoint_validation_and_empty_data_rejected(tmp_path) -> None:
    values = records("sft") * 2
    train_set, validation = split_by_task(values, .2, 42)
    assert {record["task_id"] for record in train_set}.isdisjoint({record["task_id"] for record in validation})
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_records(empty, "dpo")
    with pytest.raises(ValueError, match="SFT"):
        train(config(empty, tmp_path / "bad", "dpo"))


def test_evaluation_labels_rejected_and_k_sweep_counts_excluded(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    values = records("sft")
    values[0]["split"] = "eval"
    write_records(path, values)
    with pytest.raises(ValueError, match="held-out"):
        read_records(path, "sft")
    values[0]["split"] = "train"
    values[0]["decision"] = RoutingDecision(["planner", "math"]).to_dict()
    write_records(path, values)
    settings = config(path, tmp_path / "k1")
    settings["routing"]["k"] = 1
    settings["training"]["epochs"] = 1
    metrics = train(settings)
    assert metrics["excluded_actions_exceeding_k"] == 1


def test_hf_random_olmoe_local_sft_dpo_smoke(tmp_path) -> None:
    """Real HF/PEFT path; random tiny OLMoE, never pretrained-model evidence."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import OlmoeConfig, OlmoeModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    base = tmp_path / "random-olmoe"
    base.mkdir()
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "User": 3, "task": 4, ":": 5,
                  "Calculate": 6, "Task": 7, "type": 8, "math": 9, "Execution": 10, "state": 11,
                  "{": 12, "}": 13, '"': 14, "current_step": 15, "remaining_budget": 16,
                  "0": 17, "1": 18, "2": 19, "3": 20, "4": 21, "5": 22, "+": 23}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    wrapped.save_pretrained(base)
    architecture = OlmoeConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                              num_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
                              pad_token_id=0, eos_token_id=2)
    OlmoeModel(architecture).save_pretrained(base)
    sft_data = tmp_path / "sft.jsonl"
    dpo_data = tmp_path / "dpo.jsonl"
    write_records(sft_data, records("sft")[:4])
    write_records(dpo_data, records("dpo")[:4])
    settings = config(sft_data, tmp_path / "hf-sft", backend="hf")
    settings["model"].update(name=str(base), pretrained=False, max_length=64, dtype="float32", local_files_only=True,
                            lora={"enabled": True, "r": 2, "alpha": 4, "target_modules": ["q_proj", "v_proj", "gate"]})
    settings["training"].update(epochs=2, validation_fraction=0, gradient_checkpointing=True)
    controller = build_controller(settings)
    state = ExecutionState("Calculate 2+3", "math", previous_agent_outputs=[{"content": "large " * 500}])
    encoded = controller.encode_states([state])
    assert len(encoded["input_ids"][0]) <= 64
    assert 6 in encoded["input_ids"][0].tolist()  # Task header survives long execution history.
    decision = controller.batch_route([state, state], 2)
    assert all(len(value.selected_agents) <= 2 for value in decision)
    assert controller.last_batch_tokens == [64, 64]
    assert controller.expert_stats()["layers"]["0"]["observations"] == 128
    # Pretrained backbone parameters frozen, LoRA gate/attention parameters trainable.
    assert all(not parameter.requires_grad for name, parameter in controller.model.backbone.named_parameters() if "lora_" not in name)
    metrics = train(settings)
    assert metrics["pretrained"] is False
    restored = build_controller(settings, str(tmp_path / "hf-sft"))
    assert any(bool(parameter.abs().sum()) for name, parameter in restored.model.backbone.named_parameters() if "lora_B" in name)
    dpo_settings = {**settings, "training": {**settings["training"], "stage": "dpo", "data": str(dpo_data),
                                          "output": str(tmp_path / "hf-dpo"), "learning_rate": .001}}
    optimized = train(dpo_settings, str(tmp_path / "hf-sft"))
    assert optimized["reference_checkpoint_sha256"]
    assert optimized["initial_train"]["loss"] == pytest.approx(.693147, abs=1e-5)
    metadata = json.loads((tmp_path / "hf-dpo" / "controller.json").read_text())
    assert metadata["local_checkpoint_sha256"] and not metadata["pretrained"]
    optimized_controller = build_controller(dpo_settings, str(tmp_path / "hf-dpo"))
    exported = tmp_path / "merged"
    manifest = optimized_controller.export_merged(exported, [state])
    assert manifest["validation"]["validation_probe_count"] == 1
    assert (exported / "source_adapter" / "adapter_config.json").exists()
    reloaded = build_controller(settings, str(exported))
    torch.testing.assert_close(optimized_controller.forward_states([state]), reloaded.forward_states([state]), rtol=1e-4, atol=1e-5)
    assert not any(parameter.requires_grad for parameter in reloaded.model.parameters())
    optimized_controller._token_cache_size = 2
    optimized_controller.token_cache_clear()
    optimized_controller.encode_states([state, state])
    assert optimized_controller.token_cache_stats()["hits"] == 1
    assert optimized_controller.token_cache_stats()["entries"] == 1
    reference_logits = reloaded.forward_states([state])
    reloaded.compile_for_inference(backend="eager")
    torch.testing.assert_close(reference_logits, reloaded.forward_states([state]))
    assert reloaded.compile_status["validated"]
    training_controller = build_controller(settings, training=True)
    training_controller.config["training"]["router_auxiliary_weight"] = .02
    masks = torch.tensor([[1, 1, 0], [1, 1, 1]])
    ids = torch.tensor([[3, 4, 0], [3, 4, 3]])
    training_controller.forward_encoded({"input_ids": ids, "attention_mask": masks}, k=2)
    auxiliary = training_controller.last_auxiliary_loss
    changed = ids.clone()
    changed[0, 2] = 6
    training_controller.forward_encoded({"input_ids": changed, "attention_mask": masks}, k=2)
    torch.testing.assert_close(auxiliary, training_controller.last_auxiliary_loss)
    assert auxiliary.requires_grad and float(auxiliary.detach()) > 0


def test_exact_resume_matches_uninterrupted_with_partial_accumulation_window(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    write_records(path, records("sft")[:11])
    settings = config(path, tmp_path / "uninterrupted")
    settings["training"].update(epochs=3, batch_size=2, gradient_accumulation_steps=3, validation_fraction=0)
    train(settings)
    interrupted = copy.deepcopy(settings)
    interrupted["training"].update(output=str(tmp_path / "resumed"), stop_after_updates=1)
    partial = train(interrupted)
    assert not partial["completed"]
    interrupted["training"].pop("stop_after_updates")
    resumed = train(interrupted, resume=partial["resume_checkpoint"])
    assert resumed["completed"] and resumed["optimizer_windows"] == 6
    expected = build_controller(settings, str(tmp_path / "uninterrupted"))
    actual = build_controller(interrupted, str(tmp_path / "resumed"))
    for name, tensor in expected.model.state_dict().items():
        torch.testing.assert_close(tensor, actual.model.state_dict()[name], rtol=0, atol=0)
    incompatible = copy.deepcopy(interrupted)
    incompatible["training"]["learning_rate"] = .007
    with pytest.raises(ValueError, match="identity changed"):
        train(incompatible, resume=partial["resume_checkpoint"])


def test_accumulation_matches_actual_large_batch_including_partial_final_batch(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    write_records(path, records("sft")[:11])
    large = config(path, tmp_path / "large")
    large["training"].update(epochs=1, batch_size=6, validation_fraction=0)
    accumulated = copy.deepcopy(large)
    accumulated["training"].update(output=str(tmp_path / "accumulated"), batch_size=2, gradient_accumulation_steps=3)
    train(large)
    train(accumulated)
    expected = build_controller(large, str(tmp_path / "large"))
    actual = build_controller(accumulated, str(tmp_path / "accumulated"))
    for name, tensor in expected.model.state_dict().items():
        torch.testing.assert_close(tensor, actual.model.state_dict()[name], rtol=2e-5, atol=2e-7)


def test_two_process_cpu_gloo_matches_single_process_with_uneven_shards(tmp_path) -> None:
    import yaml
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch build has no Gloo")
    path = tmp_path / "sft.jsonl"
    write_records(path, records("sft")[:11])
    single = config(path, tmp_path / "single")
    single["training"].update(epochs=2, batch_size=4, validation_fraction=0)
    train(single)
    parallel = copy.deepcopy(single)
    parallel["training"].update(output=str(tmp_path / "parallel"), batch_size=2)
    source = tmp_path / "parallel.yaml"
    source.write_text(yaml.safe_dump(parallel))
    environment = {**os.environ, "CONDUCTOR_TORCH_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "GLOO_SOCKET_IFNAME": "lo0" if platform.system() == "Darwin" else "lo"}
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    process = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--master_addr=127.0.0.1",
                              f"--master_port={port}", "--nproc_per_node=2",
                              "--module", "conductor.train", "--config", str(source)], env=environment,
                             capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    metrics = json.loads((tmp_path / "parallel" / "metrics.json").read_text())
    assert metrics["world_size"] == 2 and metrics["train_examples"] == 11
    expected = build_controller(single, str(tmp_path / "single"))
    actual = build_controller(parallel, str(tmp_path / "parallel"))
    for name, tensor in expected.model.state_dict().items():
        torch.testing.assert_close(tensor, actual.model.state_dict()[name], rtol=3e-5, atol=3e-7)
