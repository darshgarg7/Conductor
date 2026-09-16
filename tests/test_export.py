"""Merged export correctness with a small random local HF MoE, never model evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from conductor.controller.artifacts import atomic_directory
from conductor.controller.factory import build_controller
from conductor.schema import ExecutionState


def _file_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def saved_hf_controller(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from transformers import OlmoeConfig, OlmoeModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    base = tmp_path / "random-local-olmoe"
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "Calculate": 3, "math": 4,
                  "2": 5, "3": 6, "plus": 7, "User": 8, "task": 9}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                      eos_token="[EOS]", unk_token="[UNK]")
    wrapped.save_pretrained(base)
    architecture = OlmoeConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                              num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
                              pad_token_id=0, eos_token_id=2)
    OlmoeModel(architecture).save_pretrained(base)
    configuration = {"seed": 42, "model": {"backend": "hf", "name": str(base), "pretrained": False,
                                              "device": "cpu", "dtype": "float32", "max_agents": 2,
                                              "max_length": 64, "local_files_only": True,
                                              "lora": {"enabled": True, "r": 2, "alpha": 4,
                                                       "target_modules": ["q_proj", "v_proj", "gate"]}}}
    controller = build_controller(configuration)
    # Nonzero adapters exercise actual safe-merge math; this is test data only.
    with torch.no_grad():
        for name, parameter in controller.model.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.005)
    source = tmp_path / "immutable-adapter-checkpoint"
    with atomic_directory(source) as temporary:
        controller.save(temporary, "sft")
    return configuration, source


def test_in_place_export_preserves_saved_adapter_and_reload_equivalence(saved_hf_controller, tmp_path, monkeypatch):
    from conductor.controller import hf

    configuration, source = saved_hf_controller
    controller = build_controller(configuration, str(source))
    states = [ExecutionState("Calculate 2 plus 3", "math"),
              ExecutionState("Calculate 3 plus 2", "math", current_step=1)]
    with torch.inference_mode():
        before = controller.forward_states(states)
    before_actions = controller.decide(before, 2)
    source_hashes = _file_hashes(source)
    original_model = controller.model
    original_base = controller.model.backbone.get_base_model()
    original_deepcopy = hf.copy.deepcopy

    def no_module_copy(value, *args, **kwargs):
        if isinstance(value, torch.nn.Module):
            raise AssertionError("in-place export must never clone a full model")
        return original_deepcopy(value, *args, **kwargs)

    exported = tmp_path / "merged"
    with monkeypatch.context() as patch:
        patch.setattr(hf.copy, "deepcopy", no_module_copy)
        manifest = controller.export_merged(exported, states, preserve_model=False)
    assert controller.model is original_model
    assert controller.model.backbone is original_base
    assert not any("lora_" in name for name, _ in controller.model.named_parameters())
    assert manifest["validation"]["preserve_model"] is False
    assert manifest["validation"]["in_place_merge"] is True
    assert _file_hashes(source) == source_hashes
    assert _file_hashes(exported / "source_adapter") == _file_hashes(source / "adapter")
    reloaded = build_controller(configuration, str(exported))
    with torch.inference_mode():
        after = reloaded.forward_states(states)
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    assert [(action.selected_agents, action.execution_mode, action.terminate) for action in before_actions] == [
        (action.selected_agents, action.execution_mode, action.terminate) for action in reloaded.decide(after, 2)]
    with pytest.raises(ValueError, match="consumed"):
        controller.enable_training()
    with pytest.raises(ValueError, match="cannot save adapters"):
        controller.save(tmp_path / "invalid-adapter", "sft")


def test_in_place_export_rejects_unsaved_and_training_instances(saved_hf_controller, tmp_path):
    configuration, source = saved_hf_controller
    unsaved = build_controller(configuration)
    original_backbone = unsaved.model.backbone
    with pytest.raises(ValueError, match="source checkpoint with adapter files"):
        unsaved.export_merged(tmp_path / "unsaved-export", preserve_model=False)
    assert unsaved.model.backbone is original_backbone and not unsaved._in_place_merged
    training = build_controller(configuration, str(source), training=True)
    with pytest.raises(ValueError, match="eval mode"):
        training.export_merged(tmp_path / "training-export", preserve_model=False)
    assert not training._in_place_merged
    loaded = build_controller(configuration, str(source))
    with pytest.raises(ValueError, match="immutable source checkpoint"):
        loaded.export_merged(source / "nested-export", preserve_model=False)
    assert not loaded._in_place_merged


def test_failed_in_place_merge_blocks_inference_and_publication(saved_hf_controller, tmp_path, monkeypatch):
    configuration, source = saved_hf_controller
    controller = build_controller(configuration, str(source))
    source_hashes = _file_hashes(source)

    def merge_failure(**kwargs):
        raise RuntimeError("simulated safe merge failure")

    monkeypatch.setattr(controller.model.backbone, "merge_and_unload", merge_failure)
    exported = tmp_path / "failed-export"
    with pytest.raises(RuntimeError, match="safe merge failure"):
        controller.export_merged(exported, preserve_model=False)
    assert not exported.exists() and _file_hashes(source) == source_hashes
    with pytest.raises(RuntimeError, match="discard this instance"):
        controller.forward_states([ExecutionState("Calculate 2 plus 3", "math")])
    with pytest.raises(RuntimeError, match="discard this instance"):
        controller.compile_for_inference(backend="eager")


@pytest.mark.parametrize("in_place", [False, True])
def test_export_cli_forwards_preservation_option(tmp_path, monkeypatch, in_place):
    from conductor.controller import export

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "controller.json").write_text(json.dumps({"configuration": {"model": {"backend": "hf"}}}))
    calls = []

    class ExportController:
        def export_merged(self, path, *, preserve_model=True):
            calls.append((path, preserve_model))

    monkeypatch.setattr(export, "build_controller", lambda *args: ExportController())
    arguments = ["conductor.controller.export", "--checkpoint", str(source), "--output", "merged-output"]
    if in_place:
        arguments.append("--in-place")
    monkeypatch.setattr(sys, "argv", arguments)
    export.main()
    assert calls == [("merged-output", not in_place)]
