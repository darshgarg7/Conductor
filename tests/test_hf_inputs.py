"""HF input/pooling semantics without downloading or loading model weights."""
from __future__ import annotations

import copy
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from conductor.controller.hf import HFController, HFRoutingModel
from conductor.schema import ExecutionState


class CharacterTokenizer:
    """Lossless test tokenizer with explicit special-token budget accounting."""
    def num_special_tokens_to_add(self, pair=False):
        return 1

    def encode(self, value, add_special_tokens=False):
        return [ord(char) + 2 for char in value]

    def build_inputs_with_special_tokens(self, values):
        return [*values, 1]

    def pad(self, values, padding=True, return_tensors="pt"):
        rows = values["input_ids"]
        maximum = max(map(len, rows))
        return {"input_ids": torch.tensor([row + [0] * (maximum - len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1] * len(row) + [0] * (maximum - len(row)) for row in rows])}

    def decode(self, values):
        return "".join(chr(int(value) - 2) for value in values if value > 1)


def controller(mode="priority_v1", length=384, cache_size=4):
    result = HFController.__new__(HFController)
    result.state_serialization = mode
    result.max_length = length
    result.tokenizer = CharacterTokenizer()
    result._token_cache = OrderedDict()
    result._token_cache_details = OrderedDict()
    result._token_cache_size = cache_size
    result._token_cache_hits = result._token_cache_misses = 0
    return result


def test_priority_preserves_progress_and_short_task_despite_large_history():
    instance = controller()
    state = ExecutionState("Compute 2+3.", "math", current_step=7,
                           agents_already_called=["math", "math"], remaining_budget={"tokens": 123, "agent_calls": 4},
                           previous_routing_decisions=[{"selected_agents": ["math"], "execution_mode": "parallel",
                                                        "terminate": False, "confidence": .91}],
                           previous_agent_outputs=[{"agent": "math", "content": "x" * 3000,
                                                    "metadata": {"answer": "5", "expected_answer": "SECRET"},
                                                    "latency_seconds": 9.8765, "cost_usd": 1.2345, "tokens": 800}])
    serialized = instance.serialize(state)
    assert "SECRET" not in serialized and "latency_seconds" not in serialized and "cost_usd" not in serialized
    encoded = instance.tokenize_states([state])
    decoded = instance.tokenizer.decode(encoded["input_ids"][0])
    for required in ('"step":7', '"called":["math"]', '"tokens":123', '"agent_calls":4',
                     '"recent_routes":', '"answer_present":true', 'Task type: math', 'Compute 2+3.'):
        assert required in decoded
    assert encoded["attention_mask"].sum() <= instance.max_length
    details = instance.last_tokenization_details[0]
    assert details["priority_fields"]["step"] == 7
    assert not details["task_truncated"] and details["history_truncated"]


def test_priority_is_invariant_to_measured_accounting_and_never_promises_full_huge_task():
    instance = controller()
    state = ExecutionState("public task " * 500, "retrieval", previous_agent_outputs=[{
        "agent": "retriever", "content": "public evidence", "tokens": 20, "cost_usd": 1,
        "latency_seconds": .1, "metadata": {"answer": None, "latency": .2, "expected_answer": "SECRET"}}])
    changed = copy.deepcopy(state)
    changed.previous_agent_outputs[0].update(tokens=200, cost_usd=9, latency_seconds=77)
    changed.previous_agent_outputs[0]["metadata"].update(latency=99, expected_answer="ANOTHER SECRET")
    assert instance.serialize(state) == instance.serialize(changed)
    inputs = instance.tokenize_states([state])
    assert instance.last_tokenization_details[0]["task_truncated"]
    assert inputs["input_ids"].shape[1] <= instance.max_length
    assert instance.last_tokenization_details[0]["priority_fields"]["step"] == 0
    # Cache hits retain a fresh audit record rather than a mutable shared alias.
    instance.last_tokenization_details[0]["priority_fields"]["step"] = 99
    instance.tokenize_states([state])
    assert instance.last_tokenization_details[0]["priority_fields"]["step"] == 0
    assert instance.token_cache_stats()["hits"] == 1


def test_priority_fails_explicitly_if_mandatory_fields_cannot_fit():
    instance = controller(length=16)
    with pytest.raises(ValueError, match="progress/type fields do not fit"):
        instance.tokenize_states([ExecutionState("task", "math")])


def test_legacy_class_and_instance_serialization_and_tokenization_are_unchanged():
    instance = controller("legacy", length=128)
    state = ExecutionState("task" * 30, "retrieval", previous_agent_outputs=[{"content": "output" * 100}])
    assert instance.serialize(state) == HFController.serialize(state)
    assert instance.serialize(state) == HFController.serialize(state=state)
    head, tail = instance.serialize(state).split("\0CONDUCTOR_STATE\0")
    budget = 127
    header = instance.tokenizer.encode(head)[:budget // 3]
    expected = instance.tokenizer.build_inputs_with_special_tokens(header + instance.tokenizer.encode(tail)[-(budget - len(header)):])
    actual = instance.tokenize_states([state])["input_ids"][0].tolist()
    assert actual == expected


@pytest.mark.parametrize("pooling", ["last", "mean"])
@pytest.mark.parametrize("normalization", ["none", "layer_norm"])
def test_pooling_masks_both_padding_sides_and_matches_forward(pooling, normalization):
    hidden = torch.tensor([[[1., 3.], [3., 7.], [999., 999.]],
                           [[999., 999.], [5., 9.], [7., 15.]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    class FixedBackbone(nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(last_hidden_state=hidden, router_logits="actual-router-output")

    model = HFRoutingModel(FixedBackbone(), 2, 2, pooling, normalization)
    with torch.no_grad():
        model.head.weight.copy_(torch.eye(2))
        model.head.bias.zero_()
    expected = torch.tensor([[3., 7.], [7., 15.]]) if pooling == "last" else torch.tensor([[2., 5.], [6., 12.]])
    if normalization == "layer_norm":
        expected = F.layer_norm(expected, (2,))
    pooled = model.pooled_hidden(hidden, mask)
    torch.testing.assert_close(pooled, expected)
    logits, router = model({"input_ids": mask, "attention_mask": mask})
    torch.testing.assert_close(logits, expected)
    assert router == "actual-router-output"
    assert set(model.state_dict()) == {"head.weight", "head.bias"}
    pooled[:, 0].sum().backward()
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad[mask.bool()]) > 0
    assert torch.equal(hidden.grad[mask == 0], torch.zeros_like(hidden.grad[mask == 0]))


def test_pooling_rejects_all_padding_and_invalid_options():
    model = HFRoutingModel(nn.Identity(), 2, 2, pooling="mean")
    with pytest.raises(ValueError, match="unmasked token"):
        model.pooled_hidden(torch.zeros(1, 2, 2), torch.zeros(1, 2))
    for options in ({"pooling": "max"}, {"head_input_normalization": "batch_norm"}):
        with pytest.raises(ValueError):
            HFRoutingModel(nn.Identity(), 2, 2, **options)


def test_checkpoint_context_preservation_is_opt_in_and_does_not_block_device_override(tmp_path, monkeypatch):
    """Exercise saved metadata loading; model allocation is irrelevant to this policy."""
    def capture_initialization(self, config, checkpoint=None):
        self.config = config
        self.source_checkpoint = str(checkpoint)

    monkeypatch.setattr(HFController, "__init__", capture_initialization)
    evaluation = {"model": {"max_length": 128, "device": "cpu"},
                  "inference": {"preserve_checkpoint_context": True}}
    for length in (128, 256):
        checkpoint = tmp_path / str(length)
        checkpoint.mkdir()
        (checkpoint / "controller.json").write_text(json.dumps({"configuration": {
            "model": {"max_length": length, "device": "cuda:0"}, "inference": {}}}))
        preserved = HFController.load(checkpoint, evaluation)
        assert preserved.config["model"]["max_length"] == length
        assert preserved.config["model"]["device"] == "cpu"
        ordinary = HFController.load(checkpoint, {"model": {"max_length": 128, "device": "cpu"}})
        assert ordinary.config["model"]["max_length"] == 128
        assert ordinary.config["model"]["device"] == "cpu"
