from __future__ import annotations

import pytest
import torch

from conductor.controller.actions import ActionCatalog
from conductor.schema import ExecutionState, RoutingDecision
from conductor.training.probe import fit_factorized_head, fit_head


def test_linear_probe_uses_full_catalog_and_selects_validation_epoch() -> None:
    catalog = ActionCatalog(2)
    math = catalog.index(RoutingDecision(["math"]))
    coder = catalog.index(RoutingDecision(["coder"]))
    features = torch.tensor([[1., 0.], [0., 1.], [1.1, 0.], [0., 1.1]])
    labels = torch.tensor([math, coder, math, coder])
    weights, summary, trace = fit_head(features, labels, [0, 1], [2, 3], catalog, 2,
                                       epochs=100, learning_rate=.1, seed=42)
    assert weights["weight"].shape == (len(catalog), 2)
    assert summary["validation_accuracy"] == 1
    assert summary["validation_predictions"] == [math, coder]
    assert summary["validation_loss"] == min(row["validation_loss"] for row in trace if row["validation_accuracy"] == 1)


@pytest.mark.parametrize("train,validation", [([0, 1], [1, 2]), ([0], []), ([0], [1])])
def test_probe_rejects_leaking_empty_or_incomplete_partitions(train: list[int], validation: list[int]) -> None:
    with pytest.raises(ValueError, match="partitions"):
        fit_head(torch.ones(3, 2), torch.zeros(3, dtype=torch.long), train, validation,
                 ActionCatalog(1), 1, epochs=1, learning_rate=.01, seed=42)


def test_factorized_probe_fits_complete_actions_and_records_head_semantics() -> None:
    catalog = ActionCatalog(2)
    retrieve_math = catalog.index(RoutingDecision(["retriever", "math"], "sequential"))
    stop = catalog.index(RoutingDecision([], terminate=True))
    features = torch.tensor([[1., 0.], [0., 1.], [1.1, 0.], [0., 1.1]])
    states = [ExecutionState("request", "coordination_workflow") for _ in range(4)]
    labels = torch.tensor([retrieve_math, stop, retrieve_math, stop])
    weights, summary, trace = fit_factorized_head(
        features, states, labels, [0, 1], [2, 3], catalog, 2,
        epochs=100, learning_rate=.05, seed=42, hidden_dim=8)
    assert weights and trace
    assert summary["validation_accuracy"] == 1
    assert summary["head_metadata"]["factorization"].startswith("stop -> count")
