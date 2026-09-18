from __future__ import annotations

import pytest
import torch

from conductor.controller.actions import ActionCatalog
from conductor.schema import RoutingDecision
from conductor.training.probe import fit_head


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
