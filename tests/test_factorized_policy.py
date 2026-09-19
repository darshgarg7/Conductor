"""Joint-likelihood and public-mask semantics, with no model download/training."""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.nn import functional as F

from conductor.controller.factorized import FactorizedHead
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision


def state(calls=3, tokens=100, called=None):
    return ExecutionState("public request", "coordination_workflow",
                          remaining_budget={"agent_calls": calls, "tokens": tokens},
                          agents_already_called=called or [])


def teacher_path(head, feature, decision, k, allowed=None, calls=3, tokens=100):
    """Independent one-path reference, without catalog traversal/buffer tables."""
    decision = head.catalog.actions[head.catalog.index(decision)]
    allowed = list(range(len(head.agents))) if allowed is None else allowed
    capacity = min(k, math.floor(calls), len(allowed)) if tokens >= 1 else 0
    context = head.context(feature.unsqueeze(0))
    stop_logits = head.stop_head(context)[0]
    if capacity == 0:
        assert decision.terminate
        return stop_logits[0] * 0
    stop = stop_logits.log_softmax(-1)
    if decision.terminate:
        return stop[0]
    count = len(decision.selected_agents)
    assert count <= capacity
    count_logits = head.count_head(context)[0]
    count_logps = count_logits[:capacity].log_softmax(-1)
    mode = int(decision.execution_mode == "sequential")
    result = stop[1] + count_logps[count - 1]
    if count > 1:
        mode_context = context + head.count_embedding.weight[count - 1]
        result = result + head.mode_head(torch.tanh(mode_context))[0].log_softmax(-1)[mode]
    hidden = context + head.count_embedding.weight[count - 1] + head.mode_embedding.weight[mode]
    selected = []
    for position, name in enumerate(decision.selected_agents):
        index = head.agents.index(name)
        if mode:
            eligible = [i for i in allowed if i not in selected]
        else:
            previous = selected[-1] if selected else -1
            eligible = [i for i in allowed if i > previous and sum(j > i for j in allowed) >= count - position - 1]
        assert index in eligible
        logits = head.agent_head(torch.tanh(hidden))[0]
        result = result + logits[eligible].log_softmax(-1)[eligible.index(index)]
        selected.append(index)
        hidden = head.prefix_cell(head.agent_embedding.weight[index].unsqueeze(0), hidden)
    return result


@pytest.mark.parametrize("agents,max_agents", [(AGENT_NAMES, 3), (("math", "coder", "retriever"), 3)])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_exact_joint_normalization_full_and_small_catalogs_with_admission_holes(agents, max_agents, k):
    torch.manual_seed(19)
    head = FactorizedHead(6, max_agents, agents, hidden_dim=8)
    features = torch.randn(4, 6)
    mask = torch.ones(4, len(agents), dtype=torch.bool)
    mask[1, 1] = False
    mask[2, :] = False
    mask[2, -1] = True
    mask[3, :] = False
    logps = head.all_log_probabilities(features, [state(3), state(2), state(1), state(3)], k, admissible_mask=mask)
    torch.testing.assert_close(logps.exp().sum(-1), torch.ones(4), rtol=1e-6, atol=1e-6)
    assert not torch.isnan(logps).any()
    for row, available in zip(logps, mask):
        for action, value in zip(head.catalog.actions, row):
            if len(action.selected_agents) > k or any(not available[agents.index(agent)] for agent in action.selected_agents):
                assert value.isneginf()
    assert logps[3, 0] == 0 and logps[3, 1:].isneginf().all()
    assert head.out_features == (485 if agents == AGENT_NAMES else 20)


@pytest.mark.parametrize("budget", [state(0), state(.5), state(3, 0), state(3, .5)])
def test_forced_stop_has_zero_log_probability_and_zero_finite_gradient(budget):
    head = FactorizedHead(4, hidden_dim=8)
    features = torch.randn(1, 4, requires_grad=True)
    result = head.log_prob(features, [budget], [RoutingDecision([], terminate=True)], 3)
    assert result.item() == 0
    (-result.sum()).backward()
    assert torch.equal(features.grad, torch.zeros_like(features))
    assert all(parameter.grad is None or (torch.isfinite(parameter.grad).all() and not parameter.grad.count_nonzero())
               for parameter in head.parameters())


def test_parallel_canonical_order_and_remaining_feasibility_with_holes():
    agents = ("planner", "retriever", "researcher", "coder", "verifier")
    head = FactorizedHead(4, 3, agents, 8)
    features = torch.randn(1, 4)
    available = torch.tensor([[True, False, True, False, True]])
    action = RoutingDecision(["planner", "researcher", "verifier"], "parallel")
    reverse = RoutingDecision(list(reversed(action.selected_agents)), "parallel")
    joint = head.log_prob(features, [state()], [action], 3, admissible_mask=available)
    torch.testing.assert_close(joint, head.log_prob(features, [state()], [reverse], 3, admissible_mask=available))
    # With only three admissible agents, their count-three parallel subset is
    # forced after stop/count/mode. No agent-choice probability is lost to a
    # first choice that cannot complete the remaining picks.
    reference = teacher_path(head, features[0], action, 3, allowed=[0, 2, 4])
    torch.testing.assert_close(joint[0], reference)
    logps = head(features, [state()], 3, admissible_mask=available)
    torch.testing.assert_close(logps.exp().sum(-1), torch.ones(1))


def test_sequential_order_is_distinct_and_prior_round_calls_can_recur():
    head = FactorizedHead(5, hidden_dim=8)
    features = torch.randn(1, 5)
    previous = state(called=["math", "coder", "math"])
    a = RoutingDecision(["math", "coder", "retriever"], "sequential")
    b = RoutingDecision(["coder", "math", "retriever"], "sequential")
    values = [head.log_prob(features, [previous], [action], 3)[0] for action in (a, b)]
    assert all(value.isfinite() for value in values)
    assert not torch.isclose(values[0], values[1])
    torch.testing.assert_close(values[0], teacher_path(head, features[0], a, 3))
    torch.testing.assert_close(head.log_prob(features, [previous], [RoutingDecision(["math"])], 1),
                               head.log_prob(features, [state()], [RoutingDecision(["math"], "sequential")], 1))
    with pytest.raises(ValueError, match="duplicate"):
        head.log_prob(features, [previous], [RoutingDecision(["math", "math"], "sequential")], 3)


def test_teacher_forced_joint_and_gradients_match_vectorized_catalog_paths():
    torch.manual_seed(17)
    head = FactorizedHead(4, 3, ("math", "coder", "retriever"), 8)
    reference = copy.deepcopy(head)
    features = torch.randn(4, 4)
    actions = [RoutingDecision([], terminate=True), RoutingDecision(["math"]),
               RoutingDecision(["coder", "retriever"], "parallel"),
               RoutingDecision(["retriever", "math", "coder"], "sequential")]
    target = head.log_prob(features, None, actions, 3)
    independent = torch.stack([teacher_path(reference, feature, action, 3) for feature, action in zip(features, actions)])
    torch.testing.assert_close(target, independent, rtol=1e-6, atol=1e-6)
    (-target.mean()).backward()
    (-independent.mean()).backward()
    for (_, parameter), (_, expected) in zip(head.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None and expected.grad is not None
        torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-5, atol=1e-7)
    assert head.stop_head.weight.grad.count_nonzero()
    assert head.count_head.weight.grad.count_nonzero()
    assert head.mode_head.weight.grad.count_nonzero()
    assert head.prefix_cell.weight_hh.grad.count_nonzero()


def test_single_mode_has_no_redundant_probability_or_mode_gradient():
    head = FactorizedHead(4, hidden_dim=8)
    features = torch.randn(1, 4)
    result = head.log_prob(features, None, [RoutingDecision(["math"], "sequential")], 1)
    torch.testing.assert_close(result[0], teacher_path(head, features[0], RoutingDecision(["math"]), 1))
    (-result.sum()).backward()
    assert head.mode_head.weight.grad is not None
    assert head.mode_head.weight.grad.count_nonzero() == 0


def test_tensor_constraints_match_states_and_illegal_targets_fail_explicitly():
    head = FactorizedHead(4, hidden_dim=8)
    features = torch.randn(2, 4)
    states = [state(1), state(3, 0)]
    expected = head(features, states, 3)
    actual = head(features, k=3, call_budgets=torch.tensor([1., 3.]), token_budgets=torch.tensor([100., 0.]))
    torch.testing.assert_close(expected, actual)
    with pytest.raises(ValueError, match="illegal"):
        head.log_prob(features, states, [RoutingDecision(["math", "coder"]), RoutingDecision([], terminate=True)], 3)
    with pytest.raises(ValueError, match="admission"):
        head.log_prob(features[:1], None, [RoutingDecision(["math"])], 3,
                      admissible_mask=torch.zeros(1, len(AGENT_NAMES), dtype=torch.bool))
    for constraints in ({"call_budgets": torch.tensor([-1., 1.])}, {"token_budgets": torch.tensor([float("nan"), 0.])},
                        {"admissible_mask": torch.ones(2, len(AGENT_NAMES))}):
        with pytest.raises(ValueError):
            head(features, **constraints)
    with pytest.raises(ValueError, match="k"):
        head(features, k=4)


def test_greedy_whole_joint_argmax_and_sampling_only_produce_legal_actions():
    head = FactorizedHead(4, hidden_dim=8)
    features = torch.randn(4, 4)
    budgets = [state(3), state(2), state(1), state(0)]
    logps = head(features, budgets, 3)
    greedy = head.greedy(features, budgets, 3)
    assert [head.catalog.index(action) for action in greedy] == logps.argmax(-1).tolist()
    for _ in range(5):
        draws = head.draw(features, budgets, 3)
        for action, budget in zip(draws, budgets):
            action.validate(3)
            assert len(action.selected_agents) <= budget.remaining_budget["agent_calls"]
    assert greedy[-1].terminate and greedy[-1].confidence == 1


def test_dpo_equal_reference_uses_joint_likelihood_and_leaves_reference_frozen():
    head = FactorizedHead(4, hidden_dim=8)
    reference = copy.deepcopy(head).requires_grad_(False).eval()
    features = torch.randn(2, 4)
    chosen = [RoutingDecision(["math"]), RoutingDecision(["math", "coder"], "sequential")]
    rejected = [RoutingDecision([], terminate=True), RoutingDecision(["coder", "math"], "sequential")]
    actor_margin = head.log_prob(features, None, chosen, 3) - head.log_prob(features, None, rejected, 3)
    with torch.no_grad():
        reference_margin = reference.log_prob(features, None, chosen, 3) - reference.log_prob(features, None, rejected, 3)
    loss = -F.logsigmoid(.1 * (actor_margin - reference_margin)).mean()
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.count_nonzero() for parameter in head.parameters())
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in reference.parameters())


def test_head_identity_state_dict_and_shared_prefix_traversal_are_explicit():
    head = FactorizedHead(4, hidden_dim=8)
    restored = FactorizedHead(4, hidden_dim=8)
    restored.load_state_dict(head.state_dict())
    feature = torch.randn(2, 4)
    torch.testing.assert_close(head(feature, k=3), restored(feature, k=3))
    assert head.metadata() == restored.metadata()
    reversed_order = FactorizedHead(4, agents=tuple(reversed(AGENT_NAMES)), hidden_dim=8)
    assert head.metadata()["agent_order_sha256"] != reversed_order.metadata()["agent_order_sha256"]
    assert head.metadata()["mask_version"]
    assert head.metadata()["shared_prefix_nodes_per_depth"] == [5, 29, 77]
    calls = []
    hook = head.context.register_forward_hook(lambda module, inputs, outputs: calls.append(len(inputs[0])))
    head(feature, k=3)
    hook.remove()
    assert calls == [2]  # One shared feature projection, not 485 network passes.
