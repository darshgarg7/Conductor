import asyncio

import pytest
import torch

from conductor.agents import build_agents
from conductor.orchestration.runner import run_trajectory
from conductor.routing.policies import StaticSupervisorPolicy
from conductor.schema import ExecutionState, Task


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 1
    def __init__(self, output):
        self.output = output
    def __call__(self, *args, **kwargs):
        return Inputs(input_ids=torch.tensor([[1, 1]]), attention_mask=torch.tensor([[1, 1]]))
    def decode(self, *args, **kwargs):
        return self.output


class Model:
    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 1, 2]])


def supervisor(output):
    policy = StaticSupervisorPolicy.__new__(StaticSupervisorPolicy)
    policy.tokenizer, policy.model, policy.device, policy._torch = Tokenizer(output), Model(), torch.device('cpu'), torch
    policy.use_chat_template, policy.max_new_tokens, policy.max_context, policy.rate = False, 96, 2048, 1.0
    return policy


@pytest.mark.parametrize('text', ['no JSON', '{"selected_agents": ["unknown"]}', '{broken}',
    '{"selected_agents":["math","coder"],"execution_mode":"sequential","confidence":0.8,"terminate":false}'])
def test_invalid_supervisor_output_is_charged_and_fails_closed(text):
    policy = supervisor(text)
    decision = policy.route(ExecutionState('Compute 2+3', 'math'), 1)
    assert decision.terminate and decision.confidence == 0
    assert policy.last_invalid and policy.invalid_decisions == 1 and policy.last_error
    assert policy.last_tokens == 3 and policy.last_cost_usd == 3e-6
    task = Task('bad-supervisor', 'Compute 2+3', 'math', expected_answer='5')
    trajectory = asyncio.run(run_trajectory(task, policy, build_agents(), k=1))
    assert not trajectory.task_success and not trajectory.steps[0].agent_outputs
    assert trajectory.metadata['invalid_routing_decisions'][0]['step'] == 0


def test_valid_supervisor_json_has_no_fallback_flag():
    policy = supervisor('{"selected_agents":["math"],"execution_mode":"sequential","confidence":0.8,"terminate":false}')
    assert policy.route(ExecutionState('Compute 2+3','math'), 1).selected_agents == ['math']
    assert not policy.last_invalid
