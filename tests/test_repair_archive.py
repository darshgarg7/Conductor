"""Archive contract regression fixtures; no language-model weights are loaded."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from conductor.datasets.tasks import make_tasks
from conductor.evaluate import evaluate
from conductor.utils.runs import write_json

RENDERER_PATH = Path(__file__).resolve().parents[1] / 'scripts/report_routing_repair.py'
spec = importlib.util.spec_from_file_location('reviewed_repair_renderer', RENDERER_PATH)
assert spec and spec.loader
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


@pytest.fixture
def comparison(tmp_path):
    tasks = [asdict(task) for task in make_tasks(314159, 36, 48)]
    data = tmp_path / 'tasks.jsonl'
    data.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
    directory = tmp_path / 'evaluation'
    config = {'data': str(data), 'output': str(directory), 'seed': 42, 'policies': ['rule_based'],
              'agents': {'backend': 'deterministic'}, 'k': 2, 'max_rounds': 3,
              'token_budget': 8192, 'agent_call_budget': 12,
              'routing_diagnostics': {'permutation_samples': 0}, 'bootstrap_samples': 100}
    asyncio.run(evaluate(config))
    return directory, {task['id']: task for task in tasks}


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


def change_raw(directory, change):
    path = directory / 'trajectories.jsonl'
    rows = renderer.read_jsonl(path)
    change(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_valid_raw_coverage_and_observed_aggregates(comparison):
    directory, inventory = comparison
    raw, summaries, conditions = renderer.checked_comparison(directory, {'rule_based'}, inventory)
    assert len(raw) == summaries[0]['task_count'] == 48
    assert summaries[0]['success_rate'] == 1.0
    assert summaries[0]['mean_agent_calls'] == pytest.approx(4 / 3)
    assert summaries[0]['mean_downstream_tokens'] > 0
    assert conditions['budgets']['agent_call_budget'] == 12


@pytest.mark.parametrize('mutation', ['empty_status', 'missing_raw', 'duplicate_raw', 'summary', 'exact', 'fingerprint', 'budget', 'unfrozen'])
def test_invalid_evidence_fails_closed(comparison, mutation):
    directory, inventory = comparison
    if mutation == 'empty_status':
        change_json(directory / 'metrics.json', lambda value: value.update(policy_status=[]))
    elif mutation == 'missing_raw':
        change_raw(directory, lambda rows: rows.pop())
    elif mutation == 'duplicate_raw':
        change_raw(directory, lambda rows: rows.__setitem__(-1, rows[0]))
    elif mutation == 'summary':
        change_json(directory / 'metrics.json', lambda value: value['policies'][0].update(mean_agent_calls=0))
    elif mutation == 'exact':
        change_raw(directory, lambda rows: rows[0].update(final_answer='incorrect'))
    elif mutation == 'fingerprint':
        change_raw(directory, lambda rows: rows[0]['metadata'].update(specialist_sha256='invalid'))
    elif mutation == 'budget':
        change_raw(directory, lambda rows: rows[0]['metadata']['evaluation_budgets'].update(agent_call_budget=999))
    elif mutation == 'unfrozen':
        change_json(directory / 'specialist_audit.json', lambda value: value['specialists'][0].update(frozen=False))
    with pytest.raises(ValueError):
        renderer.checked_comparison(directory, {'rule_based'}, inventory)


def test_shared_conditions_mismatch_is_rejected(comparison):
    directory, inventory = comparison
    _, _, conditions = renderer.checked_comparison(directory, {'rule_based'}, inventory)
    conditions['seed'] = 100
    with pytest.raises(ValueError, match='conditions'):
        renderer.checked_comparison(directory, {'rule_based'}, inventory, conditions)


def test_gzip_round_trip_checks_original_bytes_and_crc(tmp_path):
    path = tmp_path / 'trace.jsonl.gz'
    content = b'{"test":1}\n' * 4000
    path.write_bytes(gzip.compress(content, mtime=0))
    expected = hashlib.sha256(content).hexdigest()
    renderer.checked_gzip(path, expected, len(content))
    with pytest.raises(ValueError, match='round-trip'):
        renderer.checked_gzip(path, expected, len(content) - 1)
    with pytest.raises(ValueError, match='round-trip'):
        renderer.checked_gzip(path, '0' * 64, len(content))
    broken = bytearray(path.read_bytes())
    broken[-8] ^= 1
    path.write_bytes(broken)
    with pytest.raises(gzip.BadGzipFile):
        renderer.checked_gzip(path, expected, len(content))


def valid_expert_fixture():
    return {'layers': {'0': {'activation_counts': [2, 2, 2, 2], 'num_experts': 4,
                            'observations': 4, 'utilized_experts': 4,
                            'activation_frequency': [.25] * 4, 'routing_entropy': .9}},
            'task_type_activation_counts': {'math': {'0': [2, 0, 0, 2]},
                                            'code': {'0': [0, 2, 2, 0]}}}


def test_expert_counter_contract_does_not_substitute_for_device_validation():
    stats = valid_expert_fixture()
    architecture = {'num_hidden_layers': 1, 'num_local_experts': 4, 'num_experts_per_tok': 2}
    renderer.checked_expert_stats(stats, architecture)
    stats['layers']['0']['observations'] = 5
    with pytest.raises(ValueError, match='nonpadding'):
        renderer.checked_expert_stats(stats, architecture)
    stats = valid_expert_fixture()
    stats['task_type_activation_counts']['code']['0'][0] = 1
    with pytest.raises(ValueError, match='task-type'):
        renderer.checked_expert_stats(stats, architecture)


def test_saved_proof_artifact_contract_synthetic_not_a_tensor_or_cuda_validation(tmp_path):
    proof = {'nvidia_execution_verified': False, 'frozen_base_weight_audit_performed': False,
             'git_clean_at_verification': True,
             'verification_script_sha256': renderer.file_digest(Path('scripts/verify_pilot_adapters.py')),
             'stage_metrics': {}, 'stage_weight_summaries': {}, 'artifact_files_sha256': {},
             'sft_to_preference_weight_deltas': {'adapters': {'changed_tensor_count': 1},
                                               'head': {'changed_tensor_count': 1}},
             'changed_router_adapter_tensor_count': 1, 'exact_sft_reference_sha256': 'synthetic-reference'}
    for phase in ('sft', 'preference'):
        stage = tmp_path / phase
        checkpoint = stage / 'resume/final'
        checkpoint.mkdir(parents=True)
        artifact = checkpoint / 'head.pt'
        artifact.write_bytes(b'opaque synthetic bytes, never deserialized')
        metrics = {'phase': phase, 'reference_checkpoint_sha256': 'synthetic-reference' if phase == 'preference' else None}
        write_json(stage / 'metrics.json', metrics)
        write_json(stage / 'checkpoint_pointer.json', {'path': 'resume/final'})
        proof['stage_metrics'][phase] = metrics
        proof['stage_weight_summaries'][phase] = {'router_lora_B_tensor_count': 1,
            'nonzero_router_lora_B_tensor_count': 1, 'lora_B_tensor_count': 1, 'nonzero_lora_B_tensor_count': 1}
        proof['artifact_files_sha256'][phase] = {'head.pt': renderer.file_digest(artifact)}
    write_json(tmp_path / 'preference/reference_log_probabilities.json', {'sft_checkpoint_sha256': 'synthetic-reference'})
    (tmp_path / 'adapter-verification').mkdir()
    write_json(tmp_path / 'adapter-verification/post_training_verification.json', proof)
    assert renderer.checked_adapter_proof(tmp_path)['nvidia_execution_verified'] is False
    (tmp_path / 'sft/resume/final/head.pt').write_bytes(b'changed synthetic bytes')
    with pytest.raises(ValueError, match='artifact hash'):
        renderer.checked_adapter_proof(tmp_path)
