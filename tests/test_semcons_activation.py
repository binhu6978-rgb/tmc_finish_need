import copy
import json

import torch

from test_semcons_scope import setup
from test_adapterfl_methods import assert_states, assert_rng
from adapterfl_methods import rng_state
from semcons_scope import ScopeHooks
from tools.preflight_semcons_activation import observed_aggregate
from tools.report_semcons_activation import summarize_round, analyze, main as report_main


def test_real_input_counterfactual_observer_preserves_actual_result_and_rng(tmp_path):
    h, u, n, w, s = setup('s07')
    h.history = {f'{cid}:{r["id"]}': 2 for r in h.regions for cid in (1, 2, 3)}
    reference = copy.deepcopy(h); before = rng_state()
    expected = reference.aggregate(u, n, w, s)
    result = observed_aggregate(ScopeHooks.aggregate, tmp_path / 'extra.jsonl')(h, u, n, w, s)
    assert_states(result, expected)
    assert h.history == reference.history
    assert h.round_diagnostics == reference.round_diagnostics
    assert_rng(before, rng_state())
    extra = json.loads((tmp_path / 'extra.jsonl').read_text())
    pair = extra['same_input_s07_s09']
    assert pair['gates_identical'] and pair['history_identical']
    assert pair['aggregate_difference_l2'] > 0


def test_semantic_only_extra_direction_does_not_enter_decision(tmp_path):
    h, u, n, w, s = setup('s04')
    expected = copy.deepcopy(h).aggregate(u, n, w, s)
    result = observed_aggregate(ScopeHooks.aggregate, tmp_path / 'extra.jsonl')(h, u, n, w, s)
    assert_states(result, expected)
    assert all(not r.get('direction_risks') for r in h.round_diagnostics['regions'])
    extra = json.loads((tmp_path / 'extra.jsonl').read_text())
    assert any(extra['direction_observation_only'].values())
    row = summarize_round(h.round_diagnostics, h.slots, extra)
    assert row['direction']['n'] > 0
    assert row['joint_positive']['n'] > 0


def test_two_client_common_gate_cancellation_counted():
    h, u, n, w, s = setup('s06', profiles=(0, 0), cids=(1, 2),
                          deltas=[[1.], [-2.]], response_values=(1., -1.))
    h.aggregate(u, n, w, s)
    row = summarize_round(h.round_diagnostics, h.slots, {})
    assert row['gate_non_neutral_fraction'] == 1
    assert row['actual_changed_parameter_fraction_full'] == 0
    assert row['counts']['normalized_uniform_canceled_regions'] == 1
    assert row['two_distinct_clients']['active_gate_regions'] == 1
    assert row['joint_positive']['positive_fraction'] == 1


def test_zero_scores_negative_control_does_not_claim_changed_information():
    h, u, n, w, s = setup('s08', response_values=(1., 1., 1.))
    h.aggregate(u, n, w, s)
    row = summarize_round(h.round_diagnostics, h.slots, {})
    assert row['counts']['permuted_region_client_scores'] > 0
    assert row['counts']['changed_region_client_scores'] == 0
    assert row['semantic']['positive'] == 0


def test_report_end_to_end_and_incomplete_run_rejected(tmp_path, monkeypatch):
    import csv
    import sys
    import pytest
    for mode in ('s03', 's04', 's06', 's07', 's08', 's09'):
        h, u, n, w, s = setup(mode)
        h.aggregate(u, n, w, s)
        path = tmp_path / mode / 'activation_1'; path.mkdir(parents=True)
        # Deliberately synthetic journal fixture: never emitted as real preflight evidence.
        provenance = {'full_data': True, 'train_examples': 50000, 'test_examples': 10000, 'partition_sha256': 'fixture'}
        identity = {key: 'fixture' for key in ('source_sha256', 'seed', 'num_users', 'frac', 'local_ep', 'local_bs', 'lr', 'lr_decay', 'momentum', 'weight_decay')}
        identity.update(epochs=1, partition_sha256='fixture')
        files = {'preflight_provenance.json': provenance, 'manifest.json': {'identity': identity, 'initial_state_sha256': ['fixture']},
                 'preflight_resources.json': {'status': 'completed_short_preflight'}, 'completed.json': {}}
        for name, value in files.items():
            (path / name).write_text(json.dumps(value))
        (path / 'activation_observations.jsonl').write_text(json.dumps({'round': 1}) + '\n')
        with (path / 'method_diagnostics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'metrics_json']); writer.writeheader()
            writer.writerow({'round': 1, 'metrics_json': json.dumps({'server': h.round_diagnostics, 'slots': h.slots})})
    result = analyze(tmp_path)
    assert all(result['pairing'].values())
    monkeypatch.setattr(sys, 'argv', ['report', str(tmp_path)])
    report_main()
    assert len(list(csv.DictReader((tmp_path / 'activation_each_round.csv').open()))) == 6
    (tmp_path / 's09/activation_1/completed.json').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(tmp_path)
