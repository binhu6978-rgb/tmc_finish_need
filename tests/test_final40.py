"""Fixed definitions, no extra performance runs or hyperparameter selection."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path

import pytest
import torch

from capability_final40 import ABLATIONS, Final40Hooks, source_weights, active_cells
from native_group_aggregation import NativeGroupHooks, update_norm
from models.Fed import Aggregation_AdaptiveFL, split_model, select_clients
from test_adapterfl_methods import args_for, tiny_models, assert_states, assert_rng
from adapterfl_methods import rng_state
from tools.final40_data import config, context_for
from tools.run_final40 import jobs, execute


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def setup(method, profiles=(0, 2, 3, 6), cids=(1, 2, 40, 70), lens=(2, 6, 10, 20)):
    models = tiny_models()
    args = args_for(method, num_classes=2)
    args._final40_context = dict(condition='A', population=[40, 30, 30], boundaries=[40, 70], data_mass=[20000, 15000, 15000])
    h = Final40Hooks(args, models)
    h.set_slots(zip(cids, profiles))
    anchor = {k: torch.full_like(v, .01) for k, v in models[-1].state_dict().items()}
    uploads = [{k: v.clone() + (i + 1) * .02 for k, v in split_model(anchor, models[p].state_dict()).items()} for i, p in enumerate(profiles)]
    return h, uploads, list(lens), anchor, models


@pytest.mark.parametrize('method', ABLATIONS + ('baseline', 'g03', 'p04', 'p05'))
def test_fixed_methods_first_round_native_uncovered_snapshots_and_rng(method):
    h, uploads, lens, anchor, _ = setup(method, profiles=(0, 2, 3, 4))
    original = copy.deepcopy((uploads, anchor))
    before = rng_state()
    native = Aggregation_AdaptiveFL(uploads, lens, anchor)
    assert_states(h.aggregate(uploads, lens, anchor), native)
    out = h.aggregate(uploads, lens, anchor)
    assert_rng(before, rng_state())
    for a, b in zip(uploads, original[0]):
        assert_states(a, b)
    assert_states(anchor, original[1])
    for region in h.regions:
        if not any(p in region['profiles'] for _, p in h.slots):
            from semcons_scope import slices
            for c in region['cells']:
                assert torch.equal(out[c['key']][slices(c)], anchor[c['key']][slices(c)])
    assert all(torch.isfinite(v).all() for v in out.values())


@pytest.mark.parametrize('method', ABLATIONS)
def test_one_tier_remains_exact_native(method):
    h, u, n, w, _ = setup(method, cids=(70, 71, 72, 73))
    h.aggregate(u, n, w)
    assert_states(h.aggregate(u, n, w), Aggregation_AdaptiveFL(u, n, w))


@pytest.mark.parametrize('method', ('g03', 'p04', 'p05'))
def test_existing_reference_aggregate_exactly_reused(method):
    h, u, n, w, models = setup(method)
    old = NativeGroupHooks(args_for(method), models)
    old.set_slots(h.slots)
    for _ in range(2):
        assert_states(h.aggregate(u, n, w), old.aggregate(u, n, w))


@pytest.mark.parametrize('method,virtual,native_direction', [
    ('a01', 'p04', True), ('a02', 'p04', False), ('a03', 'p05', True), ('a04', 'p05', False)])
def test_four_same_input_norm_controls(method, virtual, native_direction):
    h, u, n, w, _ = setup(method)
    v, _, _, _, _ = setup(virtual)
    h.aggregate(u, n, w); v.aggregate(u, n, w)
    out = h.aggregate(u, n, w)
    other = v.aggregate(u, n, w)
    native = Aggregation_AdaptiveFL(u, n, w)
    target = update_norm(other if native_direction else native, w)
    assert update_norm(out, w) == pytest.approx(target, rel=2e-6)
    assert h.round_diagnostics['matched_norm_relative_error'] < 2e-6


@pytest.mark.parametrize('method', [f'a{i:02}' for i in range(5, 11)])
def test_cell_scope_is_exactly_native_outside_selected_parameters(method):
    h, u, n, w, _ = setup(method)
    ref, _, _, _, _ = setup('p04')
    h.aggregate(u, n, w); ref.aggregate(u, n, w)
    out = h.aggregate(u, n, w); full = ref.aggregate(u, n, w)
    native = Aggregation_AdaptiveFL(u, n, w)
    from semcons_scope import slices
    for region in h.regions:
        allowed = active_cells(method, region, h.bn_keys)
        for c in region['cells']:
            expected = full if c in allowed else native
            assert torch.equal(out[c['key']][slices(c)], expected[c['key']][slices(c)])


def test_explicit_mass_definitions_and_physical_dedup():
    slots = [(1, 0), (1, 2), (2, 0), (40, 3), (70, 6), (71, 6)]
    lens = [2, 2, 6, 10, 20, 30]
    # Region covers first five, global A14/A15 also see high client71.
    members = [0, 1, 2, 3, 4]
    population, data = (40, 30, 30), (20000, 15000, 15000)
    expected = {'a11': [8, 10, 20], 'a12': [2, 1, 1], 'a13': [3, 1, 1],
                'a14': [10, 10, 50], 'a15': [8, 10, 50], 'a16': [10, 10, 20],
                'a17': [10, 10, 20], 'a19': [10, 10, 20], 'a20': [10, 10, 20]}
    for method, mass in expected.items():
        weights = source_weights(method, members, slots, lens, (40, 70), population, data)
        exponent = -.25 if method == 'a19' else -.75 if method == 'a20' else -.5
        prior = population if method == 'a16' else data if method == 'a17' else [1, 1, 1]
        target = [p * m ** exponent for p, m in zip(prior, mass)]
        observed = [sum(weights[i] for i in [0, 1, 2]), weights[3], weights[4]]
        assert [w / sum(weights) for w in observed] == pytest.approx([w / sum(target) for w in target])
        if method == 'a11':
            assert weights[0] == weights[1]
            assert (weights[0] + weights[1]) / weights[2] == pytest.approx(2/6)


def test_new_population_boundaries_and_prior_and_sampling():
    c = config()['conditions']['S4']
    ctx = context_for('S4', c, {i: list(range(i+10)) for i in range(100)})
    assert ctx['boundaries'] == [20, 50] and ctx['population'] == [20, 30, 50]
    slots = [(1, 0), (20, 3), (50, 6)]
    weights = source_weights('p05', [0, 1, 2], slots, [2, 9, 4], (20, 50), (20, 30, 50), (1, 1, 1))
    assert [w / sum(weights) for w in weights] == pytest.approx([.2, .3, .5])
    from utils.set_seed import set_random_seed
    args = args_for(client_hetero_ration='2:3:5')
    set_random_seed(1)
    ids = select_clients(args, [0, 3, 6] * 5, 7)
    assert all(cid >= [0, 20, 50][i % 3] for i, cid in enumerate(ids))


def test_exact_forty_unique_paths_devices_and_duplicate_protection(tmp_path):
    flat = [j for gpu in (0, 1) for batch in jobs(gpu, Path('python'), Path('data'), tmp_path) for j in batch]
    assert len(flat) == len({j['id'] for j in flat}) == len({j['output'] for j in flat}) == 40
    assert [j['id'] for j in flat[:2]] == ['A01', 'A02']
    assert sum(j['physical_gpu'] == 0 for j in flat) == 20
    (tmp_path / 'prepared').mkdir(); (tmp_path / 'prepared/manifest.json').write_text('{}')
    (tmp_path / 'status.json').write_text('{}')
    with pytest.raises(FileExistsError):
        execute(Path('python'), Path('data'), tmp_path)


def test_half_interpolation_preserves_native_fusion_and_reports_actual_shares():
    h, u, n, w, _ = setup('a18')
    p, _, _, _, _ = setup('p04')
    h.aggregate(u, n, w); p.aggregate(u, n, w)
    result, p04 = h.aggregate(u, n, w), p.aggregate(u, n, w)
    native = Aggregation_AdaptiveFL(u, n, w)
    for key in w:
        assert torch.allclose(result[key], .5 * native[key] + .5 * p04[key], atol=1e-7, rtol=1e-7)
    for r in h.round_diagnostics['regions']:
        if r['weighted_parameter_numel']:
            assert r['applied_slot_shares'] == pytest.approx([.5*a+.5*b for a,b in zip(r['native_slot_shares'], r['virtual_p04_slot_shares'])])


def test_queue_failure_finishes_peer_stops_later_batches_and_maps_only_one_gpu(tmp_path, monkeypatch):
    import tools.run_final40 as queue
    import tools.report_native_groups10 as validation
    launched = []
    class Child:
        def __init__(self, index):
            self.pid = 100 + index
            self.index = index
        def wait(self):
            return 1 if self.index == 0 else 0
    def launch(argv, **kwargs):
        launched.append((argv, kwargs))
        return Child(len(launched)-1)
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda argv, **kw: '24000' if any('memory.free' in s for s in argv) else 'GPU snapshot')
    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    monkeypatch.setattr(validation, 'validate_run', lambda path: None)
    assert queue.run_queue(1, Path('python'), Path('dataset'), tmp_path) == 1
    assert len(launched) == 2
    assert all(item[1]['env']['CUDA_VISIBLE_DEVICES'] == '1' for item in launched)
    state = json.loads((tmp_path / 'gpu1/status.json').read_text())
    assert state['failed'] == ['S1-Native'] and state['completed'] == ['S1-G03']
    assert state['state'] == 'failed'


def test_final40_uses_original_training_and_no_pt_two_rounds(tmp_path, monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    data = torch.utils.data.TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 0, 1]))
    monkeypatch.setattr(training, 'get_model_list', lambda args: (tiny_models(), [(.4, 2, i+1.) for i in range(7)]))
    monkeypatch.setattr(training, 'tqdm', lambda x: x)
    for method in ('a01', 'a11', 'a18', 'baseline', 'p05'):
        h, _, _, _, _ = setup(method)
        args = args_for(method, tmp_path / method, num_classes=2, epochs=2, eval_profiles='full', no_save_pt=True)
        args._final40_context = h.config['context']
        set_random_seed(1)
        with contextlib.redirect_stdout(io.StringIO()):
            training.AdaptiveFL(args, data, data, {i: [0, 1, 2, 3] for i in range(100)})
        path = next((tmp_path / method / method).iterdir())
        with (path / 'accuracy_each_round.csv').open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == 2 and set(k for k in records[0] if k.startswith('P')) == {'P6_accuracy', 'P6_ce'}
        assert json.loads((path / 'final_model_check.json').read_text())['finite_full_parameters']
        assert not list(path.glob('*.pt'))
