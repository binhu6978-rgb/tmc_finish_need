"""Focused observational contracts; real-data regression is a separate GPU run."""
import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from semcons_a07.config import load
from semcons_a07.adapter import Instrument, merge_candidates
from semcons_a07.measurement import Measurement, head_digest, probe_indices
from semcons_a07.semantic_memory import SemanticMemory
from semcons_a07.reliability_memory import ReliabilityMemory
from semcons_a07.prototypes import summarize, scores
from semcons_a07.diagnostics import rng_digest, model_digest, utility_records
from semcons_a07.aggregation import neutral_a07
from test_final40 import setup
from test_adapterfl_methods import assert_states


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4, track_running_stats=False)
        self.classifier = nn.Linear(4, 3)

    def forward(self, x):
        h = self.bn(self.conv(x)).relu()
        return dict(representation=h, output=self.classifier(h.mean((2, 3))))


class TrainData:
    def __init__(self):
        self.targets = [0, 1, 2] * 4
        self.images = torch.randn(12, 3, 4, 4)

    def __getitem__(self, i):
        return self.images[i].clone(), self.targets[i]


def proto(value):
    v = torch.tensor(value, dtype=torch.float64)
    return dict(count=2, embedding=v / v.norm(), probability=v / v.sum())


def test_adapter_detaches_backbone_and_head_has_no_gradient():
    cfg = load(); x = torch.randn(6, 4, requires_grad=True)
    instrument = Instrument(4, 3, 128, 17, 1, 0)
    digest = head_digest(instrument); before = rng_digest()
    state, count = instrument.candidate(x, torch.tensor([0, 1, 2]*2), cfg)
    assert count == 6 and x.grad is None
    assert all(p.grad is None and not p.requires_grad for p in instrument.head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in instrument.decoder.parameters())
    assert any(p.grad is not None for p in instrument.adapter.parameters())
    assert head_digest(instrument) == digest and before == rng_digest()
    assert state.keys() == instrument.adapter.state_dict().keys()


def test_frozen_heads_shared_across_profiles_seeds_not_training_seeds():
    before = rng_digest()
    a = Instrument(4, 3, 128, 17, 1, 0)
    b = Instrument(8, 3, 128, 17, 80, 6)
    c = Instrument(4, 3, 128, 29, 1, 0)
    assert head_digest(a) == head_digest(b) != head_digest(c)
    assert before == rng_digest()


def test_probe_is_training_only_fixed_no_missing_class_fabrication():
    dataset = TrainData()
    plan = probe_indices(dataset, list(range(12)), 1, 2)
    assert plan == probe_indices(dataset, list(reversed(range(12))), 1, 2)
    assert len(plan) == 6
    assert probe_indices(dataset, [0, 1, 2], 1, 2) == []
    values = summarize(None, torch.tensor([[.2, .3, .5], [.1, .4, .5]]), torch.tensor([2, 2]), 2)
    assert list(values) == [2] and values[2]['probability'].sum() == 1


def test_prepost_snapshot_repeat_slot_isolation_and_state_rng_gradients():
    cfg = load(); dataset = TrainData(); net = TinyNet()
    measurement = Measurement(cfg, dataset, {1: list(range(12))}, 3)
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    grads = [p.grad.clone() for p in net.parameters()]
    digest = model_digest(net); rng = rng_digest()
    before = measurement.before(net, 1)
    h = dict(client=dict(support=0), client_profile=dict(support=0))
    first = measurement.after(net, 1, 0, 1, before, {}, h)
    second = measurement.after(net, 1, 0, 1, before, {}, h)
    assert model_digest(net) == digest and rng_digest() == rng and net.training
    assert all(torch.equal(p.grad, g) for p, g in zip(net.parameters(), grads))
    assert measurement.bank == {}
    for seed in cfg['head_seeds']:
        a, b = first['heads'][seed], second['heads'][seed]
        assert not a['adapter_snapshot_prior'] and a['evidence']['S0'] is None
        for y in a['before']:
            assert torch.equal(a['before'][y]['embedding'], a['after'][y]['embedding'])
            assert torch.equal(a['after'][y]['probability'], b['after'][y]['probability'])
    measurement.commit()
    assert len(measurement.bank) == 3 and not measurement.pending


def test_semantic_memory_round1_and_duplicate_physical_votes_snapshot_and_age():
    cfg = load(); memory = SemanticMemory(cfg)
    records = [(1, 17, {0: proto([.8, .2])}), (1, 17, {0: proto([.2, .8])})]
    memory.commit(1, records)
    memory.commit(2, records)
    assert memory.anchors == {}  # two slots are one person
    snapshot = memory.snapshot()
    memory.commit(3, records + [(2, 17, {0: proto([.4, .6])})])
    assert snapshot == {} and memory.anchors[17][0]['support'] == 2
    current = {0: proto([.7, .3])}
    result = scores(current, current, {}, {}, memory.anchors[17], {}, 4, 100)
    assert result['S1'] == 0 and result['S0'] is not None
    assert scores(current, current, {}, {}, memory.anchors[17], {}, 104, 100)['S0'] is None
    assert scores(current, current, {}, {}, memory.anchors[17], {}, 3, 100)['S0'] is None


def test_reliability_past_only_per_client_profile_and_neutral_coldstart():
    memory = ReliabilityMemory(load()); snap = memory.snapshot()
    memory.commit(1, [(1, 0, .4)])
    memory.commit(2, [(1, 0, .2), (1, 0, .4), (1, 2, .6), (2, 0, None)])
    assert memory.read(snap, 1, 0, 3)['client']['support'] == 0
    read = memory.read(memory.snapshot(), 1, 0, 3)
    assert read['client']['support'] == 1 and read['client_profile']['support'] == 1
    assert not read['client']['ready']
    assert memory.read(memory.snapshot(), 1, 6, 3)['client_profile']['support'] == 0
    memory.commit(3, [(1, 0, .1)])
    assert memory.read(memory.snapshot(), 1, 0, 4)['client']['ready']
    assert not memory.read(memory.snapshot(), 1, 0, 104)['client']['ready']


def test_candidate_merge_one_commit_per_identity():
    assert torch.equal(merge_candidates([{'w': torch.tensor([1.])}, {'w': torch.tensor([3.])}])['w'], torch.tensor([2.]))


@pytest.mark.parametrize('profiles,cids', [((0, 2, 3, 6), (1, 1, 40, 70)), ((0, 2, 3, 4), (70, 71, 72, 73))])
def test_neutral_dispatcher_exact_frozen_a07_including_first_round(profiles, cids):
    a, uploads, lens, old, _ = setup('a07', profiles=profiles, cids=cids, lens=(2, 2, 10, 20))
    b = copy.deepcopy(a)
    for _ in range(2):
        assert_states(neutral_a07(a, uploads, lens, old, gamma=[1]*4), b.aggregate(uploads, lens, old))
    with pytest.raises(ValueError, match='No scientific Go'):
        neutral_a07(a, uploads, lens, old, gamma=[.9]*4)


def test_receiver_injection_fixed_bn_pair_identity_and_no_model_pollution():
    cfg = load(); cfg['senders_per_region'] = 2; cfg['receivers_per_sender_region'] = 1
    net = TinyNet(); models = [net, copy.deepcopy(net)]
    dataset = TrainData(); measurement = Measurement(cfg, dataset, {1: list(range(12)), 2: list(range(12))}, 3)
    original = copy.deepcopy(net.state_dict()); uploads = [copy.deepcopy(original), copy.deepcopy(original)]
    uploads[0]['classifier.bias'].add_(.3)  # uniform logit shift -> same CE to roundoff
    regions = [dict(id=0, profiles=[0, 1], numel=3, cells=[dict(key='classifier.bias', bounds=[(0, 3)], numel=3)])]
    observation = dict(own_gain=0., reliability={}, task_before={}, task_after={}, heads={
        seed: dict(before={}, after={}, instrument_probability_change=0.) for seed in cfg['head_seeds']})
    rng = rng_digest(); digest = model_digest(net)
    rows = utility_records(cfg, models, regions, [(1, 0), (2, 1)], [2, 6], uploads, [original, original],
        [observation, observation], measurement, {}, 2)
    assert len(rows) == 6
    assert all(r['sender'] != r['receiver'] and abs(r['delta_ce']) < 1e-6 for r in rows)
    assert set(r['alpha'] for r in rows) == {.25, .75}
    assert rng_digest() == rng and model_digest(net) == digest
    assert all(r['evidence']['S0'] is None for r in rows)
