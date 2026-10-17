"""Synthetic correctness only: no CIFAR download or federated performance run."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from experiments.baseline import (content_hash, dispatch, initialize_models, load_partition,
                                  parse_args, write_json_new)
from experiments.run_semconsfl import atomic_checkpoint, read_checkpoint
from models.Fed import Aggregation_AdaptiveFL, get_model_list, select_clients, split_model
from models.resnet import ResNet18_cifar
from models.Update import LocalUpdate_AdaptiveFL
from semconsfl.aggregation import aggregate, coverage
from semconsfl.config import SemanticConfig
from semconsfl.measurement import (Signature, capture_rng, confusion, isolated_rng, measure,
                                  restore_rng)
from semconsfl.reliability_memory import ReliabilityMemory
from semconsfl.semantic_memory import SemanticMemory, js_divergence
from semconsfl.trainer import Memories, run_round
from utils.set_seed import set_random_seed


torch.set_num_threads(1)


def equal_states(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def equal_rng(a, b):
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    assert np.array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2:] == b["numpy"][2:]
    assert torch.equal(a["torch"], b["torch"])
    assert len(a["cuda"]) == len(b["cuda"])
    assert all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"]))


def cfg(**kw):
    return replace(SemanticConfig(), **kw)


def toy_states():
    full = {"conv.weight": torch.randn(3, 4, 2, 2),
            "bn.weight": torch.randn(3), "bn.bias": torch.randn(3),
            "classifier.weight": torch.randn(100, 3), "classifier.bias": torch.randn(100)}
    small = {k: (v[:2, :2].clone() if v.ndim > 1 else v[:2].clone()) for k, v in full.items()}
    return full, small


@pytest.mark.parametrize("synchronized", [False, True])
@pytest.mark.parametrize("round_id", [0, 4])
def test_gate_one_bitwise_original(synchronized, round_id):
    full, small = toy_states()
    if not synchronized:
        small = {k: torch.randn_like(v) for k, v in small.items()}
    uploads = [small, {k: v + 0.1 for k, v in full.items()}]
    expected = Aggregation_AdaptiveFL(uploads, [3, 7], full)
    equal_states(expected, aggregate(full, uploads, [3, 7], [1, 1], round_id))


def test_round_zero_forces_original_even_if_supplied_gate_is_lower():
    full, small = toy_states()
    equal_states(Aggregation_AdaptiveFL([small], [3], full),
                 aggregate(full, [small], [3], [0.8], 0))


def test_single_coverage_attenuation_and_uncovered_coordinates():
    full = {"x.weight": torch.tensor([10., 20., 30.])}
    local = {"x.weight": torch.tensor([15., 10.])}
    result = aggregate(full, [local], [12], [0.8], 1)["x.weight"]
    assert torch.equal(result, torch.tensor([14., 12., 30.]))


def test_multiple_coverage_uses_ungated_denominator():
    full = {"x.weight": torch.tensor([10., 20., 30.])}
    locals_ = [{"x.weight": torch.tensor([14.])}, {"x.weight": torch.tensor([12., 24.])}]
    actual = aggregate(full, locals_, [1, 3], [0.5, 0.75], 2)["x.weight"]
    assert torch.equal(actual, torch.tensor([11.625, 23., 30.]))


def test_conv_prefix_does_not_zero_fill_deltas():
    old = {"conv.weight": torch.full((3, 4, 2, 2), 5.)}
    local = {"conv.weight": torch.full((2, 1, 2, 2), 7.)}
    actual = aggregate(old, [local], [5], [0.75], 2)["conv.weight"]
    expected = old["conv.weight"].clone()
    expected[:2, :1] = 6.5
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("kind", ["missing", "shape", "nan", "integer", "scalar", "count", "gamma"])
def test_invalid_aggregation_fails_loudly(kind):
    full, small = toy_states()
    counts, gates = [2], [1]
    if kind == "missing": small.pop("bn.bias")
    if kind == "shape": small["conv.weight"] = torch.zeros(2, 2, 1, 2)
    if kind == "nan": small["bn.bias"][0] = float("nan")
    if kind == "integer": full["bn.bias"] = full["bn.bias"].long()
    if kind == "scalar": full["counter"] = small["counter"] = torch.tensor(1)
    if kind == "count": counts = [0]
    if kind == "gamma": gates = [float("nan")]
    with pytest.raises(ValueError): aggregate(full, [small], counts, gates, 2)


class TinyNet(nn.Module):
    def __init__(self, width=8, classes=100, tracking=False):
        super().__init__()
        self.feature = nn.Linear(4, width)
        self.bn = nn.BatchNorm1d(width, track_running_stats=tracking)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(width, classes)
        self.trace = []

    def forward(self, x):
        if self.training:
            self.trace.append(x[:, 0].detach().cpu().tolist())
        return {"output": self.classifier(self.dropout(torch.relu(self.bn(self.feature(x)))))}


def train_args(device="cpu"):
    return SimpleNamespace(model="resnet", dataset="cifar100", num_classes=100,
                           optimizer="sgd", lr=.01, lr_decay=.998, momentum=.5,
                           weight_decay=1e-4, device=torch.device(device), local_bs=4,
                           local_ep=2, num_users=100, frac=.1, client_chosen_mode="available",
                           client_hetero_ration="4:3:3", num_channels=3,
                           width_ration=[.4, .66, 1.], depth_saved=[2, 3, 4])


def data():
    x = torch.arange(96, dtype=torch.float32).reshape(24, 4) / 10
    return TensorDataset(x, torch.arange(24) % 3)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("tracking", [False, True])
def test_observer_preserves_training_batches_weights_gradients_modes_rng(device, tracking):
    args, configuration = train_args(device), cfg(measurement_batch_size=4)
    dataset = data()
    set_random_seed(43)
    original = TinyNet(tracking=tracking).to(device)
    initial_rng = capture_rng()
    pure = copy.deepcopy(original)
    pure_weights = LocalUpdate_AdaptiveFL(args, dataset, list(range(24))).train(3, pure)
    pure_rng = capture_rng()
    restore_rng(initial_rng)
    observer = copy.deepcopy(original)
    observer.dropout.eval()  # measurement restores mixed module modes
    flags = [m.training for m in observer.modules()]
    before = copy.deepcopy(observer.state_dict())
    pre = measure(observer, dataset, list(range(24)) + [1, 2], configuration, device)
    assert flags == [m.training for m in observer.modules()]
    equal_states(before, observer.state_dict())
    equal_rng(initial_rng, capture_rng())
    observed_weights = LocalUpdate_AdaptiveFL(args, dataset, list(range(24))).train(3, observer)
    gradients = [p.grad.clone() for p in observer.parameters()]
    post = measure(observer, dataset, range(24), configuration, device)
    equal_states(pure_weights, observed_weights)
    assert pure.trace == observer.trace
    assert all(torch.equal(g, p.grad) for g, p in zip(gradients, observer.parameters()))
    equal_rng(pure_rng, capture_rng())
    assert pre.metadata == post.metadata and torch.equal(pre.counts, post.counts)
    assert int(pre.counts.sum()) == 24 and pre.probabilities.shape == (100, 100)


def test_fixed_bn_batches_repeat_and_different_batches_can_change_measurement():
    model, dataset = TinyNet(), data()
    configuration = cfg(measurement_batch_size=4, min_class_samples=1)
    first = measure(model, dataset, range(24), configuration, "cpu")
    other = measure(model, dataset, range(24), replace(configuration, measurement_batch_size=6), "cpu")
    repeat = measure(model, dataset, reversed(range(24)), configuration, "cpu")
    assert torch.equal(first.probabilities, repeat.probabilities)
    assert first.metadata != other.metadata
    assert not torch.equal(first.probabilities, other.probabilities)


def signature(row=(.2, .7, .1), support=8):
    return Signature(torch.tensor([row, [0., 0., 0.], [0., 0., 0.]], dtype=torch.float64),
                     torch.tensor([support, 0, 0]), {"same_indices": True, "batch_size": 8})


def test_js_confidence_protection_support_and_unstable_error_tail():
    configuration = cfg(num_classes=3)
    memory = SemanticMemory(configuration)
    memory.anchors = {0: (1, torch.tensor([.875, .125], dtype=torch.float64))}
    pre, post = signature(), signature((.2, .1, .7))
    result = memory.score(pre, post, 2, memory.snapshot())
    assert result["anomaly"] > 0 and result["classes"][0]["signed_drift"] > 0
    improved = memory.score(pre, signature((.3, .1, .6)), 2, memory.snapshot())
    assert improved["anomaly"] == 0
    assert confusion(signature(support=2), 0, configuration)[0] is None
    assert confusion(signature((1., 0., 0.)), 0, configuration)[1] == "insufficient_error_mass"
    bad = signature((float("nan"), .5, .5))
    assert confusion(bad, 0, configuration)[0] is None
    assert 0 <= js_divergence(torch.tensor([1., 0.]), torch.tensor([0., 1.])) <= math.log(2)
    with pytest.raises(ValueError):
        memory.score(pre, signature(support=7), 2, memory.snapshot())


def test_anchor_round_zero_old_snapshot_and_duplicate_client_single_vote():
    configuration = cfg(num_classes=3)
    memory = SemanticMemory(configuration)
    b, alternative = signature(), signature((.2, .1, .7))
    memory.update(0, [(1, b), (2, b)])
    assert memory.anchors == {}
    old = memory.snapshot()
    memory.update(1, [(1, b)] * 20 + [(2, alternative)])
    assert memory.score(b, alternative, 2, old)["anomaly"] is None
    assert torch.allclose(memory.anchors[0][1], torch.tensor([.5, .5], dtype=torch.float64))
    assert memory.anchors[0][0] == 1
    assert memory.score(b, alternative, 102, memory.snapshot())["anomaly"] is None
    with pytest.raises(ValueError): memory.score(b, alternative, 1, memory.snapshot())
    snapshot = memory.snapshot()
    memory.update(2, [(1, b), (2, b)])
    assert torch.allclose(snapshot[0][1], torch.tensor([.5, .5], dtype=torch.float64))
    # A single physical client cannot initialize a robust two-client reference.
    lone = SemanticMemory(configuration)
    lone.update(1, [(1, b)] * 20)
    assert lone.anchors == {}


def history_ready(configuration):
    memory = ReliabilityMemory(configuration)
    memory.profiles = {0: [(1, i, 0.) for i in range(10)],
                       1: [(1, i, .4) for i in range(10)]}
    memory.clients = {42: [(1, .8), (2, .6)]}
    return memory


def test_reliability_uses_only_old_profile_calibrated_and_client_shrunk_values():
    configuration = cfg()
    memory = history_ready(configuration)
    old = memory.snapshot()
    d0 = memory.decide(42, 0, .2, 3, old)
    d1 = memory.decide(42, 1, .2, 3, old)
    assert d0["z"] == 1 and d1["z"] == 0
    assert d0["r"] == pytest.approx(2 / 4 * .7)
    assert d0["risk"] == pytest.approx(.35)
    assert .75 <= d0["gamma"] <= 1
    assert d1["gamma"] == 1
    memory.update(3, [(42, 0, .2, d0)] * 15 + [(42, 1, .2, d1),
                                            (43, 0, .1, d0)])
    assert len(memory.clients[42]) == 3
    assert memory.clients[42][-1] == (3, .5)
    assert len([r for r in memory.profiles[0] if r[0] == 3 and r[1] == 42]) == 1
    assert memory.decide(42, 0, .2, 3, old) == d0
    with pytest.raises(ValueError): memory.decide(42, 0, .2, 3, memory.snapshot())


@pytest.mark.parametrize("case", ["no_profile", "no_client", "missing", "stale", "few"])
def test_missing_or_stale_evidence_is_neutral_and_not_fake_zero_history(case):
    memory = history_ready(cfg())
    anomaly, round_id, client, profile = .2, 3, 42, 0
    if case == "no_profile": profile = 2
    if case == "no_client": client = 99
    if case == "missing": anomaly = None
    if case == "stale": round_id = 200
    if case == "few": memory.profiles[0] = memory.profiles[0][:2]
    decision = memory.decide(client, profile, anomaly, round_id, memory.snapshot())
    assert decision["gamma"] == 1 and decision["risk"] == 0
    before = len(memory.clients.get(client, []))
    memory.update(round_id, [(client, profile, anomaly, decision)])
    if decision["z"] is None:
        assert len(memory.clients.get(client, [])) == before


def test_three_round_synthetic_a0_a1_a2_fallback_trajectory_and_rng_identity():
    configuration = cfg(min_class_samples=1, measurement_batch_size=4)
    dataset = data()
    users = {0: list(range(12)), 1: list(range(12, 24))}
    trajectories = []
    for mode in ("A0", "A1", "A2"):
        set_random_seed(7)
        models = [TinyNet(width) for width in range(2, 9)]
        masks, _ = coverage(models)
        memories = Memories(configuration)
        trajectory = []
        for t in range(3):
            report = run_round(models, dataset, users, [(0, 0), (1, 6), (0, 3)],
                               train_args(), configuration, memories, mode, t, masks)
            assert report["gate_min"] == 1 and report["duplicate_slots"] == 1
            trajectory.append((copy.deepcopy(models[-1].state_dict()), capture_rng()))
        trajectories.append(trajectory)
    for trajectory in trajectories[1:]:
        for (a, rng_a), (b, rng_b) in zip(trajectories[0], trajectory):
            equal_states(a, b)
            equal_rng(rng_a, rng_b)


def test_baseline_default_settings_scheduler_and_original_initialization():
    args, _, configuration = parse_args(["--gpu", "-1"])
    assert (args.epochs, args.num_users, args.frac, args.local_ep, args.local_bs) == (1000, 100, .1, 5, 50)
    assert (args.lr, args.lr_decay, args.momentum, args.weight_decay) == (.01, .998, .5, 1e-4)
    assert configuration == SemanticConfig()
    set_random_seed(1)
    original = []
    for _ in range(10):
        profiles = np.random.choice(range(7), 10)
        original.append(list(zip(select_clients(args, profiles, 7), map(int, profiles))))
    set_random_seed(1)
    assert [dispatch(args) for _ in range(10)] == original
    for slots in original:
        for client, profile in slots:
            assert client >= (0 if profile < 3 else 40 if profile < 6 else 70)
    with isolated_rng():
        set_random_seed(1)
        unused = ResNet18_cifar(num_channels=3, num_classes=100)
        del unused
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            golden, _ = get_model_list(args)
        set_random_seed(1)
        actual, profiles = initialize_models(args)
        assert len(profiles) == 7
        for a, b in zip(golden, actual): equal_states(a.state_dict(), b.state_dict())
        _, stats = coverage(actual)
        assert stats["total_parameters"] == sum(p.numel() for p in actual[-1].parameters())
        assert all(stats[k] > 0 for k in ("all_profiles", "partial_profiles", "full_only"))
        # No running-stat/scalar state enters the real seven-profile aggregators.
        assert all(v.dtype == torch.float32 and v.ndim > 0 for v in actual[-1].state_dict().values())


def test_partition_is_read_only_and_rejects_duplicates_no_new_split(tmp_path):
    args = SimpleNamespace(num_users=2, seed=1, iid=0, data_beta=.3, local_bs=2, num_classes=3)
    payload = {"dataset": "cifar100", "num_users": 2, "seed": 1, "iid": False,
               "data_beta": .3, "train_data": {"0": [0, 1], "1": [2, 3]}}
    path = tmp_path / "partition.json"
    write_json_new(path, payload)
    before = path.read_bytes()
    users, report = load_partition(path, args, [0, 1, 2, 2])
    assert users == {0: [0, 1], 1: [2, 3]} and report["total_indices"] == 4
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError): write_json_new(path, payload)
    payload["train_data"]["1"] = [1, 3]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): load_partition(path, args, [0, 1, 2, 2])
    with pytest.raises(ValueError): parse_args(["--gpu", "-1", "--client_chosen_mode", "RL"])
    with pytest.raises(SystemExit): parse_args(["--gpu", "-1", "--generate_data", "1"])


def test_checkpoint_round_boundary_source_mode_config_and_rng(tmp_path):
    path, log = tmp_path / "latest.pt", tmp_path / "rounds.jsonl"
    state = {"fingerprint": "frozen", "mode": "A1", "next_round": 1,
             "memories": {"last_round": 0}, "rng": capture_rng()}
    atomic_checkpoint(path, state)
    log.write_text('{"round":0}\n')
    loaded = read_checkpoint(path, "frozen", log, "A1")
    equal_rng(state["rng"], loaded["rng"])
    with pytest.raises(ValueError): read_checkpoint(path, "changed", log, "A1")
    with pytest.raises(ValueError): read_checkpoint(path, "frozen", log, "A2")
    log.write_text('{"round":0}\n{"round":1}\n')
    with pytest.raises(ValueError): read_checkpoint(path, "frozen", log, "A1")


def test_measurement_exception_restores_modes_and_rng():
    class Broken(TinyNet):
        def forward(self, x):
            torch.rand(2)
            np.random.rand()
            random.random()
            raise RuntimeError("synthetic intentional failure")
    model = Broken()
    state = capture_rng()
    with pytest.raises(RuntimeError, match="intentional"):
        measure(model, data(), range(24), cfg(measurement_batch_size=4), "cpu")
    assert model.training
    equal_rng(state, capture_rng())


def test_a2_actual_gate_integration_and_post_decision_history(monkeypatch):
    import semconsfl.trainer as trainer
    configuration = cfg(measurement_batch_size=4)
    set_random_seed(9)
    models = [TinyNet(width) for width in range(2, 9)]
    masks, _ = coverage(models)
    old = copy.deepcopy(models[-1].state_dict())
    memories = Memories(configuration)
    memories.last_round = 2
    memories.reliability = history_ready(configuration)
    pre = torch.zeros(100, 100, dtype=torch.float64)
    post = pre.clone()
    pre[0, :3] = torch.tensor([.2, .7, .1])
    post[0, :3] = torch.tensor([.2, .1, .7])
    counts = torch.zeros(100, dtype=torch.long)
    counts[0] = 12
    before = Signature(pre, counts, {"same": True})
    after = Signature(post, counts.clone(), {"same": True})
    memories.semantic.anchors[0] = (2, pre[0, 1:] / pre[0, 1:].sum())
    memories.reliability.profiles[6] = memories.reliability.profiles[0]
    measures = iter([before, after])
    monkeypatch.setattr(trainer, "measure", lambda *a, **kw: next(measures))
    uploads = []
    original_train = LocalUpdate_AdaptiveFL.train
    def record_train(self, round, net):
        result = original_train(self, round, net)
        uploads.append(copy.deepcopy(result))
        return result
    monkeypatch.setattr(LocalUpdate_AdaptiveFL, "train", record_train)
    report = run_round(models, data(), {42: list(range(12))}, [(42, 6)], train_args(),
                       configuration, memories, "A2", 3, masks)
    gamma = 1 - .25 * .35
    assert report["slots"][0]["gamma"] == pytest.approx(gamma)
    expected = aggregate(old, uploads, [12], [gamma], 3)
    equal_states(models[-1].state_dict(), expected)
    assert report["coverage_update_diagnostics"]["full_only"]["norm_ratio"] == pytest.approx(gamma, abs=1e-5)
    assert len(memories.reliability.clients[42]) == 3
    assert memories.reliability.clients[42][-1] == (3, 1.)


def test_synthetic_resume_reproduces_uninterrupted_rounds(tmp_path):
    configuration, dataset = cfg(min_class_samples=1, measurement_batch_size=4), data()
    users, slots = {0: list(range(12)), 1: list(range(12, 24))}, [(0, 0), (1, 6)]
    set_random_seed(21)
    models = [TinyNet(width) for width in range(2, 9)]
    masks, _ = coverage(models)
    memories = Memories(configuration)
    run_round(models, dataset, users, slots, train_args(), configuration, memories, "A1", 0, masks)
    checkpoint = {"global": copy.deepcopy(models[-1].state_dict()),
                  "memories": memories.state_dict(), "rng": capture_rng()}
    path = tmp_path / "synthetic.pt"
    atomic_checkpoint(path, checkpoint)
    run_round(models, dataset, users, slots, train_args(), configuration, memories, "A1", 1, masks)
    expected, expected_rng = copy.deepcopy(models[-1].state_dict()), capture_rng()
    saved = torch.load(path, weights_only=False)
    resumed = [TinyNet(width) for width in range(2, 9)]
    for model in resumed:
        model.load_state_dict(split_model(saved["global"], model.state_dict()))
    restored_memories = Memories(configuration)
    restored_memories.load_state_dict(saved["memories"])
    restore_rng(saved["rng"])
    run_round(resumed, dataset, users, slots, train_args(), configuration, restored_memories, "A1", 1, masks)
    equal_states(expected, resumed[-1].state_dict())
    equal_rng(expected_rng, capture_rng())


def test_measurement_detects_same_counts_but_changed_class_correspondence():
    dataset = data()
    model = TinyNet()
    configuration = cfg(measurement_batch_size=4, min_class_samples=1)
    pre = measure(model, dataset, range(24), configuration, "cpu")
    shifted = TensorDataset(dataset.tensors[0], dataset.tensors[1].roll(1))
    post = measure(model, shifted, range(24), configuration, "cpu")
    assert torch.equal(pre.counts, post.counts)
    with pytest.raises(ValueError, match="Pre/post"):
        SemanticMemory(configuration).score(pre, post, 1, {})
