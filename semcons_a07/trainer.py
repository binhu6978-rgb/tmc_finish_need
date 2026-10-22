"""Scoped instrumentation of the original loop, never a copied training engine."""
import copy
from contextlib import contextmanager
import json
import time
from collections import defaultdict
import torch
import adapterfl_methods as methods
import capability_final40 as frozen
import Algorithm.Training_AdaptiveFL as training
from .diagnostics import append_json, model_digest, rng_digest, utility_records
from .measurement import Measurement
from .semantic_memory import SemanticMemory
from .reliability_memory import ReliabilityMemory

OriginalHooks = frozen.Final40Hooks
OriginalLog = methods.RunLog


def plain_observation(obs):
    if obs is None:
        return None
    return {k: v for k, v in obs.items() if k not in ('task_before', 'task_after', 'heads')} | dict(
        heads={str(seed): {k: v for k, v in head.items() if k not in ('before', 'after')}
               for seed, head in obs['heads'].items()})


class Observer:
    def __init__(self, cfg, dataset, users, path, enabled):
        self.cfg = cfg; self.path = path; self.enabled = enabled
        self.measurement = Measurement(cfg, dataset, users, 100)
        self.semantic = SemanticMemory(cfg); self.reliability = ReliabilityMemory(cfg)
        self.observations = []; self.dispatches = []; self.local_checks = []; self.slot_cursor = 0
        self.slots = []; self.round_number = 0; self.pairs = 0

    def start(self, clients, profiles):
        self.round_number += 1
        self.slots = [(int(c), int(p)) for c, p in zip(clients, profiles)]
        self.anchors = self.semantic.snapshot(); self.history = self.reliability.snapshot()
        self.observations = []; self.dispatches = []; self.local_checks = []; self.slot_cursor = 0

    def wrap(self, local):
        slot = self.slot_cursor; self.slot_cursor += 1
        cid, profile = self.slots[slot]
        original_train = local.train

        def train(round, net):
            if round + 1 != self.round_number:
                raise AssertionError('Observation round differs from actual SGD')
            before_rng = rng_digest()
            diagnostic = any(self.round_number in r for r in self.cfg['diagnostic_rounds'].values())
            if self.enabled and diagnostic:
                self.dispatches.append({k: v.detach().clone() for k, v in net.state_dict().items()})
            start = time.perf_counter()
            pre = self.measurement.before(net, cid) if self.enabled else None
            pre_seconds = time.perf_counter() - start
            if rng_digest() != before_rng:
                raise AssertionError('Pre measurement changed training RNG')
            start = time.perf_counter()
            result = original_train(round=round, net=net)
            native_seconds = time.perf_counter() - start
            after_rng = rng_digest()
            post_state = model_digest(net) if self.round_number <= 20 else None
            gradient_snapshot = [None if p.grad is None else p.grad.detach().clone() for p in net.parameters()] if self.enabled else []
            obs = self.measurement.after(net, cid, profile, self.round_number, pre, self.anchors,
                self.reliability.read(self.history, cid, profile, self.round_number)) if self.enabled else None
            if rng_digest() != after_rng:
                raise AssertionError('Post measurement changed training RNG')
            if self.enabled:
                if self.round_number <= 20 and model_digest(net) != post_state:
                    raise AssertionError('Measurement changed main state/BN buffers')
                for p, grad in zip(net.parameters(), gradient_snapshot):
                    if (p.grad is None) != (grad is None) or (grad is not None and not torch.equal(p.grad, grad)):
                        raise AssertionError('Measurement changed Backbone gradients')
            if obs:
                obs['pre_measurement_seconds'] = pre_seconds
                obs['native_local_train_seconds'] = native_seconds
            self.observations.append(obs)
            self.local_checks.append(dict(slot=slot, client=cid, profile=profile,
                before_train_rng=before_rng, after_train_rng=after_rng, uploaded_state=post_state))
            local.adapterfl_metrics = plain_observation(obs) or {}
            return result
        local.train = train

    def after_aggregate(self, models, hook, uploads, lens):
        if not self.enabled or not self.dispatches:
            return
        begin = time.perf_counter(); rng = rng_digest()
        records = utility_records(self.cfg, models, hook.regions, self.slots, lens, uploads, self.dispatches,
                                  self.observations, self.measurement, self.anchors, self.round_number)
        if rng_digest() != rng:
            raise AssertionError('Utility changed training RNG')
        # No receiver images, representations or model tensors are written.
        for record in records:
            append_json(self.path / 'utility_pairs.jsonl', record)
        self.pairs += len(records) // len(self.cfg['head_seeds'])
        append_json(self.path / 'utility_rounds.jsonl', dict(round=self.round_number,
            physical_pairs=len(records)//len(self.cfg['head_seeds']), negative_loo_pairs=sum(
                r['negative_loo'] for r in records if r['head_seed']==self.cfg['head_seeds'][0]),
            seconds=time.perf_counter()-begin, receiver_forward_examples=sum(
                r['receiver_forward_examples'] for r in records if r['head_seed']==self.cfg['head_seeds'][0])))
        self.dispatches = []

    def commit(self):
        records = []; history = []
        for obs in self.observations:
            if obs is None:
                continue
            cid = obs['client']; profile = obs['profile']
            records.append((cid, 'task', obs['task_after']))
            for seed, head in obs['heads'].items():
                records.append((cid, seed, head['after']))
            value = obs['heads'][self.cfg['head_seeds'][0]]['evidence']['S1']
            history.append((cid, profile, value))
        semantic = self.semantic.commit(self.round_number, records)
        self.reliability.commit(self.round_number, history)
        self.measurement.commit()
        comparability = {}
        for seed in self.cfg['head_seeds']:
            grouped = defaultdict(list)
            confidences = []; accuracies = []
            for obs in self.observations:
                if obs is None:
                    continue
                for y, p in obs['heads'][seed]['after'].items():
                    grouped[obs['client'], y].append(p['embedding'])
                    confidences.append(float(p['probability'][y]))
                    accuracies.append(float(int(p['probability'].argmax()) == y))
            pooled = {key: torch.stack(v).mean(0) for key, v in grouped.items()}
            if pooled:
                identities = list(pooled)
                vectors = torch.nn.functional.normalize(torch.stack(list(pooled.values())), dim=1)
                pair_cos = vectors @ vectors.T
                cids = torch.tensor([c for c, y in identities]); labels = torch.tensor([y for c, y in identities])
                distinct = cids[:, None] < cids[None, :]
                same = pair_cos[distinct & (labels[:, None] == labels[None, :])]
                different = pair_cos[distinct & (labels[:, None] != labels[None, :])]
            else:
                same = different = torch.tensor([])
            comparability[seed] = dict(same_class_pairs=same.numel(), different_class_pairs=different.numel(),
                same_class_cosine=float(same.mean()) if same.numel() else None,
                different_class_cosine=float(different.mean()) if different.numel() else None,
                prototype_true_class_probability=sum(confidences)/len(confidences) if confidences else None,
                prototype_classification_accuracy=sum(accuracies)/len(accuracies) if accuracies else None)
        append_json(self.path / 'observation_rounds.jsonl', dict(round=self.round_number, gamma=1,
            semantic=semantic, clients=[plain_observation(o) for o in self.observations],
            adapter_storage_bytes=self.measurement.storage_bytes(), head_sha256=self.measurement.head_hashes,
            comparability=comparability,
            nonneutral_gate_count=0, affected_parameter_numel=0, gate_learning_delta_l2=0))


@contextmanager
def instrument_original_loop(cfg, dataset, users, path, enabled):
    observer = Observer(cfg, dataset, users, path, enabled)
    old_select = training.select_clients

    def select(args, profiles, number):
        clients = old_select(args, profiles, number)
        observer.start(clients, profiles)
        return clients

    class Hooks(OriginalHooks):
        def __init__(self, args, models):
            super().__init__(args, models)
            self.models = models

        def configure_local(self, local):
            super().configure_local(local)
            observer.wrap(local)

        def aggregate(self, uploads, lens, previous, dispatched=None):
            # This exact call is the frozen A07 computational path, including round1.
            result = super().aggregate(uploads, lens, previous, dispatched)
            observer.after_aggregate(self.models, self, uploads, lens)
            return result

    class Log(OriginalLog):
        def __init__(self, args, models):
            super().__init__(args, models)
            append_json(path / 'regression.jsonl', dict(round=0, models=[model_digest(m) for m in models], rng=rng_digest()))

        def record(self, round_index, models, evaluations, *args):
            super().record(round_index, models, evaluations, *args)
            if enabled:
                observer.commit()
            append_json(path / 'regression.jsonl', dict(round=round_index+1,
                models=[model_digest(m) for m in models], rng=rng_digest(), evaluations=evaluations,
                slots=observer.slots, local_checks=observer.local_checks))

    try:
        training.select_clients = select
        frozen.Final40Hooks = Hooks
        methods.RunLog = Log
        yield observer
    finally:
        training.select_clients = old_select
        frozen.Final40Hooks = OriginalHooks
        methods.RunLog = OriginalLog
