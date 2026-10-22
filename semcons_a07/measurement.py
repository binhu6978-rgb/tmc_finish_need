"""Fixed training probes, detached features, frozen pre/post instrument."""
import copy
import hashlib
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from adapterfl_methods import isolated_evaluation_rng
from .adapter import Instrument, merge_candidates
from .config import seed_of
from .prototypes import summarize, scores


def probe_indices(dataset, indices, cid, samples, maximum_classes=None):
    groups = {}
    for i in sorted(set(map(int, indices))):
        groups.setdefault(int(dataset.targets[i]), []).append(i)
    classes = sorted([c for c, values in groups.items() if len(values) >= samples],
                     key=lambda c: seed_of('classes-v1', cid, c))
    if maximum_classes is not None:
        classes = classes[:maximum_classes]
    return [i for c in sorted(classes) for i in sorted(groups[c], key=lambda i: seed_of('probe-v1', cid, i))[:samples]]


def materialize(dataset, indices, cid, batch_size):
    batches = []
    with isolated_evaluation_rng():
        seed = seed_of('probe-views-v1', cid)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        values = [dataset[i] for i in indices]
        for start in range(0, len(values), batch_size):
            part = values[start:start+batch_size]
            batches.append((torch.stack([v[0] for v in part]), torch.tensor([v[1] for v in part], dtype=torch.long)))
    return batches


@torch.no_grad()
def features(net, batches):
    model = copy.deepcopy(net).eval()
    device = next(model.parameters()).device
    hh = []; qq = []; yy = []
    for x, y in batches:
        output = model(x.to(device))
        hh.append(F.adaptive_avg_pool2d(output['representation'], 1).flatten(1).detach())
        qq.append(output['output'].softmax(1).detach())
        yy.append(y.to(device))
    if not batches:
        return None
    return torch.cat(hh), torch.cat(qq), torch.cat(yy)


def head_digest(instrument):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for part in (instrument.head, instrument.decoder)
                                  for v in part.state_dict().values())).hexdigest()


class Measurement:
    def __init__(self, cfg, dataset, users, classes):
        self.cfg = cfg; self.dataset = dataset; self.users = users; self.classes = classes
        self.bank = {}; self.pending = {}; self.plans = {}; self.head_hashes = {}

    def batches(self, cid, utility=False):
        key = cid, utility
        if key not in self.plans:
            count = self.cfg['utility_samples_per_class'] if utility else self.cfg['samples_per_class']
            maximum = self.cfg['utility_classes_per_receiver'] if utility else None
            indices = probe_indices(self.dataset, self.users[cid], cid, count, maximum)
            self.plans[key] = materialize(self.dataset, indices, cid, self.cfg['measurement_batch_size'])
        return self.plans[key]

    def before(self, net, cid):
        with isolated_evaluation_rng():
            value = features(net, self.batches(cid))
            if next(net.parameters()).is_cuda:
                torch.cuda.synchronize(next(net.parameters()).device)
            return value

    def after(self, net, cid, profile, round_number, before, anchors, reliability):
        started = time.perf_counter()
        with isolated_evaluation_rng():
            after = features(net, self.batches(cid))
            if before is None or after is None:
                return None
            pre_h, pre_q, labels = before
            post_h, post_q, post_labels = after
            if not torch.equal(labels, post_labels):
                raise AssertionError('Paired class/batch order changed')
            cpu_labels = labels.cpu()
            task_before = summarize(None, pre_q.cpu(), cpu_labels, self.cfg['minimum_class_support'])
            task_after = summarize(None, post_q.cpu(), cpu_labels, self.cfg['minimum_class_support'])
            own_gain = float(F.nll_loss(pre_q.clamp_min(1e-12).log(), labels) - F.nll_loss(post_q.clamp_min(1e-12).log(), labels))
            record = dict(client=cid, profile=profile, task_before=task_before, task_after=task_after,
                          own_gain=own_gain, heads={}, reliability=reliability,
                          probe_examples=len(labels), backbone_forward_examples=2*len(labels), adapter_backward_examples=0)
            for seed in self.cfg['head_seeds']:
                key = cid, profile, seed
                instrument = Instrument(pre_h.shape[1], self.classes, self.cfg['embedding_dim'], seed, cid, profile,
                                        self.bank.get(key)).to(pre_h.device)
                digest = head_digest(instrument)
                if seed in self.head_hashes and self.head_hashes[seed] != digest:
                    raise AssertionError('Frozen head differs between clients/profiles')
                self.head_hashes[seed] = digest
                with torch.no_grad():
                    z0, l0 = instrument(pre_h); z1, l1 = instrument(post_h)
                p0 = summarize(z0.cpu(), l0.softmax(1).cpu(), cpu_labels, self.cfg['minimum_class_support'])
                p1 = summarize(z1.cpu(), l1.softmax(1).cpu(), cpu_labels, self.cfg['minimum_class_support'])
                evidence = scores(p0, p1, task_before, task_after, anchors.get(seed, {}), anchors.get('task', {}),
                                  round_number, self.cfg['anchor_max_age'])
                candidate, backward_examples = instrument.candidate(post_h.detach(), labels, self.cfg)
                self.pending.setdefault(key, []).append(candidate)
                with torch.no_grad():
                    _, changed_logits = instrument(post_h)
                drift = float((changed_logits.softmax(1) - l1.softmax(1)).abs().mean())
                record['adapter_backward_examples'] += backward_examples
                record['heads'][seed] = dict(before=p0, after=p1, evidence=evidence,
                    adapter_snapshot_prior=key in self.bank, instrument_probability_change=drift,
                    head_sha256=digest, post_measurement_ce=float(F.cross_entropy(l1, labels)),
                    post_candidate_ce=float(F.cross_entropy(changed_logits, labels)))
                if any(p.grad is not None for p in instrument.head.parameters()) or any(p.grad is not None for p in instrument.decoder.parameters()):
                    raise AssertionError('Frozen head received gradients')
            record['logical_upload_bytes'] = sum(v.numel()*v.element_size() for head in record['heads'].values()
                for proto in head['after'].values() for v in (proto['embedding'], proto['probability'])) + sum(
                    p['probability'].numel()*p['probability'].element_size() for p in task_after.values())
        record['post_measurement_seconds'] = time.perf_counter() - started
        return record

    def commit(self):
        for key, candidates in self.pending.items():
            self.bank[key] = merge_candidates(candidates)
        self.pending = {}

    def storage_bytes(self):
        return sum(v.numel()*v.element_size() for state in self.bank.values() for v in state.values())
