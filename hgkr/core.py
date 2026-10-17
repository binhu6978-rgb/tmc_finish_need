"""Past-only shared-kernel history, copy-only reset, and anchored aggregation."""
import copy
import hashlib
import math
from collections import deque
from dataclasses import dataclass, asdict

import torch

from models.Fed import Aggregation_AdaptiveFL
from semconsfl.aggregation import prefix, validate


@dataclass(frozen=True)
class Config:
    reset_fraction: float = 0.01
    history_window: int = 5
    min_history: int = 3
    max_age: int = 20
    norm_epsilon: float = 1e-12
    budget_fraction: float = 0.80
    budget_match_tolerance: float = 0.02

    def __post_init__(self):
        if not (0 <= self.reset_fraction <= 1 and 0 < self.budget_fraction <= 1):
            raise ValueError("Invalid reset/budget fraction")
        if not (2 <= self.min_history <= self.history_window and self.max_age >= self.history_window):
            raise ValueError("Invalid history window")
        if self.norm_epsilon <= 0 or not 0 <= self.budget_match_tolerance < 1:
            raise ValueError("Invalid numerical/matching tolerance")


def keyed_seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2**63-1)


def clone_state(state):
    return {k: v.detach().clone() for k, v in state.items()}


def shared_specs(models):
    """An output-channel segment restricted to Small's input prefix, not a Full filter."""
    states = [m.state_dict() for m in models]
    if not all(s.keys() == states[-1].keys() for s in states):
        raise ValueError("Non-nested model keys")
    for s in states:
        for k, v in s.items():
            prefix(states[-1][k], v, k)
    return {k: tuple(v.shape) for k, v in states[0].items() if v.ndim == 4 and k.endswith('.weight')}


class KernelHistory:
    def __init__(self, specs, config):
        self.specs, self.config = specs, config
        self.records = {k: deque(maxlen=config.history_window) for k in specs}
        self.last_round = -1

    def scores(self, round_id):
        if round_id != self.last_round + 1:
            raise ValueError("History decision must precede this round's write")
        result = {}
        for key, shape in self.specs.items():
            records = [r for r in self.records[key] if 0 < round_id-r['round'] <= self.config.max_age]
            n = shape[0]
            count = torch.zeros(n, dtype=torch.long)
            total = torch.zeros((n, math.prod(shape[1:])), dtype=torch.float32)
            for r in records:
                count += r['valid'].long()
                total += r['direction']
            score = 1 - torch.linalg.vector_norm(total / count.clamp_min(1)[:, None], dim=1).clamp(0, 1)
            result[key] = {'score': score, 'valid': count >= self.config.min_history,
                           'observations': count, 'covered_rounds': len(records),
                           'ready': len(records) >= self.config.min_history}
        return result

    @torch.no_grad()
    def update(self, round_id, old, new, covered=None):
        if round_id != self.last_round + 1:
            raise ValueError("Duplicate/out-of-order history update")
        # Independent initial states make round0's change unsuitable as a learning direction.
        if round_id > 0:
            for key, shape in self.specs.items():
                # An observation requires complete coverage of this comparable segment.
                # In the fixed protocol EVERY nonempty slot covers the Small prefix.
                if covered is not None and not bool(covered[key]):
                    continue
                index = tuple(slice(0, d) for d in shape)
                delta = (new[key][index] - old[key][index]).detach().float().cpu().flatten(1)
                norm = torch.linalg.vector_norm(delta, dim=1)
                valid = torch.isfinite(delta).all(1) & (norm > self.config.norm_epsilon)
                direction = torch.zeros_like(delta)
                direction[valid] = delta[valid] / norm[valid, None]
                self.records[key].append({'round': round_id, 'direction': direction,
                                          'valid': valid, 'norm': norm})
        self.last_round = round_id

    def state_dict(self):
        return copy.deepcopy({'specs': self.specs, 'config': asdict(self.config),
                              'records': self.records, 'last_round': self.last_round})

    def load_state_dict(self, state):
        if state['specs'] != self.specs or state['config'] != asdict(self.config):
            raise ValueError('History checkpoint protocol mismatch')
        self.records = copy.deepcopy(state['records'])
        self.last_round = state['last_round']


def choose_kernels(scores, config, mode, seed, round_id, slot):
    if mode not in ('B0', 'B1', 'B2', 'B3'):
        raise ValueError('Unknown HGKR arm')
    chosen, diagnostics = {}, {}
    for key, info in scores.items():
        n = len(info['score'])
        generator = torch.Generator().manual_seed(keyed_seed(seed, round_id, slot, key, 'select'))
        random_order = torch.randperm(n, generator=generator).tolist()
        k = math.floor(config.reset_fraction * n) if info['ready'] and mode in ('B2', 'B3') else 0
        # Same structural warmup/count even if a direction is zero/invalid in one arm.
        # B3 falls back to the same isolated random ordering when evidence is insufficient.
        order = sorted(random_order, key=lambda i: (not bool(info['valid'][i]), -float(info['score'][i]) if info['valid'][i] else 0))
        ids = (order if mode == 'B3' else random_order)[:k]
        chosen[key] = ids
        diagnostics[key] = {'count': k, 'indices': ids, 'valid_history_kernels': int(info['valid'].sum()),
                            'covered_history_rounds': info['covered_rounds'],
                            'fallback_count': sum(not bool(info['valid'][i]) for i in ids),
                            'selected_instability_mean': sum(float(info['score'][i]) for i in ids)/k if k else None,
                            'all_instability_mean': float(info['score'].mean()),
                            'random_reference_overlap': len(set(ids) & set(random_order[:k]))}
    return chosen, diagnostics


@torch.no_grad()
def reset_copy(dispatched, old, specs, selected, seed, round_id, slot):
    """Input must be an owned clone; only selected shared segments change. CPU RNG is isolated."""
    squared = 0.0
    for key, shape in specs.items():
        ids = selected[key]
        if not ids:
            continue
        shared = old[key][:shape[0], :shape[1]].detach()
        mean, std = shared.double().mean().item(), shared.double().std(correction=0).item()
        generator = torch.Generator().manual_seed(keyed_seed(seed, round_id, slot, key, 'values'))
        values = torch.randn((len(ids), *shape[1:]), generator=generator) * std + mean
        values = values.to(dispatched[key].device)
        before = dispatched[key][ids, :shape[1]].clone()
        dispatched[key][ids, :shape[1]] = values
        squared += float((values - before).double().square().sum())
    return math.sqrt(squared)


@torch.no_grad()
def anchored_aggregate(old, uploads, starts, counts, round_id, reset_active):
    validate(old, uploads, counts, [1.] * len(uploads))
    if len(starts) != len(uploads):
        raise ValueError('Missing actual dispatch anchors')
    if round_id == 0:
        if reset_active:
            raise ValueError('Independent round0 must never reset')
        return Aggregation_AdaptiveFL(uploads, counts, old)
    for start, upload in zip(starts, uploads):
        if start.keys() != upload.keys() or any(start[k].shape != upload[k].shape for k in start):
            raise ValueError('Dispatch anchor/upload mismatch')
    if not reset_active:
        # Golden path avoids a reduction-order perturbation becoming an ablation factor.
        for start in starts:
            for k, v in start.items():
                if not torch.equal(v, old[k][prefix(old[k], v, k)]):
                    raise ValueError('No-reset start differs from synchronized server prefix')
        return Aggregation_AdaptiveFL(uploads, counts, old)
    return anchored_delta(old, uploads, starts, counts)


@torch.no_grad()
def anchored_delta(old, uploads, starts, counts):
    result = {}
    for k, theta in old.items():
        numerator, denominator = torch.zeros_like(theta), torch.zeros_like(theta)
        for upload, start, n in zip(uploads, starts, counts):
            index = prefix(theta, upload[k], k)
            numerator[index] += (upload[k] - start[k]) * n
            denominator[index] += n
        covered = denominator > 0
        new = theta.clone()
        new[covered] += numerator[covered] / denominator[covered]
        result[k] = new
    return result
