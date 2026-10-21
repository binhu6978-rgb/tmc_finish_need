"""Fixed final20 server ablations and condition-aware existing G03/P04/P05.

Reuses Native aggregation, real prefix geometry and existing capability weights.
No client-side change, score, memory, loss or test-dependent decision.
"""
from collections import defaultdict
import math

import torch
from torch import nn

from models.Fed import Aggregation_AdaptiveFL
from native_group_aggregation import (balanced_weights, capability_weights,
    write_weighted_region, nonneutral_weights, norm_cosine, update_norm)
from semcons_scope import build_regions, slices

ABLATIONS = tuple(f'a{i:02d}' for i in range(1, 21))


def source_weights(method, members, slots, lens, boundaries, population, data_mass):
    """All weights retain Native total covered mass and normalize per region."""
    groups = defaultdict(list)
    for i in members:
        groups[sum(slots[i][0] >= b for b in boundaries)].append(i)
    if len(groups) == 1 or method == 'baseline':
        return [lens[i] for i in members]
    if method == 'g03':
        return balanced_weights('g03', members, slots, lens, boundaries)
    if method in ('p04', 'p05'):
        return capability_weights(method, members, slots, lens, boundaries, population)
    if method not in ABLATIONS:
        raise ValueError('Unknown fixed method')
    # A01/A02 use P04, A03/A04 use P05. The scale is applied later.
    if method in ('a03', 'a04'):
        return capability_weights('p05', members, slots, lens, boundaries, population)
    if method not in tuple(f'a{i:02}' for i in range(11, 18)) + ('a19', 'a20'):
        return capability_weights('p04', members, slots, lens, boundaries, population)
    masses = {}
    for g, ids in groups.items():
        if method in ('a14', 'a15'):
            ids = [i for i, (cid, _) in enumerate(slots) if sum(cid >= b for b in boundaries) == g]
        if method in ('a11', 'a15'):
            unique = {slots[i][0]: lens[i] for i in ids}
            masses[g] = sum(unique.values())
        elif method == 'a12':
            masses[g] = len({slots[i][0] for i in ids})
        elif method == 'a13':
            masses[g] = len(ids)
        else:
            masses[g] = sum(lens[i] for i in ids)
    exponent = -.25 if method == 'a19' else -.75 if method == 'a20' else -.5
    priors = population if method == 'a16' else data_mass if method == 'a17' else (1., 1., 1.)
    targets = {g: priors[g] * masses[g] ** exponent for g in groups}
    total = sum(lens[i] for i in members)
    denominator = sum(targets.values())
    result = {}
    for g, ids in groups.items():
        target = total * targets[g] / denominator
        if method == 'a11':
            clients = defaultdict(list)
            for i in ids:
                clients[slots[i][0]].append(i)
            for js in clients.values():
                for i in js:
                    result[i] = target * lens[i] / (len(js) * masses[g])
        else:
            mass = sum(lens[i] for i in ids)
            for i in ids:
                result[i] = target * lens[i] / mass
    return [result[i] for i in members]


def active_cells(method, region, bn_keys):
    if method == 'a05' and len(region['profiles']) != 7:
        return []
    if method == 'a06' and not 1 < len(region['profiles']) < 7:
        return []
    def allowed(key):
        classifier = key in ('classifier.weight', 'classifier.bias')
        bn = key in bn_keys
        return not ((method in ('a07', 'a10') and classifier)
                    or (method == 'a08' and not classifier)
                    or (method in ('a09', 'a10') and bn))
    return [c for c in region['cells'] if allowed(c['key'])]


class Final40Hooks:
    is_native_groups = True
    needs_dispatch = False

    def __init__(self, args, models):
        context = args._final40_context
        if args.method not in ABLATIONS + ('baseline', 'g03', 'p04', 'p05') or len(models) != 7:
            raise ValueError('Fixed final40 original seven models required')
        self.method = args.method
        self.population = tuple(context['population'])
        self.boundaries = tuple(context['boundaries'])
        self.data_mass = tuple(context['data_mass'])
        if args.num_users != 100 or sum(self.population) != 100:
            raise ValueError('100 physical clients required')
        self.shapes = [{k: tuple(v.shape) for k, v in m.state_dict().items()} for m in models]
        self.regions = build_regions(models)
        self.bn_keys = {name + '.' + p for name, module in models[-1].named_modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm)
                        for p in ('weight', 'bias') if getattr(module, p) is not None}
        if models[-1].classifier.out_features != args.num_classes:
            raise ValueError('Dataset output classes differ')
        self.config = dict(version=1, mode=self.method, first_round='exact Native',
            context=context, region_count=len(self.regions),
            masses='covered sample-slot mass except explicit A11-A15 definitions',
            normalization='Native covered-coordinate normalization; original uploaded states, not a replacement delta rule',
            single_tier='Native source weights; A01-A04 global norm control may rescale covered single-tier regions',
            scope='A06 means partially shared only (1 < profile support < 7); BN keys from module types',
            scale='same-input Full displacement from original Full anchor; zero source/nonzero target raises')
        self.rounds_seen = 0
        self.slots = None
        self.round_diagnostics = {}

    def configure_local(self, local):
        pass

    def set_slots(self, slots):
        self.slots = [(int(c), int(p)) for c, p in slots]

    def state_dict(self):
        return dict(method=self.method, config=self.config, rounds_seen=self.rounds_seen)

    def load_state_dict(self, state, full):
        raise ValueError('Final fixed budget is fresh/no-PT; resume not supported')

    @torch.no_grad()
    def aggregate(self, uploads, lens, previous, dispatched=None):
        if not uploads or self.slots is None or len(uploads) != len(self.slots) or len(lens) != len(uploads):
            raise ValueError('Actual slots and masses required')
        seen = {}
        for i, (cid, profile) in enumerate(self.slots):
            if not 0 <= cid < 100 or not 0 <= profile < 7 or not math.isfinite(lens[i]) or lens[i] <= 0:
                raise ValueError('Invalid upload identity/mass')
            if cid in seen and seen[cid] != lens[i]:
                raise ValueError('Repeated physical client must have identical dataset mass')
            seen[cid] = lens[i]
            if {k: tuple(v.shape) for k, v in uploads[i].items()} != self.shapes[profile]:
                raise ValueError('Actual upload profile shapes differ')
        native = Aggregation_AdaptiveFL(uploads, lens, previous)
        result = {k: v.clone() for k, v in native.items()}
        p04 = {k: v.clone() for k, v in native.items()}
        warmup = self.rounds_seen == 0
        details = []
        for region in self.regions:
            members = [i for i, (_, profile) in enumerate(self.slots) if profile in region['profiles']]
            tiers = sorted({sum(self.slots[i][0] >= b for b in self.boundaries) for i in members})
            record = dict(region_id=region['id'], numel=region['numel'], support_profiles=region['profiles'],
                slots=members, capability_tiers=tiers,
                distinct_physical_clients=len({self.slots[i][0] for i in members}),
                reason='first_round_native' if warmup else 'no_coverage' if not members else 'native',
                weighted_parameter_numel=0)
            details.append(record)
            if not members:
                continue
            mass = sum(lens[i] for i in members)
            shares = [lens[i] / mass for i in members]
            record.update(native_slot_shares=shares, applied_slot_shares=list(shares))
            if warmup:
                continue
            pw = source_weights('p04', members, self.slots, lens, self.boundaries, self.population, self.data_mass)
            record['virtual_p04_slot_shares'] = [w / sum(pw) for w in pw]
            if nonneutral_weights(pw, members, lens):
                write_weighted_region(p04, previous, uploads, members, pw, region['cells'])
            cells = active_cells(self.method, region, self.bn_keys)
            weights = source_weights(self.method, members, self.slots, lens, self.boundaries, self.population, self.data_mass)
            if cells and nonneutral_weights(weights, members, lens):
                write_weighted_region(result, previous, uploads, members, weights, cells)
                record.update(reason='source_reweight', applied_slot_shares=[w / sum(weights) for w in weights],
                    weighted_parameter_numel=sum(c['numel'] for c in cells))
            else:
                record['reason'] = 'outside_scope' if not cells else 'native_equivalent'
        native_norm = update_norm(native, previous)
        virtual_norm = update_norm(result, previous)
        scale, target = 1., None
        if not warmup and self.method in ('a01', 'a02', 'a03', 'a04'):
            native_direction = self.method in ('a01', 'a03')
            source, target = (native_norm, virtual_norm) if native_direction else (virtual_norm, native_norm)
            if source == 0:
                if target != 0:
                    raise FloatingPointError('Cannot match nonzero norm from zero direction')
            else:
                scale = target / source
            if native_direction:
                result = {k: v.clone() for k, v in native.items()}
                for record in details:
                    if 'native_slot_shares' in record:
                        record['applied_slot_shares'] = list(record['native_slot_shares'])
                        record['weighted_parameter_numel'] = 0
            if scale != 1:
                for region, record in zip(self.regions, details):
                    if record['slots']:
                        for cell in region['cells']:
                            k, sl = cell['key'], slices(cell)
                            result[k][sl] = previous[k][sl] + scale * (result[k][sl] - previous[k][sl])
        if not warmup and self.method == 'a18':
            for region, record in zip(self.regions, details):
                if record['slots'] and record['weighted_parameter_numel']:
                    record['applied_slot_shares'] = [.5 * a + .5 * b for a, b in zip(
                        record['native_slot_shares'], record['virtual_p04_slot_shares'])]
                    for cell in region['cells']:
                        k, sl = cell['key'], slices(cell)
                        result[k][sl] = .5 * native[k][sl] + .5 * p04[k][sl]
        for region, record in zip(self.regions, details):
            record['actual_changed_coordinates'] = sum(int(result[c['key']][slices(c)].ne(native[c['key']][slices(c)]).sum()) for c in region['cells'])
        actual_norm = update_norm(result, previous)
        self.rounds_seen += 1
        self.round_diagnostics = dict(mode=self.method, round=self.rounds_seen, first_round_native=warmup,
            regions=details, population=list(self.population), boundaries=list(self.boundaries),
            actual_update_l2=actual_norm, native_update_l2=native_norm, virtual_p04_update_l2=update_norm(p04, previous),
            virtual_candidate_update_l2=virtual_norm, global_scale=scale, matched_norm_target=target,
            matched_norm_relative_error=abs(actual_norm - target) / target if target else None,
            matched_norm_absolute_error=abs(actual_norm - target) if target is not None else None,
            actual_update_l2_ratio_native=actual_norm / native_norm if native_norm else None,
            difference_from_native_l2=update_norm(result, native), difference_from_p04_l2=update_norm(result, p04),
            actual_update_cosine_native=norm_cosine(result, native, previous),
            actual_update_cosine_p04=norm_cosine(result, p04, previous),
            changed_from_native_fraction=sum(r['actual_changed_coordinates'] for r in details) / sum(r['numel'] for r in self.regions),
            source_share_scope='Applied shares describe reweighted cells only when weighted_parameter_numel>0; other cells Native; A01/A03 Native source direction with global scale',
            direction='read-only displacement diagnostics, never a score')
        return result
