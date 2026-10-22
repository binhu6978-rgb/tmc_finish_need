"""No PT, no test-set access: paired regional recipient train-probe CE."""
import copy
import hashlib
import json
import math
import pickle
from collections import defaultdict
import torch
from torch.nn import functional as F
from adapterfl_methods import isolated_evaluation_rng, rng_state
from semcons_scope import slices
from .config import seed_of
from .prototypes import scores


def model_digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_digest():
    # Pickling a Tensor includes storage identity; hash numerical streams instead.
    state = rng_state()
    h = hashlib.sha256()
    for k, value in sorted(state.items()):
        if isinstance(value, torch.Tensor):
            h.update(k.encode()); h.update(value.cpu().numpy().tobytes())
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
            for v in value:
                h.update(v.cpu().numpy().tobytes())
        else:
            h.update(k.encode()); h.update(pickle.dumps(value, protocol=4))
    return h.hexdigest()


@torch.no_grad()
def ce_vector(model, batches):
    model.eval(); result = []; labels = []
    device = next(model.parameters()).device
    for x, y in batches:
        loss = F.cross_entropy(model(x.to(device))['output'], y.to(device), reduction='none')
        result.extend(loss.detach().cpu().double().tolist()); labels.extend(y.tolist())
    return result, labels


def flat_delta(upload, dispatch, region):
    return torch.cat([(upload[c['key']][slices(c)] - dispatch[c['key']][slices(c)]).detach().flatten().cpu()
                      for c in region['cells']])


def cosine(a, b):
    a = a.double(); b = b.double()
    denominator = float(a.norm()*b.norm())
    return max(-1., min(1., float(a.dot(b))/denominator)) if denominator > 1e-12 else None


@torch.no_grad()
def utility_records(cfg, models, regions, slots, lens, uploads, dispatched, observations, measurement, anchors, round_number):
    """Pair selection uses only identities/coverage and a frozen hash, never outcomes."""
    rows = []
    stage = next((s for s, rounds in cfg['diagnostic_rounds'].items() if round_number in rounds), None)
    if stage is None:
        return rows
    with isolated_evaluation_rng():
        for region in regions:
            members = [i for i, (_, p) in enumerate(slots) if p in region['profiles']]
            clients = sorted({slots[i][0] for i in members})
            if len(clients) < 2:
                continue
            # One deterministic slot representative per physical client for pairs.
            representatives = {cid: min([i for i in members if slots[i][0] == cid], key=lambda i: seed_of('representative', round_number, region['id'], i)) for cid in clients}
            vectors = {i: flat_delta(uploads[i], dispatched[i], region) for i in members}
            physical = {cid: torch.stack([vectors[i] for i in members if slots[i][0] == cid]).mean(0) for cid in clients}
            senders = sorted(clients, key=lambda c: seed_of('sender', round_number, region['id'], c))[:cfg['senders_per_region']]
            for cid in senders:
                si = representatives[cid]
                reference = torch.stack([v for c, v in physical.items() if c != cid]).mean(0)
                direction = cosine(vectors[si], reference)
                alpha = lens[si] / sum(lens[i] for i in members)
                receivers = sorted([c for c in clients if c != cid], key=lambda c: seed_of('receiver', round_number, region['id'], cid, c))[:cfg['receivers_per_sender_region']]
                for receiver in receivers:
                    ri = representatives[receiver]; profile = slots[ri][1]
                    batches = measurement.batches(receiver, utility=True)
                    if not batches or observations[si] is None:
                        continue
                    model = copy.deepcopy(models[profile]).eval()
                    model.load_state_dict(dispatched[ri])
                    original, labels = ce_vector(model, batches)
                    changed = model.state_dict()
                    for cell in region['cells']:
                        k, sl = cell['key'], slices(cell)
                        changed[k][sl].add_(alpha * (uploads[si][k][sl] - dispatched[si][k][sl]))
                    after, after_labels = ce_vector(model, batches)
                    if labels != after_labels:
                        raise AssertionError('Receiver BN batch/labels changed')
                    classes = sorted(set(labels)); per_class = []
                    for y in classes:
                        indices = [i for i, label in enumerate(labels) if label == y]
                        pre = sum(original[i] for i in indices) / len(indices)
                        post = sum(after[i] for i in indices) / len(indices)
                        per_class.append(dict(label=y, count=len(indices), before_ce=pre, after_ce=post, delta_ce=post-pre))
                    obs = observations[si]
                    common = dict(pair_id=f'{round_number}:{region["id"]}:{si}:{ri}', round=round_number, stage=stage,
                        sender=cid, receiver=receiver, sender_slot=si, receiver_slot=ri,
                        sender_profile=slots[si][1], receiver_profile=profile,
                        sender_tier=sum(cid >= b for b in (40, 70)), receiver_tier=sum(receiver >= b for b in (40, 70)),
                        region=region['id'], region_numel=region['numel'], support_profiles=region['profiles'],
                        region_physical_sources=len(clients), alpha=alpha, sender_sample_mass=lens[si],
                        raw_update_l2=float(vectors[si].double().norm()), scaled_update_l2=alpha*float(vectors[si].double().norm()),
                        direction_cosine=direction, negative_loo=direction is not None and direction < 0,
                        before_ce=sum(original)/len(original), delta_ce=(sum(after)-sum(original))/len(original),
                        own_gain=obs['own_gain'], receiver_probe_examples=len(labels), per_class=per_class,
                        reliability=obs['reliability'], receiver_forward_examples=2*len(labels),
                        split='development' if stage=='early' and cid<70 and receiver<70 else
                              'confirmation' if stage in ('middle','late') and cid>=70 and receiver>=70 else 'descriptive')
                    for seed, head in obs['heads'].items():
                        evidence = scores(head['before'], head['after'], obs['task_before'], obs['task_after'],
                            anchors.get(seed, {}), anchors.get('task', {}), round_number, cfg['anchor_max_age'], classes)
                        rows.append(common | dict(head_seed=seed, evidence=evidence,
                            instrument_probability_change=head['instrument_probability_change']))
                    del model
    return rows


def append_json(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
