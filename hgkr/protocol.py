"""Frozen client-first participation and transparent compute/communication budgets."""
import contextlib
import io
import itertools
import random

import torch
from torch import nn

from hgkr.core import keyed_seed
from models.Fed import get_model_list
from semconsfl.measurement import isolated_rng


def initialize_models(args):
    if args.width_ration != [.4, .66, 1.] or args.depth_saved != [2]:
        raise ValueError('Exactly three fixed width profiles, all slim_idx=2')
    with contextlib.redirect_stdout(io.StringIO()):
        models, profiles = get_model_list(args)
    if len(models) != 3:
        raise ValueError('Expected Small/Medium/Full')
    return models, profiles


def capacity(client):
    if not 0 <= client < 100:
        raise ValueError('Invalid physical client')
    return 0 if client < 40 else (1 if client < 70 else 2)


def model_costs(models):
    """Conv/linear multiply-accumulates per image. Training FLOP proxy=6*MAC*images."""
    costs = []
    with isolated_rng(), torch.no_grad():
        for model in models:
            modes = [(m, m.training) for m in model.modules()]
            counter = [0]
            def count(module, inputs, output):
                if isinstance(module, nn.Conv2d):
                    counter[0] += output[0].numel() * (module.in_channels // module.groups) * module.kernel_size[0]*module.kernel_size[1]
                else:
                    counter[0] += module.in_features * module.out_features
            handles = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            try:
                model.eval()
                model(torch.zeros((2, 3, 32, 32), device=next(model.parameters()).device))
            finally:
                for h in handles:
                    h.remove()
                for m, mode in modes:
                    m.training = mode
            costs.append({'parameters': sum(p.numel() for p in model.parameters()), 'forward_mac': counter[0]})
    return costs


def allocation_cost(clients, profiles, users, costs):
    # Original local_ep=5, bs=50, drop_last=True. Entire n still weights aggregation.
    mac = sum(5*(len(users[c])//50)*50*costs[p]['forward_mac'] for c,p in zip(clients, profiles))
    byte = sum(8*costs[p]['parameters'] for p in profiles)  # one download + upload, float32
    return mac, byte


def make_schedule(seed, rounds, users, costs, config):
    participation = random.Random(keyed_seed(seed, 'participation'))
    random_assignment = random.Random(keyed_seed(seed, 'budget-matched-random'))
    # Counts reflect past Full assignments only; lower exposure wins a priority tie.
    exposure = [0]*100
    schedule = []
    for r in range(rounds):
        clients = participation.sample(range(100), 10)
        caps = [capacity(c) for c in clients]
        minimum = allocation_cost(clients, [0]*10, users, costs)
        maximum = allocation_cost(clients, caps, users, costs)
        ceiling = tuple(max(lo, config.budget_fraction*hi) for lo,hi in zip(minimum,maximum))
        feasible = []
        for assignment in itertools.product(*(range(c+1) for c in caps)):
            mac, byte = allocation_cost(clients, assignment, users, costs)
            if mac <= ceiling[0]+1 and byte <= ceiling[1]+1:
                feasible.append((assignment, mac, byte))
        def priority(item):
            a, mac, byte = item
            return (a.count(2), -sum(exposure[c] for c,p in zip(clients,a) if p==2), mac, byte, a)
        winner = max(feasible, key=priority)
        # Random feasible comparator with both actual costs within 2% of B1.
        # B1 itself guarantees a nonempty pool; degenerate allocation contrast is logged.
        candidates = [x for x in feasible if all(abs(x[i+1]-winner[i+1]) <= config.budget_match_tolerance*winner[i+1]+1 for i in (0,1))]
        baseline = random_assignment.choice(candidates)
        for c,p in zip(clients,winner[0]):
            exposure[c] += p==2
        schedule.append({'round':r, 'clients':clients, 'capacity':caps, 'B0':list(baseline[0]),
                         'priority':list(winner[0]), 'B0_cost':list(baseline[1:]),
                         'priority_cost':list(winner[1:]), 'ceiling':list(ceiling),
                         'feasible_count':len(feasible), 'matched_candidate_count':len(candidates),
                         'allocation_identical':baseline[0]==winner[0]})
    return schedule
