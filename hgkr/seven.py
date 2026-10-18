"""Seven-profile C0--C5 protocol, isolated from the validated B and A arms."""
import contextlib
import copy
import io
import itertools
import math
import random
import time

import torch

from hgkr.core import (anchored_aggregate, choose_kernels, clone_state, keyed_seed, reset_copy)
from hgkr.protocol import allocation_cost, capacity
from models.Fed import get_model_list, split_model
from models.Update import LocalUpdate_AdaptiveFL
from semconsfl.aggregation import prefix, update_diagnostics

PROFILES = ((.4,2),(.4,3),(.4,4),(.66,2),(.66,3),(.66,4),(1.,2))
MODES = ('C0','C1','C2','C3','C4','C5')


def initialize_models(args):
    if args.width_ration != [.4,.66,1.] or args.depth_saved != [2,3,4]:
        raise ValueError('Expected original seven profiles')
    with contextlib.redirect_stdout(io.StringIO()):
        models, info = get_model_list(args)
    if tuple(p[:2] for p in info) != PROFILES:
        raise ValueError('Actual original model order differs')
    return models, info


def shared_specs(models):
    """Compute the intersection, not a three-tier mask or assumed profile prefix."""
    states = [m.state_dict() for m in models]
    full = states[-1]
    if len(states) != 7 or any(s.keys()!=full.keys() for s in states):
        raise ValueError('Invalid seven-profile states')
    for s in states:
        for key,value in s.items(): prefix(full[key],value,key)
    result = {}
    for key,value in full.items():
        if value.ndim == 4 and key.endswith('.weight'):
            shape = tuple(min(s[key].shape[d] for s in states) for d in range(4))
            if any(tuple(s[key].shape[2:])!=shape[2:] for s in states):
                raise ValueError('Incomparable spatial kernel shapes')
            result[key] = shape
    return result


def allowed_profiles(client):
    return tuple(range((2,5,6)[capacity(client)]+1))


def make_schedule(seed, rounds, users, costs, config):
    """Small, bounded client-first allocation; no test/label dependent decisions."""
    participants = random.Random(keyed_seed(seed,'participation'))
    rng = random.Random(keyed_seed(seed,'seven-feasible-allocation'))
    exposure = [0]*100
    result = []
    for round_id in range(rounds):
        clients = participants.sample(range(100),10)
        allowed = [allowed_profiles(c) for c in clients]
        minimum = allocation_cost(clients,[0]*10,users,costs)
        maximum_mac = sum(5*(len(users[c])//50)*50*max(costs[p]['forward_mac'] for p in a) for c,a in zip(clients,allowed))
        maximum_byte = sum(8*max(costs[p]['parameters'] for p in a) for a in allowed)
        ceiling = (max(minimum[0],config.budget_fraction*maximum_mac),
                   max(minimum[1],config.budget_fraction*maximum_byte))
        def feasible(a):
            x = allocation_cost(clients,a,users,costs)
            return x[0]<=ceiling[0]+1 and x[1]<=ceiling[1]+1
        # Only enumerate subsets of Full-capable participants (at most 2**10).
        high = [i for i,c in enumerate(clients) if capacity(c)==2]
        candidates = []
        for flags in itertools.product((False,True),repeat=len(high)):
            a = [0]*10
            for i,flag in zip(high,flags):
                if flag: a[i]=6
            if feasible(a): candidates.append(a)
        def priority(a):
            return (a.count(6), -sum(exposure[c] for c,p in zip(clients,a) if p==6),
                    -allocation_cost(clients,a,users,costs)[0], tuple(a))
        chosen = max(candidates,key=priority)
        order = list(range(10)); rng.shuffle(order)
        # Fill spare budget using actual MAC, not profile index or width alone.
        for i in order:
            if chosen[i]==6: continue
            trials = []
            for p in allowed[i]:
                a = chosen.copy(); a[i]=p
                if feasible(a): trials.append(a)
            chosen = max(trials,key=lambda a:(allocation_cost(clients,a,users,costs)[0],
                                              allocation_cost(clients,a,users,costs)[1],tuple(a)))
        target = allocation_cost(clients,chosen,users,costs)
        pool = set()
        # Fixed bounded feasible sampling, not training-seed or hyperparameter search.
        for _ in range(128):
            a = [0]*10; order=list(range(10)); rng.shuffle(order)
            for i in order:
                options = []
                for p in allowed[i]:
                    b=a.copy(); b[i]=p
                    if feasible(b): options.append(p)
                a[i] = rng.choice(options)
            actual=allocation_cost(clients,a,users,costs)
            if all(abs(x-y)<=config.budget_match_tolerance*y+1 for x,y in zip(actual,target)):
                pool.add(tuple(a))
        # If no sampled alternative matches, use the priority allocation transparently.
        alternatives = sorted(pool)
        baseline = list(rng.choice(alternatives)) if alternatives else chosen.copy()
        for c,p in zip(clients,chosen): exposure[c] += p==6
        result.append({'round':round_id,'clients':clients,'capability':[capacity(c) for c in clients],
                       'random':baseline,'priority':chosen,'random_cost':list(allocation_cost(clients,baseline,users,costs)),
                       'priority_cost':list(target),'ceiling':list(ceiling),
                       'matched_random_candidates':len(alternatives),'budget_match_fallback':not alternatives,
                       'allocation_identical':baseline==chosen})
    return result


def choose(scores, config, mode, seed, round_id, slot):
    if mode not in MODES: raise ValueError('Unknown C arm')
    # Shared modes use exactly the same seed namespace and position for tie breaking.
    selection_slot = slot if mode=='C2' else 0
    base = 'B1' if mode in ('C0','C1') else ('B3' if mode in ('C4','C5') else 'B2')
    if mode=='C5':
        low = {k:dict(v,score=-v['score']) for k,v in scores.items()}
        chosen,diagnostics = choose_kernels(low,config,base,seed,round_id,selection_slot)
        for key,d in diagnostics.items():
            ids=chosen[key]
            d['selected_instability_mean']=sum(float(scores[key]['score'][i]) for i in ids)/len(ids) if ids else None
            d['all_instability_mean']=float(scores[key]['score'].mean())
    else:
        chosen,diagnostics=choose_kernels(scores,config,base,seed,round_id,selection_slot)
    return chosen,diagnostics


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def run_round(models, dataset, users, entry, args, config, history, mode, masks):
    r=entry['round']; profiles=entry['random'] if mode=='C0' else entry['priority']
    if not entry['clients'] or len(profiles)!=len(entry['clients']): raise ValueError('Invalid dispatch')
    if any(c not in users or p not in allowed_profiles(c) for c,p in zip(entry['clients'],profiles)):
        raise ValueError('Device capability/participation violation')
    old=clone_state(models[-1].state_dict())
    begin=time.perf_counter(); scores=history.scores(r); score_seconds=time.perf_counter()-begin
    uploads,starts,counts,logs=[],[],[],[]
    train_seconds=reset_seconds=0.
    for slot,(client,profile) in enumerate(zip(entry['clients'],profiles)):
        local=LocalUpdate_AdaptiveFL(args=args,dataset=dataset,idxs=users[client])
        model=copy.deepcopy(models[profile]).to(args.device)
        dispatched=clone_state(model.state_dict())
        sync(args.device); begin=time.perf_counter()
        selection,diag=choose(scores,config,mode,args.seed,r,slot)
        reset_l2=reset_copy(dispatched,old,history.specs,selection,args.seed,r,slot)
        model.load_state_dict(dispatched)
        sync(args.device); reset_seconds+=time.perf_counter()-begin
        start=clone_state(model.state_dict()); begin=time.perf_counter()
        weights=local.train(round=r,net=model)
        sync(args.device); train_seconds+=time.perf_counter()-begin
        upload=clone_state(weights)
        learning_l2=sum(float((upload[k]-start[k]).double().square().sum()) for k in upload)**.5
        uploads.append(upload); starts.append(start); counts.append(len(users[client]))
        logs.append({'slot':slot,'client':client,'profile':profile,'n':len(users[client]),
                     'sgd_steps':5*(len(users[client])//50),'reset_l2':reset_l2,
                     'learning_delta_l2':learning_l2,'reset_count':sum(map(len,selection.values())),'kernels':diag})
        del model,dispatched
    if any(not torch.equal(v,models[-1].state_dict()[k]) for k,v in old.items()):
        raise ValueError('Reset polluted server Full')
    sync(args.device); begin=time.perf_counter()
    new=anchored_aggregate(old,uploads,starts,counts,r,any(x['reset_l2']>0 for x in logs))
    if any(not torch.isfinite(x).all() for x in new.values()): raise ValueError('Nonfinite Full')
    regions=update_diagnostics(old,new,new,masks,7)
    history.update(r,old,new)
    for m in models: m.load_state_dict(split_model(new,m.state_dict()))
    sync(args.device); server_seconds=time.perf_counter()-begin
    events={}
    for log in logs:
        for k,d in log['kernels'].items():
            for i in d['indices']: events[(k,i)]=events.get((k,i),0)+1
    if mode in ('C3','C4','C5') and any(n!=len(logs) for n in events.values()):
        raise ValueError('Shared reset positions diverged across clients')
    return {'round':r,'mode':mode,'slots':logs,'profile_slots':[profiles.count(p) for p in range(7)],
            'reset_total':sum(x['reset_count'] for x in logs),'reset_unique_segments':len(events),
            'max_reset_multiplicity':max(events.values(),default=0),'allocation':entry,
            'coverage_update_diagnostics':regions,'timings':{'local_train_seconds':train_seconds,
             'selection_reset_seconds':reset_seconds,'history_score_seconds':score_seconds,'server_seconds':server_seconds}}
