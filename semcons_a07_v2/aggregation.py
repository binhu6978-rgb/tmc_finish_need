"""Call original A07 first; rewrite only eligible Feature cells with original tool."""
from collections import defaultdict
import math
import torch
from capability_final40 import source_weights
from native_group_aggregation import write_weighted_region,update_norm
from semcons_scope import slices
from .memory import tier_of


def blend_shares(pi,scores,strength):
    if not math.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('semantic_lambda must be finite in [0,1]')
    if any(not math.isfinite(v) or v<0 for v in pi.values()) or not math.isclose(sum(pi.values()),1.,abs_tol=1e-12):
        raise ValueError('Invalid A07 tier shares')
    if strength==0 or len(pi)<2:
        return dict(pi)
    if any(g not in scores or not math.isfinite(scores[g]) or not 0<=scores[g]<=1 for g in pi):
        return dict(pi)
    if len({scores[g] for g in pi})==1:
        return dict(pi)
    denominator=sum(p*(1+scores[g]) for g,p in pi.items())
    return {g:(1-strength)*p+strength*p*(1+scores[g])/denominator for g,p in pi.items()}


@torch.no_grad()
def fuse(base,previous,uploads,lens,hook,decision,strength,first_round):
    # base is returned directly for all neutral conditions, preserving float order.
    report=dict(lambda_value=strength,first_round_native=first_round,decision=decision,
                regions=[],changed_parameters=0,difference_from_a07_l2=0.,adapter_in_decision=False)
    if first_round or strength==0 or not decision['valid']:
        report['reason']='first_round_native' if first_round else 'lambda_zero' if strength==0 else decision['reason']
        return base,report
    result=base
    for region in hook.regions:
        members=[i for i,(_,p) in enumerate(hook.slots) if p in region['profiles']]
        groups=defaultdict(list)
        for i in members:groups[tier_of(hook.slots[i][0],hook.boundaries)].append(i)
        cells=[c for c in region['cells'] if c['key'] not in hook.bn_keys and not c['key'].startswith('classifier.')]
        detail=dict(region=region['id'],tiers=sorted(groups),slots=members,reason='single_tier_or_outside_scope')
        report['regions'].append(detail)
        if len(groups)<2 or len(region['profiles'])<2 or not cells:
            continue
        weights=source_weights('a07',members,hook.slots,lens,hook.boundaries,hook.population,hook.data_mass)
        total=sum(weights);by_slot=dict(zip(members,weights))
        pi={g:sum(by_slot[i] for i in ids)/total for g,ids in groups.items()}
        new=blend_shares(pi,decision['scores'],strength)
        detail.update(pi_a07=pi,pi_new=new,feature_numel=sum(c['numel'] for c in cells))
        if new==pi:
            detail['reason']='equal_or_missing_scores';continue
        new_weights=[w*new[tier_of(hook.slots[i][0],hook.boundaries)]/pi[tier_of(hook.slots[i][0],hook.boundaries)] for i,w in zip(members,weights)]
        if result is base:result={k:v.clone() for k,v in base.items()}
        write_weighted_region(result,previous,uploads,members,new_weights,cells)
        changed=sum(int(result[c['key']][slices(c)].ne(base[c['key']][slices(c)]).sum()) for c in cells)
        detail.update(reason='semantic_source_blend',a07_slot_shares=[w/total for w in weights],
                      new_slot_shares=[w/sum(new_weights) for w in new_weights],changed_parameters=changed)
        report['changed_parameters']+=changed
    report['difference_from_a07_l2']=update_norm(result,base) if result is not base else 0.
    report['reason']='source_blend' if report['changed_parameters'] else 'no_effective_change'
    return result,report
