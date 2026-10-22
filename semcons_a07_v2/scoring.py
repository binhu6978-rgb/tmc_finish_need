"""One source-level score; embedding, CE utility, test scores and S1 are not inputs."""
import math


def complementarity(current,anchors,history,reliability,round_number,active_tiers,cfg):
    tiers=sorted(set(active_tiers));details={};scores={}
    if len(tiers)<2:
        return dict(valid=False,reason='one_current_tier',scores={},classes=[],details=[])
    candidates={g:{} for g in tiers}
    for (g,c),value in sorted(current.items()):
        if (g not in candidates or value['support']<cfg['minimum_anchor_clients'] or
            any(not math.isfinite(v) or not 0<=v<=1 for v in (value['pre'],value['post']))):
            continue
        own=anchors.get((g,c))
        if own is None or not 0<round_number-own['round']<=cfg['anchor_max_age']:
            continue
        own_rho,own_info=reliability.coefficient(history,(g,c),round_number)
        others=[(h,a) for (h,y),a in anchors.items() if h!=g and y==c
                and a['support']>=cfg['minimum_anchor_clients'] and 0<round_number-a['round']<=cfg['anchor_max_age']
                and math.isfinite(a['q']) and 0<=a['q']<=1]
        if own_rho is None or not others:
            continue
        # Do not replace an unreliable strongest known comparator with an easier one.
        other,reference=max(others,key=lambda x:(x[1]['q'],-x[0]))
        other_rho,other_info=reliability.coefficient(history,(other,c),round_number)
        if other_rho is None:
            continue
        rho=min(own_rho,other_rho,value['support']/(value['support']+1))
        if rho<=0:
            continue
        d=max(0.,value['post']-value['pre']);a=max(0.,value['post']-reference['q'])
        u=rho*min(d,a)
        if not all(math.isfinite(v) and 0<=v<=1 for v in (rho,d,a,u)):
            continue
        candidates[g][c]=u
        details[g,c]=dict(tier=g,category=c,q_pre=value['pre'],q_post=value['post'],
            support=value['support'],physical_clients=value['clients'],reference_tier=other,
            reference_q=reference['q'],reference_round=reference['round'],rho=rho,d=d,a=a,u=u,
            own_reliability=own_info,reference_reliability=other_info)
    common=sorted(set.intersection(*(set(candidates[g]) for g in tiers)))
    if len(common)<cfg['minimum_common_classes']:
        return dict(valid=False,reason='no_fair_common_category_support',scores={},classes=common,
            valid_class_counts={g:len(v) for g,v in candidates.items()},details=list(details.values()))
    scores={g:sum(candidates[g][c] for c in common)/len(common) for g in tiers}
    return dict(valid=True,reason='task_grounded_potential_complementarity',scores=scores,classes=common,
                details=[details[g,c] for g in tiers for c in common])
