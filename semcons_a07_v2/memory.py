"""Add tier/category task references without changing the v1 evidence or scores."""
import copy
import math
import statistics
from collections import defaultdict
from semcons_a07.semantic_memory import SemanticMemory
from semcons_a07.reliability_memory import ReliabilityMemory


def tier_of(cid, boundaries):
    return sum(cid >= b for b in boundaries)


def task_rows(observations, boundaries, minimum):
    """Average repeated slots/profiles within physical identity; no fabricated zeros."""
    clients = defaultdict(list)
    for obs in observations:
        if obs is None:
            continue
        cid, profile = int(obs['client']), int(obs['profile'])
        pre, post = obs['task_before'], obs['task_after']
        for y in sorted(pre.keys() & post.keys()):
            if min(pre[y]['count'],post[y]['count']) < minimum:
                continue
            a, b = float(pre[y]['probability'][y]), float(post[y]['probability'][y])
            if not all(math.isfinite(v) and 0<=v<=1 for v in (a,b)):
                continue
            clients[tier_of(cid,boundaries),int(y),cid].append((a,b,profile))
    groups=defaultdict(list)
    for (g,y,cid),values in clients.items():
        groups[g,y].append(dict(client=cid, pre=statistics.mean(v[0] for v in values),
            post=statistics.mean(v[1] for v in values), profiles=sorted({v[2] for v in values}), slots=len(values)))
    return {key:dict(pre=statistics.median(v['pre'] for v in values),post=statistics.median(v['post'] for v in values),
                    support=len(values),clients=sorted(v['client'] for v in values),sources=values)
            for key,values in groups.items()}


class TierSemanticMemory(SemanticMemory):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.tiers={}

    def tier_snapshot(self):
        return copy.deepcopy(self.tiers)

    def commit_tiers(self,round_number,current):
        updates={}
        if round_number==1:
            return updates
        for key,value in current.items():
            if (value['support']<self.cfg['minimum_anchor_clients'] or
                not math.isfinite(value['post']) or not 0<=value['post']<=1):
                continue
            old=self.tiers.get(key)
            if old is not None and old['round']>=round_number:
                raise ValueError('Tier semantic memory must commit strictly increasing rounds')
            valid=old is not None and 0<round_number-old['round']<=self.cfg['anchor_max_age']
            q=self.cfg['anchor_ema']*old['q']+(1-self.cfg['anchor_ema'])*value['post'] if valid else value['post']
            record=dict(q=q,observed_q=value['post'],round=round_number,support=value['support'],
                clients=list(value['clients']),change=abs(value['post']-old['observed_q']) if valid else None)
            self.tiers[key]=record;updates[key]=copy.deepcopy(record)
        return updates


class TierReliabilityMemory(ReliabilityMemory):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.tiers=defaultdict(list)

    def tier_snapshot(self):
        return copy.deepcopy(dict(self.tiers))

    def commit_tiers(self,round_number,updates):
        if round_number==1:
            return
        for key,value in updates.items():
            # Exactly one record per tier/category/round, not per duplicated slot.
            if self.tiers[key] and self.tiers[key][-1]['round']>=round_number:
                raise ValueError('Tier reliability cannot commit a round twice')
            self.tiers[key].append(copy.deepcopy(value))
        for key in self.tiers:
            self.tiers[key]=[r for r in self.tiers[key] if round_number-r['round']<=self.cfg['history_max_age']]

    def coefficient(self,snapshot,key,round_number):
        history=[r for r in snapshot.get(key,[]) if 0<round_number-r['round']<=self.cfg['history_max_age']]
        if len(history)<self.cfg['history_min_rounds']:
            return None,dict(reason='insufficient_past_history',history_rounds=len(history))
        recent=history[-self.cfg['history_min_rounds']:]
        support=min(r['support'] for r in recent)
        changes=[r['change'] for r in recent if r['change'] is not None]
        if support<self.cfg['minimum_anchor_clients'] or not changes or any(not math.isfinite(v) or not 0<=v<=1 for v in changes):
            return None,dict(reason='insufficient_independent_support_or_change')
        age=round_number-history[-1]['round']
        stability=max(0.,1-statistics.mean(changes))
        freshness=max(0.,1-age/self.cfg['history_max_age'])
        rho=(support/(support+1))*stability*freshness
        if not math.isfinite(rho):
            return None,dict(reason='nonfinite_history')
        return rho,dict(reason='past_support_age_stability',history_rounds=len(history),support=support,
            age=age,stability=stability,freshness=freshness,rho=rho)
