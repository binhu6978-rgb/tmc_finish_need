"""Independent evaluation protocol, not a new aggregation or semantic score."""
from collections import defaultdict
import math
import torch
from semcons_a07.config import seed_of
from semcons_a07.measurement import materialize
from .iv_e_diagnostics import Probes,MechanismDiagnostic,ce_summary
from .memory import tier_of


def split_users(dataset,users):
    training={};evaluation={}
    for cid,indices in sorted(users.items()):
        classes=defaultdict(list)
        for i in indices:classes[int(dataset.targets[i])].append(int(i))
        held=[]
        for y,values in sorted(classes.items()):
            if len(values)<4:continue
            n=min(len(values)-2,max(2,len(values)//5))
            held.extend(sorted(values,key=lambda i:seed_of('expanded-holdout',cid,y,i))[:n])
        evaluation[cid]=sorted(held);training[cid]=sorted(set(indices)-set(held))
        assert training[cid] and not set(training[cid])&set(held)
    train={i for values in training.values() for i in values};valid={i for values in evaluation.values() for i in values}
    assert not train&valid and train|valid=={i for values in users.values() for i in values}
    return training,evaluation


def cohort(cid):return 'confirmation' if seed_of('expanded-confirmation',int(cid))%2 else 'development'


class HeldoutProbes(Probes):
    def __init__(self,dataset,training,evaluation):
        super().__init__(dataset,training);self.evaluation=evaluation

    def batches(self,cid):
        if cid not in self.cache:
            groups=defaultdict(list)
            for i in self.evaluation[cid]:groups[int(self.dataset.targets[i])].append(i)
            plan=[i for y,values in sorted(groups.items()) if len(values)>=2
                  for i in sorted(values,key=lambda i:seed_of('expanded-probe',cid,y,i))[:2]]
            assert not set(plan)&set(self.users[cid])
            self.cache[cid]=materialize(self.dataset,plan,seed_of('expanded-views',cid),50)
        return self.cache[cid]


class HeldoutDiagnostic(MechanismDiagnostic):
    def __init__(self,dataset,users,evaluation,path):
        super().__init__(dataset,users,path);self.probes=HeldoutProbes(dataset,users,evaluation)

    def run(self,*args,**kwargs):
        self.own_cache={};return super().run(*args,**kwargs)

    def pairs(self,uploads,lens,hook,decision):
        rows,inventory=super().pairs(uploads,lens,hook,decision)
        for row in rows:
            row['cohort']=cohort(row['receiver'])
            outside=[c for c in row['per_class'] if c['category'] not in decision['classes']]
            inside=[c for c in row['per_class'] if c['category'] in decision['classes']]
            row['common_sample_fraction']=sum(c['count'] for c in inside)/row['probe_examples']
            row['noncommon_utility']=sum(c['utility']*c['count'] for c in outside)/sum(c['count'] for c in outside) if outside else None
            gains=[]
            for i in row['source_slots']:
                if i not in self.own_cache:
                    cid,p=hook.slots[i];batches=self.probes.batches(cid)
                    if batches:
                        net=self.model(p,self.observer.dispatches[i]);before,labels=self.ce(net,batches)
                        net.load_state_dict(uploads[i]);after,labels2=self.ce(net,batches);del net
                        assert labels==labels2
                        self.own_cache[i]=(sum(before)-sum(after))/len(labels)
                    else:self.own_cache[i]=None
                gains.append(self.own_cache[i])
            row['source_own_training_gain']=row['source_own_gain']
            row['source_own_holdout_gain']=sum(g*lens[i]/row['source_mass'] for g,i in zip(gains,row['source_slots'])) if all(g is not None for g in gains) else None
        return rows,inventory

    def aggregate_utility(self,variants,previous,hook,decision):
        rows=[];r=self.observer.round_number;participating={c for c,p in hook.slots}
        for g in range(3):
            for part in ('development','confirmation'):
                candidates=sorted((int(c) for c in self.probes.users if tier_of(int(c),hook.boundaries)==g
                    and int(c) not in participating and cohort(c)==part),key=lambda c:seed_of('expanded-full-panel',g,part,c))
                cid=next((c for c in candidates if self.probes.batches(c)),None)
                if cid is None:raise ValueError('No heldout Full recipient in predefined tier/cohort')
                batches=self.probes.batches(cid);net=self.model(6,previous);before,labels=self.ce(net,batches)
                for method,value in variants.items():
                    net.load_state_dict(value);after,labels2=self.ce(net,batches);assert labels==labels2
                    rows.append(dict(round=r,method=method,receiver=cid,receiver_tier=g,receiver_profile=6,
                        cohort=part,semantic_valid=decision['valid'],**ce_summary(before,after,labels,set(decision['classes']))))
                del net
        baseline={row['receiver']:row['after_ce'] for row in rows if row['method']=='A07'}
        for row in rows:row['utility_vs_a07']=baseline[row['receiver']]-row['after_ce']
        return rows
