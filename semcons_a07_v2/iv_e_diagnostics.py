"""Read-only IV-E counterfactuals. No score fitting, SGD, test-set access or PT."""
import copy
from collections import defaultdict
import math
import time
import torch
from adapterfl_methods import isolated_evaluation_rng
from capability_final40 import source_weights
from native_group_aggregation import update_norm, norm_cosine
from semcons_scope import slices
from semcons_a07.config import seed_of
from semcons_a07.measurement import probe_indices, materialize
from semcons_a07.diagnostics import append_json, ce_vector, cosine, rng_digest
from .memory import tier_of
from .aggregation import fuse


def independent_indices(dataset, indices, cid, classes=16, samples=2):
    excluded=set(probe_indices(dataset,indices,cid,2))
    grouped=defaultdict(list)
    for i in sorted(set(map(int,indices))-excluded):grouped[int(dataset.targets[i])].append(i)
    labels=sorted((y for y,v in grouped.items() if len(v)>=samples),
                  key=lambda y:seed_of('iv-e-classes',cid,y))[:classes]
    plan=[i for y in sorted(labels) for i in sorted(grouped[y],key=lambda i:seed_of('iv-e-index',cid,i))[:samples]]
    assert not set(plan)&excluded
    return plan


class Probes:
    def __init__(self,dataset,users):
        self.dataset=dataset;self.users=users;self.cache={};self.histograms={}

    def batches(self,cid):
        if cid not in self.cache:
            plan=independent_indices(self.dataset,self.users[cid],cid)
            self.cache[cid]=materialize(self.dataset,plan,seed_of('iv-e-views',cid),50)
        return self.cache[cid]

    def histogram(self,cid):
        if cid not in self.histograms:
            counts=torch.bincount(torch.tensor([int(self.dataset.targets[i]) for i in self.users[cid]]),minlength=100).double()
            self.histograms[cid]=counts/counts.sum()
        return self.histograms[cid]


def score_controls(decision):
    """Matched validity, fixed category set. Only remove named numerical terms."""
    if not decision['valid']:return {k:copy.deepcopy(decision) for k in ('no_sm','no_rm','permuted')}
    groups=defaultdict(list)
    for d in decision['details']:groups[d['tier']].append(d)
    outputs={}
    for key,term in [('no_sm',lambda d:d['rho']*d['d']),('no_rm',lambda d:min(d['d'],d['a']))]:
        outputs[key]=dict(decision,scores={g:sum(term(d) for d in values)/len(values) for g,values in groups.items()})
    ids=sorted(decision['scores'])
    outputs['permuted']=dict(decision,scores={g:decision['scores'][ids[(i+1)%len(ids)]] for i,g in enumerate(ids)})
    return outputs


def stage_scores(decision,g):
    values=[d for d in decision.get('details',[]) if d['tier']==g and d['category'] in decision['classes']]
    if not decision['valid'] or not values:return dict(D=None,M=None,U=None,rho=None,permuted_U=None)
    return dict(D=sum(d['d'] for d in values)/len(values),M=sum(min(d['d'],d['a']) for d in values)/len(values),
        U=decision['scores'][g],rho=sum(d['rho'] for d in values)/len(values),
        permuted_U=score_controls(decision)['permuted']['scores'][g])


def ce_summary(before,after,labels,common=None):
    per=[]
    for y in sorted(set(labels)):
        ids=[i for i,v in enumerate(labels) if v==y]
        a=sum(before[i] for i in ids)/len(ids);b=sum(after[i] for i in ids)/len(ids)
        per.append(dict(category=y,count=len(ids),before_ce=a,after_ce=b,utility=a-b))
    selected=[c for c in per if common is not None and c['category'] in common]
    return dict(before_ce=sum(before)/len(before),after_ce=sum(after)/len(after),
        utility=(sum(before)-sum(after))/len(before),
        common_utility=sum(c['utility'] for c in selected)/len(selected) if selected else None,
        common_class_count=len(selected),probe_examples=len(labels),per_class=per)


def js_distance(a,b):
    m=(a+b)/2
    def kl(x):return float((x*(x.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum())
    return .5*(kl(a)+kl(b))


@torch.no_grad()
def l2_control(base,actual,previous,hook):
    a=update_norm(base,previous);b=update_norm(actual,previous)
    if a==0:
        if b!=0:raise FloatingPointError('Cannot norm-match zero A07 direction')
        return base,1.
    scale=b/a
    if scale==1:return base,scale
    result={k:v.clone() for k,v in base.items()}
    for region in hook.regions:
        if any(p in region['profiles'] for _,p in hook.slots):
            for cell in region['cells']:
                k,sl=cell['key'],slices(cell)
                result[k][sl]=previous[k][sl]+scale*(base[k][sl]-previous[k][sl])
    return result,scale


class MechanismDiagnostic:
    def __init__(self,dataset,users,path):
        self.path=path;self.probes=Probes(dataset,users);self.models=None;self.observer=None

    def model(self,profile,state):
        net=copy.deepcopy(self.models[profile]).eval();net.load_state_dict(state);return net

    def ce(self,model,batches):
        values,labels=ce_vector(model,batches);self.forward_examples+=len(labels)
        return values,labels

    @torch.no_grad()
    def run(self,base,actual,previous,uploads,lens,hook,decision,first_round):
        started=time.perf_counter();state=rng_digest();self.forward_examples=0
        with isolated_evaluation_rng():
            figure7,inventory=self.pairs(uploads,lens,hook,decision) if not first_round else ([],[])
            variants={'A07':base,'V2':actual};reports={}
            for key,d in score_controls(decision).items():
                variants[key],reports[key]=fuse(base,previous,uploads,lens,hook,d,self.observer.cfg['semantic_lambda'],first_round)
            variants['A07_L2_matched'],scale=l2_control(base,actual,previous,hook)
            figure8=self.aggregate_utility(variants,previous,hook,decision)
            norms={key:dict(update_l2=update_norm(value,previous),difference_from_a07_l2=update_norm(value,base),
                update_cosine_a07=norm_cosine(value,base,previous)) for key,value in variants.items()}
            report=dict(round=self.observer.round_number,valid_semantics=decision['valid'],scores=decision['scores'],
                common_classes=decision['classes'],regions=self.observer.fusion if self.observer.fusion else None,
                counterfactual_fusion=reports,norms=norms,l2_control_scale=scale,inventory=inventory,
                figure7=figure7,figure8=figure8,seconds=time.perf_counter()-started,
                receiver_forward_examples=self.forward_examples,
                hypothetical_tier_delta_downlink_bytes=sum(r['region_numel']*4 for r in figure7),
                hypothetical_class_feedback_bytes=12*sum(len(r['per_class']) for r in figure7))
        if rng_digest()!=state:raise AssertionError('IV-E diagnostic changed training RNG')
        append_json(self.path/'iv_e_rounds.jsonl',report)
        self.observer.dispatches=[]

    @torch.no_grad()
    def pairs(self,uploads,lens,hook,decision):
        rows=[];inventory=[];slots=hook.slots;sent=self.observer.dispatches
        if len(sent)!=len(uploads):raise AssertionError('Actual per-slot dispatches not captured')
        r=self.observer.round_number;baseline={}
        for region in hook.regions:
            cells=[c for c in region['cells'] if c['key'] not in hook.bn_keys and not c['key'].startswith('classifier.')]
            members=[i for i,(_,p) in enumerate(slots) if p in region['profiles']]
            groups=defaultdict(list)
            for i in members:groups[tier_of(slots[i][0],hook.boundaries)].append(i)
            info=dict(region=region['id'],tiers=sorted(groups),feature_numel=sum(c['numel'] for c in cells),slots=members)
            inventory.append(info)
            if len(groups)<2 or len(region['profiles'])<2 or not cells:
                info['reason']='single_tier_or_outside_scope';continue
            weights=source_weights('a07',members,slots,lens,hook.boundaries,hook.population,hook.data_mass)
            alpha=dict(zip(members,[w/sum(weights) for w in weights]));contributions={};vectors={}
            for g,ids in groups.items():
                contributions[g]=[sum((alpha[i]*(uploads[i][c['key']][slices(c)]-sent[i][c['key']][slices(c)]) for i in ids),
                    torch.zeros_like(uploads[ids[0]][c['key']][slices(c)])) for c in cells]
                vectors[g]=torch.cat([v.flatten().cpu() for v in contributions[g]])
            for g,ids in groups.items():
                mass=sum(lens[i] for i in ids)
                source_hist=sum((self.probes.histogram(slots[i][0])*(lens[i]/mass) for i in ids),torch.zeros(100,dtype=torch.double))
                reference=sum((v for h,v in vectors.items() if h!=g),torch.zeros_like(vectors[g]))
                for h,targets in groups.items():
                    if h==g:continue
                    clients=sorted({slots[i][0] for i in targets},key=lambda c:seed_of('iv-e-recipient',r,region['id'],g,c))
                    cid=clients[0]
                    j=min((i for i in targets if slots[i][0]==cid),key=lambda i:seed_of('iv-e-profile',r,region['id'],g,i))
                    batches=self.probes.batches(cid)
                    if not batches:
                        info.setdefault('missing_receiver_probe',[]).append(cid);continue
                    if j not in baseline:
                        model=self.model(slots[j][1],sent[j]);baseline[j]=self.ce(model,batches)
                        check=self.ce(model,batches)
                        if baseline[j]!=check:raise AssertionError('Repeated fixed-probe CE changed')
                        del model
                    before,labels=baseline[j];model=self.model(slots[j][1],sent[j]);current=model.state_dict()
                    for c,value in zip(cells,contributions[g]):current[c['key']][slices(c)].add_(value.to(current[c['key']].device))
                    after,labels2=self.ce(model,batches);del model
                    assert labels==labels2 and cid not in {slots[i][0] for i in ids}
                    rows.append(dict(round=r,region=region['id'],source_tier=g,receiver=cid,receiver_tier=h,
                        receiver_profile=slots[j][1],receiver_slot=j,source_clients=sorted({slots[i][0] for i in ids}),
                        source_slots=ids,source_profiles=[slots[i][1] for i in ids],source_slot_shares=[alpha[i] for i in ids],
                        source_mass=mass,source_pi=sum(alpha[i] for i in ids),region_numel=sum(c['numel'] for c in cells),
                        source_delta_l2=float(vectors[g].double().norm()),reference_delta_l2=float(reference.double().norm()),
                        direction_cosine=cosine(vectors[g],reference),source_receiver_label_js=js_distance(source_hist,self.probes.histogram(cid)),
                        source_profile_mean=sum(slots[i][1]*lens[i]/mass for i in ids),
                        source_own_gain=sum(self.observer.observations[i]['own_gain']*lens[i]/mass for i in ids),
                        semantic_valid=decision['valid'],**stage_scores(decision,g),
                        **ce_summary(before,after,labels,set(decision['classes']))))
            info['reason']='measured_all_tiers_fixed_cross_tier_receivers'
        return rows,inventory

    @torch.no_grad()
    def aggregate_utility(self,variants,previous,hook,decision):
        rows=[];r=self.observer.round_number;participating={c for c,p in hook.slots}
        for g in range(3):
            candidates=sorted((int(c) for c in self.probes.users if tier_of(int(c),hook.boundaries)==g and int(c) not in participating),
                              key=lambda c:seed_of('iv-e-full-panel',g,c))
            cid=next((c for c in candidates if self.probes.batches(c)),None)
            if cid is None:raise ValueError('No independent Full evaluation recipient in tier')
            batches=self.probes.batches(cid);model=self.model(6,previous);before,labels=self.ce(model,batches)
            for method,value in variants.items():
                model.load_state_dict(value);after,labels2=self.ce(model,batches)
                assert labels==labels2
                rows.append(dict(round=r,method=method,receiver=cid,receiver_tier=g,receiver_profile=6,
                    semantic_valid=decision['valid'],**ce_summary(before,after,labels,set(decision['classes']))))
            del model
        by_receiver={row['receiver']:row['after_ce'] for row in rows if row['method']=='A07'}
        for row in rows:row['utility_vs_a07']=by_receiver[row['receiver']]-row['after_ce']
        return rows
