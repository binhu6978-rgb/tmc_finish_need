"""Prespecified scope/functional-response experiments; original SGD is untouched."""
import copy
import hashlib
import itertools
import json
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from models.Fed import Aggregation_AdaptiveFL
from adapterfl_methods import MethodHooks, prefix, isolated_evaluation_rng

EPS=1e-12
MODES=tuple(f's{i:02d}' for i in range(10))


def seed_of(*items):
    return int.from_bytes(hashlib.sha256(repr(items).encode()).digest()[:8],'little') % (2**32)


def build_regions(models):
    states=[m.state_dict() for m in models];full=states[-1];groups=defaultdict(list)
    for key, tensor in full.items():
        for state in states:
            if state.keys()!=full.keys():raise ValueError('Model keys differ')
            prefix(tensor,state[key])
        dims=min(2,tensor.ndim)
        edges=[sorted({0,*[s[key].shape[d] for s in states]}) for d in range(dims)]
        for bounds in itertools.product(*[list(zip(e,e[1:])) for e in edges]):
            support=tuple(p for p,s in enumerate(states) if all(b[1]<=s[key].shape[d] for d,b in enumerate(bounds)))
            size=math.prod(b-a for a,b in bounds)*math.prod(tensor.shape[dims:])
            groups[support].append({'key':key,'bounds':bounds,'numel':size})
    regions=[{'id':i,'profiles':list(support),'cells':cells,'numel':sum(c['numel'] for c in cells)}
             for i,(support,cells) in enumerate(sorted(groups.items()))]
    if sum(r['numel'] for r in regions)!=sum(t.numel() for t in full.values()):
        raise ValueError('Structural regions do not cover Full exactly')
    return regions


def slices(cell):
    return tuple(slice(a,b) for a,b in cell['bounds'])


def semantic_plan(dataset, indices, cid, seed, classes):
    labels=getattr(dataset,'targets',None)
    if labels is None and hasattr(dataset,'tensors'):labels=dataset.tensors[1]
    if labels is None:raise ValueError('Training dataset must expose labels without applying transforms')
    groups=defaultdict(list)
    for idx in sorted(set(int(i) for i in indices)):
        y=int(labels[idx])
        if not 0<=y<classes:raise ValueError('Class coordinate outside shared output space')
        groups[y].append(idx)
    selected=[]
    for y,items in sorted(groups.items()):
        if len(items)>=2:
            selected.extend(sorted(items,key=lambda i:seed_of('measure-index-v1',seed,cid,i))[:2])
    return selected


@torch.no_grad()
def response(net,dataset,indices,seed,classes,batch_size):
    sums=torch.zeros(classes,classes,dtype=torch.float64);counts=torch.zeros(classes,dtype=torch.long)
    if not indices:return sums.float(),counts
    with isolated_evaluation_rng():
        random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)
        measured=copy.deepcopy(net).eval()
        device=next(measured.parameters()).device
        loader=DataLoader(Subset(dataset,indices),batch_size=batch_size,shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
        for images,labels in loader:
            logits=measured(images.to(device))['output']
            if logits.ndim!=2 or logits.shape[1]!=classes:raise ValueError('Unaligned classification coordinates')
            probabilities=logits.softmax(1).detach().cpu().double();labels=labels.long().cpu()
            if not torch.isfinite(probabilities).all():raise FloatingPointError('Nonfinite functional response')
            sums.index_add_(0,labels,probabilities);counts.index_add_(0,labels,torch.ones_like(labels))
        del measured
    return (sums/counts.clamp_min(1)[:,None]).float(),counts


def cosine_risk(value,reference):
    a=float(torch.linalg.vector_norm(value));b=float(torch.linalg.vector_norm(reference))
    if min(a,b)<=EPS:return None
    return max(0.,-max(-1.,min(1.,float(torch.dot(value,reference)/(a*b)))))


def semantic_evidence(slots, observations, members, classes, shuffle_seed=None):
    # Pool observations only for evidence. Upload slots remain separate in aggregation.
    by_client=defaultdict(list)
    for i in members:by_client[slots[i][0]].append(i)
    responses={};valid={}
    for cid,indices in by_client.items():
        matrix=torch.zeros(classes,classes,dtype=torch.float64);mask=torch.zeros(classes,dtype=torch.bool)
        for y in range(classes):
            values=[observations[i]['delta'][y].double() for i in indices
                    if observations[i] is not None and observations[i]['counts'][y]>=2]
            if values:matrix[y]=torch.stack(values).mean(0);mask[y]=True
        responses[cid]=matrix;valid[cid]=mask
    scores={};valid_classes={}
    for cid, matrix in responses.items():
        risks=[];labels=[]
        for y in range(classes):
            peers=[v[y] for other,v in responses.items() if other!=cid and valid[other][y]]
            if not valid[cid][y] or not peers:continue
            risk=cosine_risk(matrix[y],torch.stack(peers).mean(0))
            if risk is not None:risks.append(risk);labels.append(y)
        if risks:scores[cid]=sum(risks)/len(risks);valid_classes[cid]=labels
    original=dict(scores);donors={cid:cid for cid in scores}
    if shuffle_seed is not None and len(scores)>1:
        ids=np.array(sorted(scores),dtype=np.int64)
        order=np.random.default_rng(shuffle_seed).permutation(ids).tolist()
        for cid,donor in zip(order,order[1:]+order[:1]):
            scores[cid]=original[donor];donors[cid]=donor
    return scores,{'valid_classes':valid_classes,'original_scores':original,'scores':dict(scores),'donors':donors,
                   'score_distribution_preserved':sorted(original.values())==sorted(scores.values()),
                   'permuted_client_count':sum(donors[c]!=c for c in donors),
                   'changed_score_count':sum(scores[c]!=original[c] for c in scores)}


class ScopeHooks:
    is_scope=True

    def __init__(self,args,models):
        self.method=args.method
        if self.method not in MODES:raise ValueError('Unknown scope experiment')
        if len(models)!=7:raise ValueError('Original seven profiles are required')
        self.profile_shapes=[{k:v.shape for k,v in m.state_dict().items()} for m in models]
        self.regions=build_regions(models)
        self.geometry_hash=hashlib.sha256(json.dumps(self.regions,sort_keys=True).encode()).hexdigest()
        self.shared_shapes={k:tuple(min(m.state_dict()[k].shape[d] for m in models) for d in range(v.ndim))
                            for k,v in models[-1].state_dict().items()}
        self.config={'spec_version':1,'gate_lambda':float(args.conflict_lambda),'classes':args.num_classes,
                     'samples_per_class':2,'min_class_samples':2,'history_pseudocount':1,
                     'history_key':'physical-client/structural-region','semantic':'class-conditioned post-minus-pre probability response',
                     'measurement_views':'replayed training transforms; isolated seed hash(seed,round,physical_client)',
                     'direction_reference':'distinct other clients; average repeated slots within each client',
                     'semantic_evidence':'classwise LOO cosine disagreement of prediction-response changes',
                     'negative_control_numpy':np.__version__,
                     'aggregation_anchor':'ungated native dispatch fusion','shuffle':'permute valid scalar scores; preserve validity metadata'}
        if not math.isfinite(args.conflict_lambda) or args.conflict_lambda<0:raise ValueError('Invalid gate strength')
        if self.method=='s01' and args.conflict_lambda!=4:raise ValueError('S01 is fixed native M2 lambda=4')
        self.seed=args.seed;self.classes=args.num_classes;self.batch_size=args.local_bs
        self.uses_measurement=self.method not in ('s00','s01')
        self.needs_dispatch=self.method not in ('s00','s02')
        self.history={};self.round_index=-1;self.plan_cache={};self.round_diagnostics={}
        if self.method=='s01':
            legacy=copy.copy(args);legacy.method='m2';self.legacy=MethodHooks(legacy,models)
            self.config['legacy_m2_options']=self.legacy.config

    def configure_local(self,local):
        pass

    def start_round(self,round_index):
        if round_index!=self.round_index+1:raise ValueError('Scope round sequence differs from checkpoint')
        self.round_index=round_index;self.slots=[];self.observations=[]

    def before_slot(self,net,dataset,indices,cid):
        if not self.uses_measurement:return None
        begin=time.perf_counter()
        if cid not in self.plan_cache:
            self.plan_cache[cid]=semantic_plan(dataset,indices,cid,self.seed,self.classes)
        plan=self.plan_cache[cid];seed=seed_of('functional-view-v1',self.seed,self.round_index,cid)
        pre,counts=response(net,dataset,plan,seed,self.classes,self.batch_size)
        return {'plan':plan,'seed':seed,'pre':pre,'counts':counts,'seconds':time.perf_counter()-begin}

    def finish_slot(self,net,dataset,before,cid,profile):
        self.slots.append((int(cid),int(profile)))
        if before is None:
            self.observations.append(None);return {}
        begin=time.perf_counter()
        post,counts=response(net,dataset,before['plan'],before['seed'],self.classes,self.batch_size)
        if not torch.equal(counts,before['counts']):raise ValueError('Pre/post measurement classes differ')
        self.observations.append({'delta':post-before['pre'],'counts':counts})
        count=int((counts>=2).sum())
        return {'semantic_measurement_seconds':before['seconds']+time.perf_counter()-begin,
                'semantic_valid_classes':count,'semantic_missing_class_fraction':1-count/self.classes,
                'semantic_measured_examples':len(before['plan']),
                'semantic_response_delta_sha256':hashlib.sha256((post-before['pre']).contiguous().numpy().tobytes()).hexdigest(),
                'functional_class_changes':[{'class':int(y),'true_class_probability_change':float(post[y,y]-before['pre'][y,y]),
                    'response_delta_l2':float((post[y]-before['pre'][y]).norm())} for y in torch.where(counts>=2)[0]],
                'semantic_payload_bytes_proxy':count*(self.classes*4+8),
                'measurement_data':'local training partition only; no official test data'}

    def state_dict(self):
        return {'method':self.method,'config':copy.deepcopy(self.config),'geometry_hash':self.geometry_hash,
                'history':dict(self.history),'round_index':self.round_index}

    def load_state_dict(self,state,full):
        if (state['method']!=self.method or state['config']!=self.config or state['geometry_hash']!=self.geometry_hash):
            raise ValueError('Scope checkpoint method/config/geometry differs')
        history=state['history']
        if any(not isinstance(v,int) or v<1 for v in history.values()):raise ValueError('Invalid confidence history')
        if not isinstance(state['round_index'],int) or state['round_index']< -1:raise ValueError('Invalid scope round state')
        for key,n in history.items():
            cid,region=map(int,key.split(':'))
            if cid<0 or not 0<=region<len(self.regions) or n>state['round_index']+1:raise ValueError('Invalid deduplicated history state')
        self.history=dict(history);self.round_index=state['round_index']

    def region_deltas(self,region,uploads,dispatched,members):
        return {i:torch.cat([(uploads[i][c['key']][slices(c)]-dispatched[i][c['key']][slices(c)]).reshape(-1)
                             for c in region['cells']]).double() for i in members}

    def direction(self,deltas,lens):
        by_client=defaultdict(list)
        for i in deltas:by_client[self.slots[i][0]].append(i)
        for cid,ids in by_client.items():
            if any(lens[i]!=lens[ids[0]] for i in ids):raise ValueError('Repeated physical client has inconsistent dataset weight')
        pooled={cid:sum((deltas[i]*lens[i] for i in ids))/sum(lens[i] for i in ids) for cid,ids in by_client.items()}
        weights={cid:lens[ids[0]] for cid,ids in by_client.items()}
        risks={}
        for i,value in deltas.items():
            cid=self.slots[i][0];peers=[c for c in pooled if c!=cid]
            if not peers:continue
            reference=sum((pooled[c]*weights[c] for c in peers))/sum(weights[c] for c in peers)
            risk=cosine_risk(value,reference)
            if risk is not None:risks[i]=risk
        return risks

    @torch.no_grad()
    def aggregate(self,uploads,lens,previous,dispatched=None):
        if len(self.slots)!=len(uploads) or len(self.observations)!=len(uploads):raise ValueError('Scope slot observations misaligned')
        if not uploads or len(lens)!=len(uploads) or any(n<=0 for n in lens):raise ValueError('Invalid upload counts')
        for i,u in enumerate(uploads):
            p=self.slots[i][1]
            if not 0<=p<7 or {k:v.shape for k,v in u.items()}!=self.profile_shapes[p]:raise ValueError('Slot/profile mapping differs from actual model shape')
            if u.keys()!=previous.keys():raise ValueError('Upload keys differ')
            for k,v in previous.items():
                prefix(v,u[k])
                if not torch.isfinite(u[k]).all():raise FloatingPointError('Nonfinite scope upload')
        if self.needs_dispatch:
            if dispatched is None or len(dispatched)!=len(uploads):raise ValueError('Actual dispatch snapshots required')
            for a,b in zip(uploads,dispatched):
                if a.keys()!=b.keys() or any(a[k].shape!=b[k].shape for k in a):raise ValueError('Dispatch shapes differ')
        if self.method=='s01':
            result=self.legacy.aggregate(uploads,lens,previous,dispatched)
            self.round_diagnostics=dict(self.legacy.round_diagnostics,scope_mode=self.method)
            return result
        base=Aggregation_AdaptiveFL(uploads,lens,previous)
        result={k:v.clone() for k,v in base.items()}
        records=[];work=[];pending_history=set()
        for region in self.regions:
            members=[i for i,(_,p) in enumerate(self.slots) if p in region['profiles']]
            clients={self.slots[i][0] for i in members}
            record={'region_id':region['id'],'numel':region['numel'],'support_profiles':region['profiles'],
                    'slots':members,'physical_clients':sorted(clients),'slot_count':len(members),'client_count':len(clients)}
            records.append(record)
            if self.method in ('s00','s02') or len(clients)<2:
                record.update(fallback='observation_only' if self.method=='s02' else 'native_control' if self.method=='s00' else 'insufficient_distinct_clients',gates=[1.]*len(members))
                if self.method=='s02' and len(clients)>=2:
                    sem,info=semantic_evidence(self.slots,self.observations,members,self.classes)
                    record.update(semantic=info,semantic_risks={str(c):v for c,v in sem.items()})
                continue
            deltas=self.region_deltas(region,uploads,dispatched,members)
            direction={} if self.method=='s04' else self.direction(deltas,lens)
            sem,sem_info=semantic_evidence(self.slots,self.observations,members,self.classes,
                seed_of('negative-control-v1',self.seed,self.round_index,region['id']) if self.method=='s08' else None)
            risks={};eligible={};confidence={}
            for i in members:
                cid=self.slots[i][0];key=f'{cid}:{region["id"]}'
                valid=(i in direction) if self.method=='s03' else (cid in sem) if self.method=='s04' else (i in direction and cid in sem)
                eligible[i]=valid
                risk=direction[i] if self.method=='s03' and valid else sem[cid] if self.method=='s04' and valid else direction[i]*sem[cid] if valid else 0.
                n=self.history.get(key,0);c=n/(n+1) if self.method in ('s07','s08','s09') else 1.
                confidence[i]=c;risks[i]=risk*c
                if valid:
                    if self.method in ('s07','s08','s09'):pending_history.add(key)
            record.update(direction_risks={str(i):v for i,v in direction.items()},semantic=sem_info,
                          semantic_risks={str(c):v for c,v in sem.items()},confidence={str(i):v for i,v in confidence.items()},
                          eligible_slots=[i for i in members if eligible[i]],
                          insufficient_history_slots=[i for i in members if eligible[i] and confidence[i]==0],fallback=None if any(eligible.values()) else 'no_valid_comparison')
            work.append((region,members,deltas,risks,record))
        global_scores={}
        if self.method=='s05':
            pooled=defaultdict(list)
            for region,members,deltas,risks,record in work:
                for cid in {self.slots[i][0] for i in record['eligible_slots']}:
                    values=[risks[i] for i in record['eligible_slots'] if self.slots[i][0]==cid]
                    pooled[cid].append((region['numel'],sum(values)/len(values)))
            global_scores={cid:sum(n*r for n,r in v)/sum(n for n,_ in v) for cid,v in pooled.items()}
        changed_regions=0;gated_region_numel=0;triggered=0;gate_comparisons=0
        old_square=0.;new_square=0.
        for region,members,deltas,risks,record in work:
            gates={i:math.exp(-self.config['gate_lambda']*(global_scores.get(self.slots[i][0],0.) if self.method=='s05' else risks[i])) for i in members}
            record['gates']=[gates[i] for i in members]
            triggered+=sum(g<1-1e-12 for g in gates.values());gate_comparisons+=len(members)
            mass=sum(lens[i] for i in members);gated_mass=sum(lens[i]*gates[i] for i in members)
            native_delta=sum((deltas[i]*lens[i] for i in members))/mass
            normalized=self.method!='s09'
            applied=sum((deltas[i]*(lens[i]*gates[i]) for i in members))/(gated_mass if normalized else mass)
            if normalized and max(gates.values())-min(gates.values())<=1e-14:applied=native_delta
            old_square+=float(native_delta.square().sum());new_square+=float(applied.square().sum())
            record.update(normalization='gated_coverage_mass' if normalized else 'native_coverage_mass',
                          gated_mass_ratio=gated_mass/mass,native_learning_update_l2=float(native_delta.norm()),
                          applied_learning_update_l2=float(applied.norm()),client_gate_extrapolation=self.method=='s05')
            intervene=any(g<1-1e-12 for g in gates.values())
            if intervene:gated_region_numel+=region['numel']
            if not intervene or (normalized and max(gates.values())-min(gates.values())<=1e-14):
                record['returned_native_exactly']=True;continue
            correction=applied-native_delta;offset=0;region_changed=0
            for cell in region['cells']:
                key=cell['key'];sl=slices(cell);n=cell['numel']
                update=correction[offset:offset+n].reshape(result[key][sl].shape).to(result[key].dtype)
                result[key][sl]=base[key][sl]+update
                region_changed+=int((result[key][sl]!=base[key][sl]).sum());offset+=n
            record['changed_coordinates']=region_changed;record['returned_native_exactly']=region_changed==0
            changed_regions+=region_changed>0
        for key in pending_history:self.history[key]=self.history.get(key,0)+1
        total=sum(v.numel() for v in previous.values());covered=sum(r['numel'] for r in records if r['slot_count'])
        differences=[(result[k]-base[k]).double() for k in base]
        direction_values=[v for r in records for v in r.get('direction_risks',{}).values()]
        sem_valid=sum(sum(self.slots[i][0] in {int(c) for c in r.get('semantic_risks',{})} for i in r['slots']) for r in records)
        candidates=sum(r['slot_count'] for r in records if r['client_count']>=2)
        self.round_diagnostics={'scope_mode':self.method,'measurement_enabled':self.uses_measurement,'regions':records,'global_client_risks':global_scores,
            'direction_valid_comparisons':len(direction_values),
            'direction_conflict_fraction':sum(v>0 for v in direction_values)/len(direction_values) if direction_values else None,
            'semantic_valid_comparison_fraction':sem_valid/candidates if candidates else 0.,
            'semantic_missing_comparison_fraction':1-sem_valid/candidates if candidates else 1.,
            'gate_trigger_region_slot_fraction':triggered/gate_comparisons if gate_comparisons else 0.,
            'gate_target_covered_parameter_fraction':gated_region_numel/covered if covered else 0.,
            'actual_changed_parameter_fraction_full':sum(int(v.ne(0).sum()) for v in differences)/total,
            'difference_from_native_aggregate_l2':math.sqrt(sum(float(v.square().sum()) for v in differences)),
            'regions_with_actual_changes':changed_regions,
            'neutral_or_fallback_region_fraction':sum(r.get('returned_native_exactly',True) for r in records)/len(records),
            'insufficient_evidence_region_fraction':sum(r.get('fallback') is not None for r in records)/len(records),
            'comparable_native_learning_update_l2':math.sqrt(old_square),
            'comparable_applied_learning_update_l2':math.sqrt(new_square),
            'history_unique_events_this_round':len(pending_history),'history_total_events':sum(self.history.values()),
            'measurement_valid_class_fraction':sum(int((o['counts']>=2).sum()) for o in self.observations if o is not None)/(len(uploads)*self.classes) if self.uses_measurement else None,
            'aggregation_note':'native dispatch fusion preserved; gate only adjusts actual local training increments'}
        return result
