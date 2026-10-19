"""Ten fixed E03 ablations, using the original sampler, SGD and aggregators."""
import copy
import math

import torch

from adapterfl_methods import EPS, MethodHooks, stats
from models.Fed import Aggregation_AdaptiveFL


ORDER = ('t01','t04','t05','t06','t09','t02','t03','t07','t08','t10')


def common_slices(states, keys):
    return {k:tuple(slice(0,min(s[k].shape[d] for s in states)) for d in range(states[0][k].ndim)) for k in keys}


def scores_from_moments(moments):
    return [max(-1.,min(1.,dot/math.sqrt(ds*rs))) if min(ds,rs)>EPS**2 else None for dot,ds,rs in moments]


class TonightHooks:
    is_tonight = True

    def __init__(self,args,models):
        if args.method not in ORDER or args.conflict_lambda != 4.:
            raise ValueError('Fixed T01-T10 require lambda=4')
        self.method=args.method
        native=copy.copy(args);native.method='m2'
        self.legacy=MethodHooks(native,models)
        self.config=dict(self.legacy.config,tonight_candidate=self.method,
            fixed_scale=.98, scoring_definition='docs/E03_TONIGHT10_DESIGN.md',
            physical_pooling='same physical client: mean slot evidence, original sample mass once')
        self.rounds_seen=0;self.slots=None;self.round_diagnostics={}

    def configure_local(self,local):
        pass

    def set_slots(self,slots):
        self.slots=[(int(c),int(p)) for c,p in slots]

    def state_dict(self):
        return dict(method=self.method,config=self.config,rounds_seen=self.rounds_seen,legacy=self.legacy.state_dict())

    def load_state_dict(self,state,full):
        if state['method']!=self.method or state['config']!=self.config:
            raise ValueError('Fixed experiment configuration differs')
        self.legacy.load_state_dict(state['legacy'],full)
        self.rounds_seen=state['rounds_seen']
        if not isinstance(self.rounds_seen,int) or self.rounds_seen<0:raise ValueError('Invalid round counter')

    def groups(self,lens):
        if self.slots is None or len(self.slots)!=len(lens):raise ValueError('Actual slot identities required')
        groups={}
        for i,(cid,_) in enumerate(self.slots):groups.setdefault(cid,[]).append(i)
        if any(len({lens[i] for i in ids})!=1 for ids in groups.values()):
            raise ValueError('Same physical client must have the same sample count')
        return groups

    def physical_scores(self,uploads,lens,sent,sl,pairwise):
        groups=self.groups(lens);moments=[];coverage=[]
        for i,u in enumerate(uploads):
            peers=[ids for cid,ids in groups.items() if cid!=self.slots[i][0]]
            if not peers:
                moments.append((0.,0.,0.));coverage.append(0);continue
            dot=ds=rs=0.;coordinates=0
            if not pairwise:
                mass=sum(lens[ids[0]] for ids in peers)
                for k in u:
                    d=(u[k][sl[k]]-sent[i][k][sl[k]]).double()
                    ref=sum(lens[ids[0]]*sum((uploads[j][k][sl[k]]-sent[j][k][sl[k]]).double() for j in ids)/len(ids) for ids in peers)/mass
                    dot+=float((d*ref).sum());ds+=float(d.square().sum());rs+=float(ref.square().sum())
                    coordinates+=d.numel()
            else:
                # Direct-sum cosine of pairwise actual overlaps. Each physical
                # peer has mass n_c once, evenly divided among its slot views.
                mass=sum(lens[ids[0]] for ids in peers)
                for ids in peers:
                    weight=lens[ids[0]]/(mass*len(ids))
                    for j in ids:
                        pair=common_slices([u,uploads[j]],u)
                        for k in u:
                            d=(u[k][pair[k]]-sent[i][k][pair[k]]).double()
                            r=(uploads[j][k][pair[k]]-sent[j][k][pair[k]]).double()
                            dot+=weight*float((d*r).sum());ds+=weight*float(d.square().sum());rs+=weight*float(r.square().sum())
                            coordinates+=d.numel()
            moments.append((dot,ds,rs));coverage.append(coordinates)
        return scores_from_moments(moments),coverage

    @torch.no_grad()
    def aggregate(self,uploads,lens,previous,dispatched=None):
        # Native M2 validates uploads/snapshots and supplies the original scores.
        e03=self.legacy.aggregate(uploads,lens,previous,dispatched)
        original=copy.deepcopy(self.legacy.round_diagnostics)
        diag=copy.deepcopy(original);sl=common_slices(uploads,previous)
        result={k:v.clone() for k,v in e03.items()}
        if self.method in ('t05','t06','t07','t08'):
            if self.method in ('t05','t06'):
                cos,coverage=self.physical_scores(uploads,lens,dispatched,sl,self.method=='t06')
                diag['pairwise_or_common_comparison_coordinates']=coverage
                diag['physical_client_count']=len(self.groups(lens))
            else:
                keys=[k for k in previous if k.startswith('classifier.')==(self.method=='t07')]
                if not keys:raise ValueError('Missing classifier/backbone evidence keys')
                # Reuse the exact M2 conflict implementation on the chosen keys.
                sub=lambda s:{k:s[k] for k in keys}
                self.legacy.conflict([sub(u) for u in uploads],lens,sub(previous),[sub(s) for s in dispatched])
                cos=self.legacy.round_diagnostics['cosine']
                diag['scoring_keys']=keys
            logits=[-4*max(0.,-v) if v is not None else 0. for v in cos]
            weights=[math.exp(max(-80.,x-max(logits))) for x in logits]
            result=Aggregation_AdaptiveFL(uploads,[n*w for n,w in zip(lens,weights)],previous)
            diag.update(cosine=cos,unnormalized_log_weights=logits,conflict_weights=weights,
                weight_distribution=stats(weights),fallback_slots=sum(c is None for c in cos))
        weights=diag['conflict_weights'];cos=diag.get('cosine',[None]*len(uploads))
        negative=[i for i,c in enumerate(cos) if c is not None and c<0]
        diag.update(candidate=self.method,original_e03_cosine=original.get('cosine'),
            original_e03_weights=original['conflict_weights'],conflict_slot_indices=negative,
            conflict_slot_count=len(negative),negative_fraction=len(negative)/len(uploads),
            valid_cosine_negative_fraction=len(negative)/max(1,sum(c is not None for c in cos)),
            common_evidence_numel=sum(v[sl[k]].numel() for k,v in previous.items()))
        if self.method=='t04' and self.rounds_seen==0:
            result=Aggregation_AdaptiveFL(uploads,lens,previous)
        if self.method in ('t01','t02','t03'):
            apply=self.method!='t03' or bool(negative)
            if apply:
                for k,old in previous.items():
                    if self.method=='t02':result[k]=old+.98*(e03[k]-old)
                    else:
                        mask=self.legacy.region(k,old,'remainder')
                        result[k][mask]=old[mask]+.98*(e03[k][mask]-old[mask])
            diag['scale_applied']=apply
        if self.method=='t10':
            deltas=[{k:u[k]-s[k] for k in previous} for u,s in zip(uploads,dispatched)]
            learning=Aggregation_AdaptiveFL(deltas,[n*w for n,w in zip(lens,weights)],{k:torch.zeros_like(v) for k,v in previous.items()})
            for k in result:
                correction=.02*learning[k];correction[sl[k]]=0
                result[k]=e03[k]-correction
            diag['learning_only_scale']=.98
        diag['protected_components']=[]
        if self.method=='t09' and negative:
            deltas=[{k:(u[k][sl[k]]-s[k][sl[k]]).double() for k in previous} for u,s in zip(uploads,dispatched)]
            mass=sum(lens);weighted={k:sum(n*d[k] for n,d in zip(lens,deltas)) for k in previous}
            beta=[n*w/sum(a*b for a,b in zip(lens,weights)) for n,w in zip(lens,weights)]
            correction={k:torch.zeros_like(v[sl[k]],dtype=torch.float64) for k,v in previous.items()}
            for i in negative:
                coefficient=max(0.,lens[i]/mass-beta[i])
                if coefficient==0:continue
                ref={k:(weighted[k]-lens[i]*deltas[i][k])/(mass-lens[i]) for k in previous}
                norm=math.sqrt(sum(float(r.square().sum()) for r in ref.values()))
                if norm<=EPS:raise AssertionError('Negative native cosine with invalid reference')
                projection_scalar=min(0.,sum(float((deltas[i][k]*(ref[k]/norm)).sum()) for k in previous))
                remainder_square=projection_square=0.
                for k in previous:
                    p=projection_scalar*(ref[k]/norm);z=deltas[i][k]-p
                    correction[k]+=coefficient*z
                    remainder_square+=float(z.square().sum());projection_square+=float(p.square().sum())
                diag['protected_components'].append(dict(slot=i,restored_coefficient=coefficient,
                    negative_projection_l2=math.sqrt(projection_square),remaining_l2=math.sqrt(remainder_square),
                    restored_l2=coefficient*math.sqrt(remainder_square)))
            for k in result:result[k][sl[k]]=e03[k][sl[k]]+correction[k].to(e03[k].dtype)
            diag['restoration_triangle_bound_l2']=sum(c['restored_l2'] for c in diag['protected_components'])
            diag['restoration_actual_l2']=math.sqrt(sum(float(c.square().sum()) for c in correction.values()))
        squares=[0.,0.,0.];dot=0.;changed=0;numel=0
        for k,old in previous.items():
            a=(result[k]-old).double();b=(e03[k]-old).double();d=(result[k]-e03[k]).double()
            for j,x in enumerate((a,b,d)):squares[j]+=float(x.square().sum())
            dot+=float((a*b).sum());changed+=int(result[k].ne(e03[k]).sum());numel+=old.numel()
        diag.update(actual_update_l2=math.sqrt(squares[0]),e03_update_l2=math.sqrt(squares[1]),
            difference_from_e03_l2=math.sqrt(squares[2]),changed_from_e03_fraction=changed/numel,
            update_l2_ratio_e03=math.sqrt(squares[0]/squares[1]) if squares[1] else None,
            effective_step_along_e03=dot/squares[1] if squares[1] else None,
            update_cosine_e03=dot/math.sqrt(squares[0]*squares[1]) if min(squares[:2]) else None,
            round_number=self.rounds_seen+1)
        self.round_diagnostics=diag;self.rounds_seen+=1
        return result
