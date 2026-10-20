"""Eleven isolated E03 controls. REF calls the original m2 hook directly."""
import copy
import math
import statistics

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from adapterfl_methods import EPS, MethodHooks, stats
from models.Fed import Aggregation_AdaptiveFL
from tonight_aggregation import common_slices, scores_from_moments


class LocalReferenceHooks(MethodHooks):
    """Exact native M2 return value; extra read-only update-norm diagnostics."""
    is_local12=True

    def __init__(self,args,models):
        super().__init__(args,models)
        if self.method!='m2' or args.conflict_lambda!=4.:raise ValueError('REF is native M2lambda4')
        self.config=dict(self.config,local_candidate='REF',diagnostic_only=True)

    def set_slots(self,slots):pass

    @torch.no_grad()
    def aggregate(self,uploads,lens,previous,dispatched=None):
        result=super().aggregate(uploads,lens,previous,dispatched)
        norm=math.sqrt(sum(float((result[k]-v).double().square().sum()) for k,v in previous.items()))
        cosine=self.round_diagnostics.get('cosine',[None]*len(uploads))
        self.round_diagnostics.update(candidate='REF',actual_update_l2=norm,e03_update_l2=norm,
            difference_from_e03_l2=0.,changed_from_e03_fraction=0.,update_l2_ratio_e03=1. if norm else None,
            effective_step_along_e03=1. if norm else None,update_cosine_e03=1. if norm else None,
            conflict_slot_count=sum(c is not None and c<0 for c in cosine),normalized_share_changed_slots=0)
        return result


class Local12Hooks:
    is_local12=True

    def __init__(self,args,models):
        if args.method not in [f'l{i:02d}' for i in range(1,12)] or args.conflict_lambda!=4.:
            raise ValueError('Fixed local controls require L01-L11 and lambda4')
        self.method=args.method
        self.classifier_start=getattr(args,'classifier_protection_start_round',1)
        self.classifier_end=getattr(args,'classifier_protection_end_round',None)
        if self.method!='l04' and self.classifier_start!=1:
            raise ValueError('Delayed classifier protection applies only to L04')
        if self.classifier_end is not None and (self.method!='l04' or self.classifier_end<self.classifier_start):
            raise ValueError('First-round-only protection requires L04 starting at round 1')
        native=copy.copy(args);native.method='m2';self.legacy=MethodHooks(native,models)
        full=models[-1]
        self.bn={f'{name}.{p}' for name,module in full.named_modules() if isinstance(module,_BatchNorm)
                 for p,_ in module.named_parameters(recurse=False)}
        self.classifier={k for k in full.state_dict() if k.startswith('classifier.')}
        self.features=set(full.state_dict())-self.bn-self.classifier
        if not self.bn or not self.classifier or not self.features:raise ValueError('Original ResNet parameter roles required')
        ratio=[float(x) for x in args.client_hetero_ration.split(':')]
        self.low_boundary=int(args.num_users*round(ratio[0]/sum(ratio),2))
        self.config=dict(self.legacy.config,local_candidate=self.method,
            parameter_roles={name:sorted(getattr(self,name)) for name in ('bn','classifier','features')},
            low_capacity_physical_ids=f'[0,{self.low_boundary})',balanced_evidence='equal mean of valid feature/noBN and classifier cosines',
            pairwise_evidence='median across distinct peer clients of their median valid slot-pair cosines',
            amplitude_epsilon=EPS)
        if self.classifier_start!=1:
            self.config['classifier_protection_start_round']=self.classifier_start
        if self.classifier_end is not None:
            self.config['classifier_protection_end_round']=self.classifier_end
        self.rounds_seen=0;self.slots=None;self.round_diagnostics={}

    def configure_local(self,local):pass

    def set_slots(self,slots):self.slots=[(int(c),int(p)) for c,p in slots]

    def state_dict(self):
        return dict(method=self.method,config=self.config,rounds_seen=self.rounds_seen,legacy=self.legacy.state_dict())

    def load_state_dict(self,state,full):
        if state['method']!=self.method or state['config']!=self.config:raise ValueError('Local method identity differs')
        self.legacy.load_state_dict(state['legacy'],full);self.rounds_seen=state['rounds_seen']
        if not isinstance(self.rounds_seen,int) or self.rounds_seen<0:raise ValueError('Invalid round index')

    def subset_scores(self,keys,u,n,w,s):
        sub=lambda d:{k:d[k] for k in d if k in keys}
        self.legacy.conflict([sub(d) for d in u],n,sub(w),[sub(d) for d in s])
        return self.legacy.round_diagnostics['cosine']

    @torch.no_grad()
    def aggregate(self,uploads,lens,previous,dispatched=None):
        e03=self.legacy.aggregate(uploads,lens,previous,dispatched)
        diag=copy.deepcopy(self.legacy.round_diagnostics)
        original_weights=list(diag['conflict_weights'])
        original_cos=diag.get('cosine',[None]*len(uploads));cos=list(original_cos)
        logits=list(diag.get('unnormalized_log_weights',[0.]*len(uploads)))
        sl=common_slices(uploads,previous)
        result={k:v.clone() for k,v in e03.items()}
        if self.method in ('l01','l02') and not (self.method=='l02' and self.rounds_seen==0):
            # F + native_fusion - M2_fusion is the same mathematical SGD-only
            # weighting, without subtract/re-add roundoff on the SGD component.
            aligned=all(torch.equal(s[k],previous[k][tuple(slice(0,d) for d in s[k].shape)]) for s in dispatched for k in previous)
            diag['dispatch_matches_full_anchor']=aligned
            if not aligned:
                native_fusion=Aggregation_AdaptiveFL(dispatched,lens,previous)
                m2_fusion=Aggregation_AdaptiveFL(dispatched,[n*g for n,g in zip(lens,diag['conflict_weights'])],previous)
                result={k:e03[k]+(native_fusion[k]-m2_fusion[k]) for k in previous}
            # Synced dispatches have the SAME fusion mathematically: keep exact
            # E03 rather than introducing a false post-init mechanism by rounding.
        classifier_active=self.method!='l04' or (self.rounds_seen+1>=self.classifier_start and
            (self.classifier_end is None or self.rounds_seen+1<=self.classifier_end))
        if self.method in ('l03','l04','l05') and classifier_active:
            protected=self.bn if self.method=='l03' else self.classifier if self.method=='l04' else self.bn|self.classifier
            native=Aggregation_AdaptiveFL(uploads,lens,previous)
            for k in protected:result[k]=native[k]
            diag['native_protected_keys']=sorted(protected)
        if self.method=='l04':
            diag['classifier_protection_active']=classifier_active
            diag['classifier_protection_start_round']=self.classifier_start
            if self.classifier_end is not None:diag['classifier_protection_end_round']=self.classifier_end
        if self.method in ('l06','l07'):
            if self.slots is None or len(self.slots)!=len(uploads):raise ValueError('Actual slot identities required')
            protected=[i for i,(cid,profile) in enumerate(self.slots) if
                (self.method=='l06' and cid<self.low_boundary and cos[i] is not None and cos[i]<0)
                or (self.method=='l07' and profile==6)]
            for i in protected:logits[i]=0.
            diag['protected_slot_indices']=protected
            diag['protection_note']='Set intrinsic gate1 BEFORE common stabilization; retain independent uploads and coverage denominators'
        if self.method=='l08':
            cos=self.subset_scores(set(previous)-self.bn,uploads,lens,previous,dispatched)
        if self.method=='l09':
            feature=self.subset_scores(self.features,uploads,lens,previous,dispatched)
            head=self.subset_scores(self.classifier,uploads,lens,previous,dispatched)
            cos=[statistics.mean([v for v in (a,b) if v is not None]) if a is not None or b is not None else None for a,b in zip(feature,head)]
            diag.update(feature_cosine=feature,classifier_cosine=head,valid_evidence_parts=[int(a is not None)+int(b is not None) for a,b in zip(feature,head)])
        if self.method=='l10':
            if self.slots is None or len(self.slots)!=len(uploads):raise ValueError('Actual physical identities required')
            deltas=[{k:(u[k][sl[k]]-s[k][sl[k]]).double() for k in previous} for u,s in zip(uploads,dispatched)]
            cos=[];peer_counts=[]
            for i,d in enumerate(deltas):
                peers={}
                for j,r in enumerate(deltas):
                    cid=self.slots[j][0]
                    if cid==self.slots[i][0]:continue
                    moment=(sum(float((d[k]*r[k]).sum()) for k in previous),
                        sum(float(v.square().sum()) for v in d.values()),sum(float(v.square().sum()) for v in r.values()))
                    value=scores_from_moments([moment])[0]
                    if value is not None:peers.setdefault(cid,[]).append(value)
                values=[statistics.median(v) for v in peers.values()]
                cos.append(statistics.median(values) if values else None);peer_counts.append(len(values))
            diag['valid_distinct_physical_peers']=peer_counts
        if self.method=='l11':
            deltas=[{k:(u[k][sl[k]]-s[k][sl[k]]).double() for k in previous} for u,s in zip(uploads,dispatched)]
            weighted={k:sum(n*d[k] for n,d in zip(lens,deltas)) for k in previous};risks=[];moments=[]
            for i,d in enumerate(deltas):
                if sum(lens)==lens[i]:risks.append(0.);moments.append(None);continue
                ref={k:(weighted[k]-lens[i]*d[k])/(sum(lens)-lens[i]) for k in previous}
                dot=sum(float((d[k]*ref[k]).sum()) for k in previous)
                ds=sum(float(v.square().sum()) for v in d.values());rs=sum(float(v.square().sum()) for v in ref.values())
                risk=min(1.,max(0.,2*max(0.,-dot)/(ds+rs+EPS))) if min(ds,rs)>EPS**2 else 0.
                risks.append(risk);moments.append(dict(dot=dot,delta_square=ds,reference_square=rs))
            logits=[-4*r for r in risks];diag.update(amplitude_risk=risks,direction_moments=moments)
        if self.method in ('l08','l09','l10'):
            logits=[-4*max(0.,-v) if v is not None else 0. for v in cos]
        if self.method in ('l06','l07','l08','l09','l10','l11'):
            weights=[math.exp(max(-80.,v-max(logits))) for v in logits]
            result=Aggregation_AdaptiveFL(uploads,[n*g for n,g in zip(lens,weights)],previous)
            diag.update(cosine=cos,conflict_weights=weights,unnormalized_log_weights=logits,weight_distribution=stats(weights))
        weights=diag['conflict_weights'];negative=[i for i,c in enumerate(cos) if c is not None and c<0]
        share=[n*g/sum(a*b for a,b in zip(lens,weights)) for n,g in zip(lens,weights)]
        original_share=[n*g/sum(a*b for a,b in zip(lens,original_weights)) for n,g in zip(lens,original_weights)]
        dot=diff=0.;changed=0;numel=0;norms=[0.,0.]
        for k,old in previous.items():
            actual=(result[k]-old).double();base=(e03[k]-old).double()
            norms[0]+=float(actual.square().sum());norms[1]+=float(base.square().sum())
            dot+=float((actual*base).sum());diff+=float((result[k]-e03[k]).double().square().sum())
            changed+=int(result[k].ne(e03[k]).sum());numel+=old.numel()
        diag.update(candidate=self.method,original_e03_cosine=original_cos,
            original_e03_weights=original_weights,normalized_slot_share=share,
            normalized_share_changed_slots=sum(a!=b for a,b in zip(share,original_share)),
            conflict_slot_count=len(negative),conflict_slot_indices=negative,negative_fraction=len(negative)/len(uploads),
            fallback_slots=sum(c is None for c in cos),actual_update_l2=math.sqrt(norms[0]),e03_update_l2=math.sqrt(norms[1]),
            difference_from_e03_l2=math.sqrt(diff),changed_from_e03_fraction=changed/numel,
            update_l2_ratio_e03=math.sqrt(norms[0]/norms[1]) if norms[1] else None,
            effective_step_along_e03=dot/norms[1] if norms[1] else None,
            update_cosine_e03=dot/math.sqrt(norms[0]*norms[1]) if min(norms) else None,
            round_number=self.rounds_seen+1)
        self.round_diagnostics=diag;self.rounds_seen+=1
        return result
