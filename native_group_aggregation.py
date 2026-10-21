"""Ten prespecified Native AdaptiveFL controls. No client-side change or scoring."""
from collections import defaultdict
import math

import torch

from models.Fed import Aggregation_AdaptiveFL
from semcons_scope import build_regions,slices

MODES=tuple(f'g{i:02d}' for i in range(1,11))
REFINEMENTS=tuple(f'p{i:02d}' for i in range(1,11))


def update_norm(state,anchor):
    return math.sqrt(sum(float((state[k]-anchor[k]).double().square().sum()) for k in anchor))


def balanced_weights(method,members,slots,lens,boundaries):
    """Preserve slot uploads; only change their effective server weights."""
    if method=='g06':return [1.]*len(members)
    groups=defaultdict(list)
    for i in members:
        cid,profile=slots[i]
        key=profile if method in ('g01','g04','g05') else cid if method=='g02' else sum(cid>=b for b in boundaries)
        groups[key].append(i)
    total=sum(lens[i] for i in members)
    if method=='g02':
        # Average repeated covered slots, then count the physical dataset once.
        unique_mass=sum(lens[ids[0]] for ids in groups.values())
        factors={key:total/(unique_mass*len(ids)) for key,ids in groups.items()}
    else:
        factors={key:total/(len(groups)*sum(lens[i] for i in ids)) for key,ids in groups.items()}
    result={i:lens[i]*factors[key] for key,ids in groups.items() for i in ids}
    return [result[i] for i in members]


def capability_weights(method,members,slots,lens,boundaries,population=(4.,3.,3.)):
    """Fixed P03-P07 group masses, on covered slots only.

    Per the prespecified neutral safeguard, one physical capability tier uses
    Native even for P06/P07. No independent update is invented for absent groups.
    """
    groups=defaultdict(list)
    for i in members:groups[sum(slots[i][0]>=b for b in boundaries)].append(i)
    if len(groups)==1:return [lens[i] for i in members]
    total=sum(lens[i] for i in members)
    if method in ('p03','p04'):
        masses={g:sum(lens[i] for i in ids) for g,ids in groups.items()}
        if len(set(masses.values()))==1:return balanced_weights('g03',members,slots,lens,boundaries)
        targets={g:math.sqrt(m) if method=='p03' else 1/math.sqrt(m) for g,m in masses.items()}
    elif method=='p05':targets={g:population[g] for g in groups}
    else:targets={g:1. for g in groups}
    denominator=sum(targets.values());result={}
    for g,ids in groups.items():
        target=total*targets[g]/denominator
        if method=='p06':
            clients=defaultdict(list)
            for i in ids:clients[slots[i][0]].append(i)
            mass=sum(lens[js[0]] for js in clients.values())
            for js in clients.values():
                for i in js:result[i]=target*lens[i]/(len(js)*mass)
        elif method=='p07':
            profiles=defaultdict(list)
            for i in ids:profiles[slots[i][1]].append(i)
            for js in profiles.values():
                mass=sum(lens[i] for i in js)
                for i in js:result[i]=target*lens[i]/(len(profiles)*mass)
        else:
            mass=sum(lens[i] for i in ids)
            for i in ids:result[i]=target*lens[i]/mass
    return [result[i] for i in members]


def write_weighted_region(result,previous,uploads,members,weights,cells):
    for cell in cells:
        k=cell['key'];sl=slices(cell);value=torch.zeros_like(previous[k][sl]);count=value.new_zeros(())
        for i,w in zip(members,weights):value+=uploads[i][k][sl]*w;count+=w
        result[k][sl]=value/count


def nonneutral_weights(weights,members,lens):
    return len(set(w/lens[i] for w,i in zip(weights,members)))!=1


def norm_cosine(a,b,anchor):
    dot=aa=bb=0.
    for k in anchor:
        x=(a[k]-anchor[k]).double();y=(b[k]-anchor[k]).double()
        dot+=float((x*y).sum());aa+=float(x.square().sum());bb+=float(y.square().sum())
    return max(-1.,min(1.,dot/math.sqrt(aa*bb))) if aa and bb else None


def reliability(deltas,members,slots,lens):
    """Cluster by physical identity; retain Native total slot masses in the mean.

    Treat client blocks as independent with a common vector dispersion. The
    weighted unbiased dispersion and sum(alpha**2) estimate mean-noise energy.
    This is an exchangeable-client approximation, not a claim of unbiased
    reliability under arbitrary non-IID heteroscedastic updates.
    """
    groups=defaultdict(list)
    for i in members:groups[slots[i][0]].append(i)
    if len(groups)<2:return 1.,dict(reason='one_physical_client',physical_clients=len(groups))
    masses={c:sum(lens[i] for i in ids) for c,ids in groups.items()}
    total=sum(masses.values());weights={c:n/total for c,n in masses.items()}
    pooled={c:sum((deltas[i]*(lens[i]/masses[c]) for i in ids)) for c,ids in groups.items()}
    mean=sum((pooled[c]*weights[c] for c in groups))
    energy=float(mean.square().sum());squared_weights=sum(a*a for a in weights.values())
    denominator=1.-squared_weights
    if denominator<=0:raise FloatingPointError('Invalid independent-client variance denominator')
    dispersion=sum(weights[c]*float((pooled[c]-mean).square().sum()) for c in groups)/denominator
    noise=dispersion*squared_weights
    signal=max(energy-noise,0.)
    gain=signal/(signal+noise) if signal+noise>0 else 1.
    if not all(math.isfinite(x) for x in (energy,dispersion,noise,gain)):
        raise FloatingPointError('Nonfinite region reliability')
    return gain,dict(reason='estimated' if noise>0 else 'zero_dispersion',physical_clients=len(groups),
        native_physical_masses=masses,physical_weight_square_sum=squared_weights,
        mean_energy=energy,unbiased_dispersion_energy=dispersion,estimated_mean_noise_energy=noise,
        nonnegative_signal_energy=signal)


class NativeGroupHooks:
    is_native_groups=True

    def __init__(self,args,models):
        if args.method not in MODES+REFINEMENTS or len(models)!=7:raise ValueError('Fixed seven-model Native controls required')
        self.method=args.method;self.regions=build_regions(models)
        self.shapes=[{k:tuple(v.shape) for k,v in m.state_dict().items()} for m in models]
        ratio=[float(x) for x in args.client_hetero_ration.split(':')]
        if args.num_users!=100 or ratio!=[4.,3.,3.]:raise ValueError('Original 100-client 4:3:3 protocol required')
        self.boundaries=(40,70)
        self.needs_dispatch=self.method in ('g07','g08','g09','g10')
        self.config=dict(version=1,mode=self.method,first_round='exact Native',region_count=len(self.regions),
            grouping='sample-weighted within group, equal present groups; G02 physical dataset mass once',
            cumulative_coverage='distinct physical clients per region per round, including neutral round1',
            compensation='clip(sqrt(arithmetic_mean(all_region_counts)/region_count),0.8,1.2)',
            reliability='physical block Native mean; V=sum(a*norm(d-mean)^2)/(1-sum(a^2)); N=V*sum(a^2); S=max(norm(mean)^2-N,0); S/(S+N); both0 -> 1',
            learning_anchor='Native minus Full after original all-model synchronization; no independent initialization intervention',
            scale_control='match virtual regional method Full update norm on the same uploads and own counter state')
        if self.method in REFINEMENTS:
            self.config=dict(version=1,mode=self.method,first_round='exact Native',region_count=len(self.regions),
                grouping='physical capability IDs 0:40,40:70,70:100; covered slots only',
                group_mass='Native sample-slot mass; P06 counts each covered physical dataset once inside its tier',
                single_tier='Native source weights including P06/P07; P01/P02 global norm matching may rescale covered regions',
                variants='P01 Native direction/G03 L2; P02 G03 direction/Native L2; P03 sqrt mass; P04 inverse sqrt mass; P05 4:3:3; P06 dedup; P07 tier/profile hierarchy; P08 common; P09 complement; P10 classifier Native',
                scale_control='same-input anchored Full increments; zero source/nonzero target is undefined and raises; both zero Native',
                diagnostics='virtual G03 on the same uploads; P01 virtual slot shares are not the applied Native direction weights')
        self.coverage=[0]*len(self.regions);self.rounds_seen=0;self.slots=None;self.round_diagnostics={}

    def configure_local(self,local):pass

    def set_slots(self,slots):self.slots=[(int(c),int(p)) for c,p in slots]

    def state_dict(self):
        return dict(method=self.method,config=self.config,rounds_seen=self.rounds_seen,coverage=list(self.coverage))

    def load_state_dict(self,state,full):
        if state['method']!=self.method or state['config']!=self.config:raise ValueError('Native control identity differs')
        n=state['rounds_seen'];counts=state['coverage']
        if not isinstance(n,int) or n<0 or len(counts)!=len(self.regions) or any(not isinstance(c,int) or not 0<=c<=10*n for c in counts):
            raise ValueError('Invalid cumulative coverage state')
        self.rounds_seen=n;self.coverage=list(counts)

    @torch.no_grad()
    def aggregate(self,uploads,lens,previous,dispatched=None):
        if not uploads or self.slots is None or len(self.slots)!=len(uploads) or len(lens)!=len(uploads):
            raise ValueError('Actual upload slot identities and dataset masses required')
        seen={}
        for i,(cid,p) in enumerate(self.slots):
            if not 0<=cid<100 or not 0<=p<7 or not math.isfinite(lens[i]) or lens[i]<=0:raise ValueError('Invalid slot')
            if cid in seen and seen[cid]!=lens[i]:raise ValueError('Physical dataset mass differs across repeated slots')
            seen[cid]=lens[i]
            if set(uploads[i])!=set(previous) or any(tuple(uploads[i][k].shape)!=self.shapes[p][k] for k in previous):
                raise ValueError('Upload shape differs from its actual profile')
        warmup=self.rounds_seen==0
        if self.needs_dispatch and not warmup:
            if dispatched is None or len(dispatched)!=len(uploads):raise ValueError('Actual dispatch snapshots required')
            if any(not torch.equal(s[k],previous[k][tuple(slice(0,d) for d in s[k].shape)]) for s in dispatched for k in previous):
                raise ValueError('Original post-round model synchronization was not preserved')
        native=Aggregation_AdaptiveFL(uploads,lens,previous)
        result={k:v.clone() for k,v in native.items()}
        g03={k:v.clone() for k,v in native.items()} if self.method in REFINEMENTS else None
        covered_members=[[i for i,(_,p) in enumerate(self.slots) if p in r['profiles']] for r in self.regions]
        if self.method in ('g07','g08'):
            for r,members in zip(self.regions,covered_members):self.coverage[r['id']]+=len({self.slots[i][0] for i in members})
        mean_count=sum(self.coverage)/len(self.coverage)
        records=[]
        for r,members in zip(self.regions,covered_members):
            record=dict(region_id=r['id'],numel=r['numel'],support_profiles=r['profiles'],slots=members,
                actual_profiles=sorted({self.slots[i][1] for i in members}),
                distinct_physical_clients=len({self.slots[i][0] for i in members}),
                capability_tiers=sorted({sum(self.slots[i][0]>=b for b in self.boundaries) for i in members}),
                gain=1.,weight_share_l1_change=0.,reason='first_round_native' if warmup else 'no_coverage' if not members else 'native')
            records.append(record)
            if not members:continue
            mass=sum(lens[i] for i in members);shares=[lens[i]/mass for i in members]
            record['native_slot_shares']=shares;record['applied_slot_shares']=list(shares)
            if warmup:continue
            if self.method in REFINEMENTS:
                gweights=balanced_weights('g03',members,self.slots,lens,self.boundaries)
                record['virtual_g03_slot_shares']=[w/sum(gweights) for w in gweights]
                if nonneutral_weights(gweights,members,lens):
                    write_weighted_region(g03,previous,uploads,members,gweights,r['cells'])
                active=self.method not in ('p08','p09') or (len(r['profiles'])==7 if self.method=='p08' else len(r['profiles'])!=7)
                if not active:record['reason']='outside_fixed_scope';continue
                weights=capability_weights(self.method,members,self.slots,lens,self.boundaries) if self.method in ('p03','p04','p05','p06','p07') else gweights
                cells=[c for c in r['cells'] if self.method!='p10' or c['key'] not in ('classifier.weight','classifier.bias')]
                record['classifier_native']=self.method=='p10'
                record['source_weight_scope']='nonclassifier cells only' if self.method=='p10' else 'all region cells'
                if nonneutral_weights(weights,members,lens):
                    record.update(reason='capability_refinement',applied_slot_shares=[w/sum(weights) for w in weights],
                        weight_share_l1_change=sum(abs(a-w/sum(weights)) for a,w in zip(shares,weights)))
                    write_weighted_region(result,previous,uploads,members,weights,cells)
                else:record['reason']='native_equivalent_weights'
            elif self.method in ('g01','g02','g03','g04','g05','g06'):
                active=self.method not in ('g04','g05') or (len(r['profiles'])==7 if self.method=='g04' else len(r['profiles'])!=7)
                if not active:record['reason']='outside_fixed_scope';continue
                weights=balanced_weights(self.method,members,self.slots,lens,self.boundaries)
                total=sum(weights);new_shares=[w/total for w in weights]
                # Avoid injecting roundoff as a false neutral algorithm effect.
                ratios=[w/lens[i] for w,i in zip(weights,members)]
                if len(set(ratios))==1:record['reason']='native_equivalent_weights';continue
                record.update(reason='source_balancing',applied_slot_shares=new_shares,
                    weight_share_l1_change=sum(abs(a-b) for a,b in zip(shares,new_shares)))
                write_weighted_region(result,previous,uploads,members,weights,r['cells'])
            else:
                if self.method in ('g07','g08'):
                    gain=min(1.2,max(.8,math.sqrt(mean_count/self.coverage[r['id']])))
                    record.update(reason='cumulative_coverage',cumulative_physical_coverage=self.coverage[r['id']],
                        all_region_mean_coverage=mean_count)
                else:
                    if record['distinct_physical_clients']<2:
                        gain,detail=1.,dict(reason='one_physical_client',physical_clients=1)
                    else:
                        deltas={i:torch.cat([(uploads[i][c['key']][slices(c)]-dispatched[i][c['key']][slices(c)]).reshape(-1)
                            for c in r['cells']]).double() for i in members}
                        gain,detail=reliability(deltas,members,self.slots,lens)
                        del deltas
                    record.update(detail)
                record['gain']=gain
                if gain!=1:
                    for cell in r['cells']:
                        k=cell['key'];sl=slices(cell)
                        result[k][sl]=previous[k][sl]+gain*(native[k][sl]-previous[k][sl])
        virtual_norm=update_norm(result,previous);native_norm=update_norm(native,previous)
        scale=1.
        scale_target=None
        if self.method in ('p01','p02') and not warmup:
            source,target=(native_norm,virtual_norm) if self.method=='p01' else (virtual_norm,native_norm)
            if source==0:
                if target!=0:raise FloatingPointError('Zero source direction cannot match nonzero target norm')
            else:scale=target/source
            scale_target=target
            if self.method=='p01':
                result={k:v.clone() for k,v in native.items()}
                for record in records:
                    if 'native_slot_shares' in record:
                        record['applied_slot_shares']=list(record['native_slot_shares']);record['weight_share_l1_change']=0.
            if scale!=1:
                for r,members in zip(self.regions,covered_members):
                    if members:
                        for cell in r['cells']:
                            k=cell['key'];sl=slices(cell)
                            result[k][sl]=previous[k][sl]+scale*(result[k][sl]-previous[k][sl])
        if self.method in ('g08','g10') and not warmup:
            if native_norm==0:
                if virtual_norm!=0:raise FloatingPointError('Cannot norm-match a nonzero virtual update to a zero Native direction')
            else:scale=virtual_norm/native_norm
            result={k:v.clone() for k,v in native.items()}
            if scale!=1:
                for r,members in zip(self.regions,covered_members):
                    if members:
                        for cell in r['cells']:
                            k=cell['key'];sl=slices(cell)
                            result[k][sl]=previous[k][sl]+scale*(native[k][sl]-previous[k][sl])
        changed=total=0;diff_sq=0.
        for k in previous:
            changed+=int(result[k].ne(native[k]).sum());total+=previous[k].numel()
            diff_sq+=float((result[k]-native[k]).double().square().sum())
        actual_norm=update_norm(result,previous)
        for r,record in zip(self.regions,records):
            record['actual_changed_coordinates']=sum(int(result[c['key']][slices(c)].ne(native[c['key']][slices(c)]).sum()) for c in r['cells'])
        self.rounds_seen+=1
        self.round_diagnostics=dict(mode=self.method,round=self.rounds_seen,first_round_native=warmup,regions=records,
            cumulative_physical_coverage=list(self.coverage) if self.method in ('g07','g08') else None,
            covered_parameter_fraction=sum(r['numel'] for r,m in zip(self.regions,covered_members) if m)/total,
            actual_update_l2=actual_norm,native_update_l2=native_norm,virtual_regional_update_l2=virtual_norm,
            global_scale=scale,scale_norm_absolute_error=abs(actual_norm-virtual_norm) if self.method in ('g08','g10') else None,
            scale_norm_relative_error=abs(actual_norm-virtual_norm)/virtual_norm if virtual_norm and self.method in ('g08','g10') else None,
            difference_from_native_l2=math.sqrt(diff_sq),changed_from_native_fraction=changed/total,
            actual_update_l2_ratio_native=actual_norm/native_norm if native_norm else None,
            source='Native uploads and actual server dispatches; no direction cosine, semantic score or client history')
        if g03 is not None:
            self.round_diagnostics.update(virtual_g03_update_l2=update_norm(g03,previous),
                difference_from_g03_l2=update_norm(result,g03),
                actual_update_cosine_native=norm_cosine(result,native,previous),
                actual_update_cosine_g03=norm_cosine(result,g03,previous),
                matched_norm_target=scale_target,
                matched_norm_absolute_error=abs(actual_norm-scale_target) if scale_target is not None else None,
                matched_norm_relative_error=abs(actual_norm-scale_target)/scale_target if scale_target else None,
                source='Same uploads and Full anchor; direction cosines are read-only diagnostics, never scores or gates')
        return result
