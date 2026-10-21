"""Opt-in hooks for the ORIGINAL seven-model AdapterFL loop.

No model, sampler, optimizer, test function or aggregation mapping is replaced.
The two named regions below are COVERAGE regions, not invented ResNet stages.
"""
import copy
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from models.Fed import Aggregation_AdaptiveFL, split_model


EPS = 1e-12
OPTIONS = ('coverage_gamma','coverage_eta_min','coverage_eta_max','conflict_lambda',
           'partition_alpha_shared','partition_alpha_remainder','shared_prox_lambda','shared_prox_mu',
           'partition_norm_multiplier','partition_norm_min_clients','history_beta','history_warmup_rounds')


def prefix(full, local):
    """Exact original prefix geometry, checked before using mapped deltas."""
    if full.ndim < 1 or local.ndim != full.ndim or full.dtype != torch.float32 or local.dtype != full.dtype:
        raise ValueError('These hooks support the original float32 ResNet affine state only')
    if any(a > b or a < 1 for a,b in zip(local.shape,full.shape)) or local.shape[2:] != full.shape[2:]:
        raise ValueError('Invalid original prefix mapping')
    return tuple(slice(0,d) for d in local.shape)


def stats(values):
    values = [float(x) for x in values]
    return {'count':len(values),'min':min(values),'median':statistics.median(values),
            'mean':statistics.mean(values),'max':max(values)} if values else {'count':0}


def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


class MethodHooks:
    """One strategy switch; only M6 owns dense history, M3 owns two scalars."""
    def __init__(self, args, models):
        self.method = args.method
        if self.method not in ('baseline','m1','m2','m3','m4','m5','m6'):
            raise ValueError('Choose one method; combinations are unsupported')
        self.config = {name:getattr(args,name) for name in OPTIONS}
        c = self.config
        if any(not math.isfinite(float(v)) for v in c.values() if v is not None):
            raise ValueError('Method options must be finite')
        if (c['coverage_gamma'] < 0 or not 0 < c['coverage_eta_min'] <= 1 <= c['coverage_eta_max']
                or min(c['conflict_lambda'],c['shared_prox_mu'],c['partition_alpha_shared'],c['partition_alpha_remainder']) < 0
                or (c['shared_prox_lambda'] is not None and c['shared_prox_lambda']<0)
                or c['partition_norm_multiplier'] <= 0 or c['partition_norm_min_clients'] < 2
                or not 0 <= c['history_beta'] < 1 or not isinstance(c['history_warmup_rounds'],int)
                or c['history_warmup_rounds']<0):
            raise ValueError('Invalid method option range')
        states = [m.state_dict() for m in models]
        full = states[-1]
        if len(states) != 7 or any(s.keys()!=full.keys() for s in states):
            raise ValueError('Methods require the unchanged seven-model configuration')
        self.shared_shapes = {}
        for key,old in full.items():
            for s in states:
                prefix(old,s[key])
            shape = tuple(min(s[key].shape[d] for s in states) for d in range(old.ndim))
            self.shared_shapes[key] = shape
        self.velocity, self.previous_active = {}, {}
        self.previous_norms = {}
        self.diagnostic_state = {}
        self.history_rounds_seen = 0
        self.round_diagnostics = {}

    def state_dict(self):
        return cpu_copy({'method':self.method,'config':self.config,'shared_shapes':self.shared_shapes,
                         'velocity':self.velocity,'previous_active':self.previous_active,
                         'previous_norms':self.previous_norms,'diagnostic_state':self.diagnostic_state,
                         'history_rounds_seen':self.history_rounds_seen})

    def load_state_dict(self, state, full):
        if (state['method'] != self.method or state['config'] != self.config
                or state['shared_shapes'] != self.shared_shapes):
            raise ValueError('Checkpoint method/configuration/coverage differs')
        for attr in ('velocity','previous_active'):
            values = state[attr]
            if values and values.keys() != full.keys():
                raise ValueError('Incomplete history checkpoint')
            restored = {}
            for k,v in values.items():
                if v.shape != full[k].shape or not torch.isfinite(v).all():
                    raise ValueError(f'Invalid history tensor: {k}')
                expected = torch.bool if attr=='previous_active' else full[k].dtype
                if v.dtype != expected:
                    raise ValueError('Invalid history dtype')
                restored[k] = v.to(full[k].device).clone()
            setattr(self,attr,restored)
        if bool(self.velocity) != bool(self.previous_active):
            raise ValueError('Velocity and active-mask history must be paired')
        self.previous_norms = dict(state['previous_norms'])
        self.diagnostic_state = copy.deepcopy(state.get('diagnostic_state',{}))
        self.history_rounds_seen = state['history_rounds_seen']
        if not isinstance(self.history_rounds_seen,int) or self.history_rounds_seen<0:
            raise ValueError('Invalid warmup progress')
        if self.velocity and self.history_rounds_seen<=self.config['history_warmup_rounds']:
            raise ValueError('Warmup checkpoint cannot contain momentum history')

    def region(self, key, tensor, name):
        mask = torch.zeros_like(tensor,dtype=torch.bool)
        sl = tuple(slice(0,min(a,b)) for a,b in zip(tensor.shape,self.shared_shapes[key]))
        mask[sl] = True
        return mask if name=='shared' else ~mask

    def configure_local(self, local):
        legacy=self.config['shared_prox_lambda']
        strength=legacy if legacy is not None else self.config['shared_prox_mu']
        if self.method=='m4' and strength>0:
            local.adapterfl_prox = (self.shared_shapes,strength,'mean' if legacy is not None else 'sum')
            local.adapterfl_prox_diagnostic_steps = 2

    @torch.no_grad()
    def aggregate(self, uploads, lens, previous, dispatched=None):
        self.round_diagnostics = {}
        if self.method=='baseline':
            return Aggregation_AdaptiveFL(uploads,lens,previous)
        if not uploads or len(uploads)!=len(lens) or any(n<=0 for n in lens):
            raise ValueError('Missing uploads or sample counts')
        for s in uploads:
            if s.keys()!=previous.keys():
                raise ValueError('Upload state keys differ')
            for k,v in previous.items():
                prefix(v,s[k])
                if not torch.isfinite(s[k]).all():
                    raise FloatingPointError(f'Nonfinite upload: {k}')
        if self.method in ('m2','m5'):
            if dispatched is None or len(dispatched)!=len(uploads):
                raise ValueError('Actual dispatched snapshots are required')
            for sent,local in zip(dispatched,uploads):
                if sent.keys()!=local.keys() or any(sent[k].shape!=local[k].shape for k in local):
                    raise ValueError('Dispatch/upload identity or shape differs')
            if self.method=='m2':
                return self.conflict(uploads,lens,previous,dispatched)
            return self.norm_control(uploads,lens,previous,dispatched)
        base = Aggregation_AdaptiveFL(uploads,lens,previous)
        if self.method=='m4':
            return base
        counts = {}
        for key,old in previous.items():
            count = torch.zeros_like(old)
            for s in uploads:
                count[prefix(old,s[key])] += 1  # SLOTS, not a second sample weighting.
            counts[key] = count
        if self.method=='m1':
            result = {}
            for k,old in previous.items():
                covered = counts[k]>0
                reference = counts[k][covered].mean()
                eta = (reference/(counts[k][covered]+EPS)).pow(self.config['coverage_gamma']).clamp(
                    self.config['coverage_eta_min'],self.config['coverage_eta_max'])
                result[k] = base[k].clone()
                if not torch.all(eta==1):
                    result[k][covered] = old[covered]+eta*(base[k][covered]-old[covered])
                coverage_histogram = {str(int(n)):int((counts[k]==n).sum()) for n in counts[k].unique().tolist()}
                eta_histogram = {}
                for n,number in coverage_histogram.items():
                    if int(n):
                        coefficient = min(self.config['coverage_eta_max'],max(self.config['coverage_eta_min'],
                            (float(reference)/(int(n)+EPS))**self.config['coverage_gamma']))
                        label = f'{coefficient:.8g}'
                        eta_histogram[label] = eta_histogram.get(label,0)+number
                self.round_diagnostics[k] = {'coverage_reference':float(reference),
                    'difference_from_native_aggregate_l2':float((result[k]-base[k]).double().square().sum().sqrt()),
                    'covered_fraction':float(covered.float().mean()),
                    'slot_coverage':{'min':float(counts[k][covered].min()),'mean':float(reference),
                                     'max':float(counts[k].max())},
                    'coverage_histogram':coverage_histogram,
                    'eta':{'min':float(eta.min()),'mean':float(eta.mean()),'max':float(eta.max()),'histogram':eta_histogram}}
            self.round_diagnostics['__summary__']={'difference_from_native_aggregate_l2':math.sqrt(sum(
                d['difference_from_native_aggregate_l2']**2 for d in self.round_diagnostics.values()))}
            return result
        if self.method=='m3':
            result = {k:v.clone() for k,v in base.items()}
            for region in ('shared','remainder'):
                alpha = self.config[f'partition_alpha_{region}']
                square, applied_square = 0.,0.
                for k,old in previous.items():
                    mask = self.region(k,old,region) & (counts[k]>0)
                    delta = base[k][mask]-old[mask]
                    square += float(delta.double().square().sum())
                    if alpha != 1:
                        result[k][mask] = old[mask]+alpha*delta
                    applied_square += float((result[k][mask]-old[mask]).double().square().sum())
                norm = math.sqrt(square)
                last = self.previous_norms.get(region)
                self.round_diagnostics[region] = {'alpha':alpha,'base_update_l2':norm,
                    'applied_update_l2':math.sqrt(applied_square),
                    'relative_norm_fluctuation':abs(norm-last)/max(last,EPS) if last is not None else None}
                self.previous_norms[region] = norm
            return result
        if self.method!='m6':
            raise ValueError('Unknown method; combinations are unsupported')
        raw = math.sqrt(sum(float((base[k]-v).double().square().sum()) for k,v in previous.items()))
        self.history_rounds_seen += 1
        self.diagnostic_state['m6_rounds'] = self.history_rounds_seen
        self.diagnostic_state.setdefault('m6_initial_update_l2',raw)
        warmup = self.config['history_warmup_rounds']
        if self.history_rounds_seen<=warmup:
            # Never retain the independent-initialization reconciliation in EMA.
            # Returning base itself preserves native float32 aggregation exactly.
            self.velocity,self.previous_active = {},{}
            self.round_diagnostics = {'phase':'warmup_passthrough','rounds_seen':self.history_rounds_seen,
                'warmup_rounds':warmup,'warmup_complete':self.history_rounds_seen>=warmup,
                'raw_update_l2':raw,'smoothed_update_l2':raw,'history_component_l2':0.,
                'current_component_l2':raw,'history_to_current_component_ratio':0. if raw>EPS else None,
                'applied_to_raw_update_ratio':1. if raw>EPS else None,'difference_from_native_aggregate_l2':0.,
                'first_round_initialization_delta_l2':self.diagnostic_state['m6_initial_update_l2'],
                'history_initialized':False,'effective_history_initialization_delta_l2':None,
                'covered_coordinates':sum(int((c>0).sum()) for c in counts.values()),
                'beta':self.config['history_beta'],'initialization_note':'Native initialization delta excluded from all future velocity; metadata only'}
            return base
        result, dot, old_square, current_square, applied_square = {},0.,0.,0.,0.
        reactivated, covered_count, consecutive_count = 0,0,0
        history_square, current_component_square, fresh_square, effect_square = 0.,0.,0.,0.
        first_observation = not self.velocity
        beta = self.config['history_beta']
        for k,old in previous.items():
            active = counts[k]>0
            prior = self.previous_active.get(k,torch.zeros_like(active))
            both = active & prior
            consecutive_count += int(both.sum())
            delta = base[k]-old
            velocity = torch.zeros_like(old)
            velocity[active] = delta[active]  # First observation or after a gap.
            fresh_square += float(delta[active & ~prior].double().square().sum())
            current_component_square += float(delta[active & ~prior].double().square().sum())
            if k in self.velocity:
                prev = self.velocity[k]
                dot += float((delta[both].double()*prev[both].double()).sum())
                old_square += float(prev[both].double().square().sum())
                current_square += float(delta[both].double().square().sum())
                if beta:
                    velocity[both] = beta*prev[both]+(1-beta)*delta[both]
                history_square += float((beta*prev[both]).double().square().sum())
            current_component_square += float(((1-beta)*delta[both]).double().square().sum())
            result[k] = base[k].clone()
            if beta and torch.any(both):
                result[k][both] = old[both]+velocity[both]
            result[k][~active] = old[~active]  # No historical drift without coverage.
            self.velocity[k],self.previous_active[k] = velocity,active.clone()
            covered_count += int(active.sum()); reactivated += int((active & ~prior).sum())
            applied_square += float(velocity.double().square().sum())
            effect_square += float((result[k]-base[k]).double().square().sum())
        if first_observation:
            self.diagnostic_state['m6_first_effective_update_l2'] = raw
        current_component_norm = math.sqrt(current_component_square)
        self.round_diagnostics = {'phase':'history_initialization' if first_observation else 'history_smoothing',
            'rounds_seen':self.history_rounds_seen,'warmup_rounds':warmup,'warmup_complete':True,
            'history_initialized':True,'effective_history_initialization_delta_l2':self.diagnostic_state['m6_first_effective_update_l2'],
            'raw_update_l2':raw,'smoothed_update_l2':math.sqrt(applied_square),
            'delta_vs_previous_velocity_cosine':dot/math.sqrt(old_square*current_square) if min(old_square,current_square)>EPS**2 else None,
            'fresh_or_reactivated_coordinates':reactivated,'covered_coordinates':covered_count,'beta':beta,
            'first_observation':first_observation,'first_round_initialization_delta_l2':self.diagnostic_state['m6_initial_update_l2'],
            'fresh_or_reactivated_delta_l2':math.sqrt(fresh_square),'consecutive_covered_coordinates':consecutive_count,
            'history_component_l2':math.sqrt(history_square),'current_component_l2':current_component_norm,
            'history_to_current_component_ratio':math.sqrt(history_square)/current_component_norm if current_component_norm>EPS else None,
            'applied_to_raw_update_ratio':math.sqrt(applied_square)/raw if raw>EPS else None,
            'difference_from_native_aggregate_l2':math.sqrt(effect_square),
            'initialization_note':'Velocity starts after configured warmup; gaps reset covered coordinates to current delta'}
        return result

    def conflict(self, uploads, lens, previous, dispatched):
        strength = self.config['conflict_lambda']
        geometry = conflict_support_diagnostics(uploads,previous)
        if len(uploads)<2 or strength==0:
            self.round_diagnostics = dict(geometry,fallback='single_slot_or_zero_strength',
                                         conflict_weights=[1.]*len(uploads),weight_distribution=stats([1.]*len(uploads)))
            return Aggregation_AdaptiveFL(uploads,lens,previous)
        # For each key, intersect ALL currently uploaded supports; align the same
        # semantic coordinates before accumulating global dot products/norms.
        slices = {k:tuple(slice(0,min(s[k].shape[d] for s in uploads)) for d in range(v.ndim)) for k,v in previous.items()}
        deltas = [{k:(s[k][slices[k]]-sent[k][slices[k]]).double() for k in previous}
                  for s,sent in zip(uploads,dispatched)]
        total = {k:sum(n*d[k] for n,d in zip(lens,deltas)) for k in previous}
        similarities, logits = [],[]
        for i,d in enumerate(deltas):
            numerator,ds,rs = 0.,0.,0.
            for k,v in d.items():
                # Leave-one-out reference avoids a client's own norm dominating.
                ref = (total[k]-lens[i]*v)/(sum(lens)-lens[i])
                numerator += float((v*ref).sum()); ds += float(v.square().sum()); rs += float(ref.square().sum())
            similarity = max(-1.,min(1.,numerator/math.sqrt(ds*rs))) if min(ds,rs)>EPS**2 else None
            similarities.append(similarity)
            logits.append(-strength*max(0.,-similarity) if similarity is not None else 0.)
        # A common multiplier cancels in coordinate-wise weight normalization.
        # Subtract the maximum to avoid all sample weights underflowing to zero.
        weights=[math.exp(max(-80.,x-max(logits))) for x in logits]
        self.round_diagnostics = dict(geometry,cosine=similarities,conflict_weights=weights,
            weight_distribution=dict(stats(weights),std_population=statistics.pstdev(weights)),
            valid_cosine_negative_fraction=sum(s is not None and s<0 for s in similarities)/max(1,sum(s is not None for s in similarities)),
            unnormalized_log_weights=logits,
            negative_fraction=sum(s is not None and s<0 for s in similarities)/len(similarities),
            fallback_slots=sum(s is None for s in similarities))
        if all(w==1 for w in weights):
            return Aggregation_AdaptiveFL(uploads,lens,previous)
        return Aggregation_AdaptiveFL(uploads,[n*w for n,w in zip(lens,weights)],previous)

    def norm_control(self, uploads, lens, previous, dispatched):
        changed = list(uploads)
        diagnostics = []
        for region in ('shared','remainder'):
            groups = {}
            for i,s in enumerate(uploads):
                # Compare exactly matching parameter subsets, not unrelated model
                # sizes. Shared support is identical across every profile.
                signature = tuple((k,tuple(min(a,b) for a,b in zip(v.shape,self.shared_shapes[k]))
                                   if region=='shared' else tuple(v.shape)) for k,v in s.items())
                groups.setdefault(signature,[]).append(i)
            for group in groups.values():
                signature = tuple((k,tuple(min(a,b) for a,b in zip(v.shape,self.shared_shapes[k]))
                                   if region=='shared' else tuple(v.shape)) for k,v in uploads[group[0]].items())
                support_id = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
                target_coordinates = sum(int(self.region(k,v,region).sum()) for k,v in uploads[group[0]].items())
                norms=[]
                for i in group:
                    square=0.
                    for k,v in uploads[i].items():
                        mask=self.region(k,v,region)
                        square+=float((v[mask]-dispatched[i][k][mask]).double().square().sum())
                    norms.append(math.sqrt(square))
                enough=len(group)>=self.config['partition_norm_min_clients']
                threshold=statistics.median(norms)*self.config['partition_norm_multiplier'] if enough else None
                factors=[min(1.,threshold/(n+EPS)) if threshold is not None and n>EPS and n>threshold else 1. for n in norms]
                actual_changes, adjustment_square, actual_slot_ids = 0,0.,[]
                for i,factor in zip(group,factors):
                    if factor==1:
                        continue
                    if changed[i] is uploads[i]:
                        changed[i]={k:v.clone() for k,v in uploads[i].items()}
                    changed_coordinates = 0
                    for k,v in uploads[i].items():
                        mask=self.region(k,v,region)
                        sent=dispatched[i][k]
                        changed[i][k][mask]=sent[mask]+factor*(v[mask]-sent[mask])
                        difference=changed[i][k][mask]-v[mask]
                        adjustment_square+=float(difference.double().square().sum())
                        changed_coordinates+=int(torch.count_nonzero(difference))
                    actual_changes+=changed_coordinates>0
                    if changed_coordinates:
                        actual_slot_ids.append(i)
                diagnostics.append({'partition':region,'slots':group,'norms_before':norms,
                    'median_norm':statistics.median(norms),'max_norm':max(norms),
                    'max_to_median_norm_ratio':max(norms)/statistics.median(norms) if statistics.median(norms)>EPS else None,
                    'nonactivation_reason':('empty_region' if not target_coordinates else 'insufficient_equal_support_slots'
                        if not enough else 'within_fixed_threshold' if not actual_changes else None),
                    'threshold':threshold,'factors':factors,'clipped_fraction':sum(f<1 for f in factors)/len(factors),
                    'norms_after':[n*f for n,f in zip(norms,factors)],'fallback':None if enough else 'insufficient_equal_support_slots',
                    'support_id':support_id,'slot_count':len(group),'target_coordinates_per_slot':target_coordinates,
                    'clip_triggered_slots':sum(f<1 for f in factors),'actually_changed_slots':actual_changes,
                    'actually_changed_slot_ids':actual_slot_ids,
                    'upload_adjustment_l2':math.sqrt(adjustment_square)})
        totals = {}
        for region in ('shared','remainder'):
            groups=[g for g in diagnostics if g['partition']==region and g['target_coordinates_per_slot']>0]
            slots=sum(g['slot_count'] for g in groups)
            fallback=sum(g['slot_count'] for g in groups if g['fallback'])
            clipped=sum(g['actually_changed_slots'] for g in groups)
            totals[region]={'nonempty_slot_regions':slots,'actually_changed_slot_regions':clipped,
                'actual_clip_rate':clipped/slots if slots else 0.,
                'insufficient_slots_fallback_count':fallback,'insufficient_slots_fallback_fraction':fallback/slots if slots else 0.,
                'group_slot_counts':[g['slot_count'] for g in groups],
                'upload_adjustment_l2':math.sqrt(sum(g['upload_adjustment_l2']**2 for g in groups))}
        changed_slots={i for g in diagnostics for i in g['actually_changed_slot_ids']}
        rounds=self.diagnostic_state.get('m5_rounds',0)+1
        idle=0 if changed_slots else self.diagnostic_state.get('m5_rounds_without_actual_clip',0)+1
        self.diagnostic_state.update(m5_rounds=rounds,m5_rounds_without_actual_clip=idle)
        self.diagnostic_state['m5_rounds_with_actual_clip']=self.diagnostic_state.get('m5_rounds_with_actual_clip',0)+bool(changed_slots)
        warnings=[] if changed_slots else ['NO_ACTUAL_CLIPPING_THIS_ROUND']
        if idle>=3:
            warnings.append('NO_ACTUAL_CLIPPING_FOR_AT_LEAST_3_ROUNDS')
        self.round_diagnostics={'groups':diagnostics,'partition_totals':totals,
            'actual_changed_upload_slot_fraction':len(changed_slots)/len(uploads),
            'rounds_without_actual_clip':idle,'rounds_with_actual_clip':self.diagnostic_state['m5_rounds_with_actual_clip'],
            'warning_codes':warnings}
        return Aggregation_AdaptiveFL(changed,lens,previous)


def conflict_support_diagnostics(uploads,previous):
    """Geometry only: a single-holder weight cancels in native normalization."""
    full=sum(v.numel() for v in previous.values())
    common,exclusive,uncovered=0,0,0
    exclusive_per_slot=[0]*len(uploads)
    for k,v in previous.items():
        shape=tuple(min(s[k].shape[d] for s in uploads) for d in range(v.ndim))
        common+=math.prod(shape)
        counts=torch.zeros_like(v,dtype=torch.int32)
        for s in uploads:
            counts[prefix(v,s[k])]+=1
        exclusive+=int((counts==1).sum());uncovered+=int((counts==0).sum())
        for i,s in enumerate(uploads):
            exclusive_per_slot[i]+=int((counts[prefix(v,s[k])]==1).sum())
    return {'common_coordinates':common,'full_coordinates':full,'common_support_fraction_full':common/full,
            'common_support_fraction_per_upload':[common/sum(v.numel() for v in s.values()) for s in uploads],
            'exclusive_coordinates':exclusive,'exclusive_coordinates_per_slot':exclusive_per_slot,
            'uncovered_coordinates':uncovered,
            'exclusive_region_weight_effect':'One-holder normalized weight is 1: conflict multiplier cancels; no exclusive damping is claimed'}


def prepare_prox(net, setup):
    """Detached actual-dispatch snapshot of shared PARAMETER prefixes only."""
    shapes,strength=setup[:2]
    if len(setup)>2 and setup[2] not in ('mean','sum'):
        raise ValueError('Unknown proximal normalization')
    targets={}
    for k,p in net.named_parameters():
        sl=tuple(slice(0,d) for d in shapes[k])
        targets[k]=(sl,p[sl].detach().clone())
    return targets,sum(v.numel() for _,v in targets.values()),strength


def prox_loss(net, targets, count, normalization='mean'):
    result=next(net.parameters()).new_zeros(())
    for k,p in net.named_parameters():
        sl,initial=targets[k]
        result=result+(p[sl]-initial).square().sum()
    return .5*result/(count if normalization=='mean' else 1)


def prox_gradient_diagnostics(net,task_loss,regularizer,targets,count,strength,normalization='mean'):
    """Sampled diagnostic only: autograd.grad does not populate parameter .grad.

    Analytic gradient is mu*drift (sum) or legacy lambda*drift/N (mean).
    No second forward, optimizer step, random draw or automatic lambda change.
    """
    params=list(net.named_parameters())
    denominator=count if normalization=='mean' else 1
    gradients=torch.autograd.grad(task_loss,[p for _,p in params],retain_graph=True,allow_unused=True)
    task_square,task_shared_square,prox_square=0.,0.,0.
    for (key,p),g in zip(params,gradients):
        sl,initial=targets[key]
        if g is not None:
            task_square+=float(g.detach().double().square().sum())
            task_shared_square+=float(g[sl].detach().double().square().sum())
        prox_square+=float((p[sl].detach().double()-initial.double()).square().sum())/denominator**2
    task_shared=math.sqrt(task_shared_square)
    proximal=math.sqrt(prox_square)
    ce=float(task_loss.detach());penalty=float(regularizer.detach())
    return {'task_ce':ce,'prox_loss':penalty,'weighted_prox_loss':strength*penalty,
            'prox_to_task_loss_ratio':penalty/max(abs(ce),EPS),
            'weighted_prox_to_task_loss_ratio':strength*penalty/max(abs(ce),EPS),
            'task_gradient_l2':math.sqrt(task_square),'task_shared_gradient_l2':task_shared,
            'proximal_gradient_l2':proximal,'weighted_proximal_gradient_l2':strength*proximal,
            'weighted_prox_to_task_shared_gradient_ratio':strength*proximal/task_shared if task_shared>EPS else None,
            'lambda_for_reference_1percent_shared_gradient_ratio':.01*task_shared/proximal if proximal>EPS else None,
            'shared_parameters':count,'prox_normalization':normalization,
            'equivalent_unnormalized_prox_coefficient':strength/denominator,
            'calibration_note':'1 percent is a dimensional reference only; no lambda is changed'}


def prepare_output(args):
    if not getattr(args,'method',None):
        return None
    if getattr(args,'_method_run_path',None):
        return Path(args._method_run_path)
    if args.method_resume:
        path=Path(args.method_resume).resolve().parent
        if path.parent.name!=args.method:
            raise ValueError('Resume path must belong to the selected method')
    else:
        name=args.method_run_name or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        if Path(name).name!=name or name in ('.','..') or '\\' in name:
            raise ValueError('Invalid method run name')
        path=Path(args.method_output_root)/args.method/name
        path.mkdir(parents=True,exist_ok=False)
    args._method_run_path=str(path)
    return path


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


@contextmanager
def isolated_evaluation_rng():
    """DataLoader iteration can draw a base seed even with shuffle=False."""
    before = rng_state()
    try:
        yield
    finally:
        restore_rng(before)


class RunLog:
    """Journal/checkpoint hook called by the native loop, not a training engine."""
    def __init__(self,args,models):
        self.args=args
        self.save_pt=not getattr(args,'no_save_pt',False)
        if not self.save_pt and args.method_resume:
            raise ValueError('Checkpoint-free runs must start fresh; resume is disabled')
        self.path=prepare_output(args)
        if getattr(args,'_final40_context',None):
            from capability_final40 import Final40Hooks
            self.hooks=Final40Hooks(args,models)
        elif args.method.startswith('a'):
            raise ValueError('A01-A20 require the fixed final40 prepared-condition entry')
        elif args.method in [f'g{i:02d}' for i in range(1,11)]+[f'p{i:02d}' for i in range(1,11)]:
            from native_group_aggregation import NativeGroupHooks
            self.hooks=NativeGroupHooks(args,models)
        elif args.method=='m2' and getattr(args,'_local12_reference',False):
            from local12_aggregation import LocalReferenceHooks
            self.hooks=LocalReferenceHooks(args,models)
        elif args.method in [f'l{i:02d}' for i in range(1,12)]:
            from local12_aggregation import Local12Hooks
            self.hooks=Local12Hooks(args,models)
        elif args.method in [f't{i:02d}' for i in range(1,11)]:
            from tonight_aggregation import TonightHooks
            self.hooks=TonightHooks(args,models)
        elif args.method in ('c1','c2'):
            from evidence_aggregation import EvidenceHooks
            self.hooks=EvidenceHooks(args,models)
        elif args.method.startswith('s'):
            from semcons_scope import ScopeHooks
            self.hooks=ScopeHooks(args,models)
        else:
            self.hooks=MethodHooks(args,models)
        self.rows=[]; self.next_round=0; self.total_time=0.
        self.started=time.perf_counter()
        self.elapsed_before=0.
        self.eval_profiles=getattr(args,'eval_profiles','all')
        self.evaluated_indices=[6] if self.eval_profiles=='full' else list(range(7))
        self.eval_rng_policy='isolated' if self.eval_profiles=='full' or getattr(args,'isolate_eval_rng',False) else 'legacy'
        if args.client_chosen_mode!='available':
            raise ValueError('Reviewed experiments use original available scheduling only')
        if args.model!='resnet' or args.width_ration!=[.4,.66,1.] or args.depth_saved!=[2,3,4]:
            raise ValueError('Keep original seven ResNet profiles')
        self.identity={k:getattr(args,k) for k in ('seed','epochs','num_users','frac','local_ep','local_bs','bs','lr',
            'lr_decay','momentum','weight_decay','optimizer','dataset','model','iid','data_beta','client_hetero_ration','client_chosen_mode')}
        self.identity['method']=args.method; self.identity['options']=self.hooks.config
        self.identity['eval_profiles']=self.eval_profiles
        self.identity['eval_rng_policy']=self.eval_rng_policy
        self.identity['logging_schema']=2
        if getattr(args,'deterministic_execution',False):
            from utils.reproducibility import execution_policy
            self.identity['execution_policy']=execution_policy()
        self.identity['partition_sha256']=getattr(args,'_method_partition_sha256',None)
        if getattr(args,'_final40_context',None):
            self.identity['num_classes']=args.num_classes
            self.identity['condition']=args._final40_context['condition']
        files=('main_fed_ori.py','Algorithm/Training_AdaptiveFL.py','models/Update.py','models/Fed.py','models/resnet.py','models/test.py','utils/options.py','adapterfl_methods.py','utils/get_datatset_new.py','utils/set_seed.py')
        if getattr(self.hooks,'is_scope',False):files=(*files,'semcons_scope.py')
        if getattr(self.hooks,'is_evidence',False):files=(*files,'evidence_aggregation.py','tools/run_evidence_candidate.py','configs/e03_scope_components400.json')
        if getattr(self.hooks,'is_tonight',False):files=(*files,'tonight_aggregation.py','tools/run_evidence_candidate.py','configs/e03_tonight10_400.json')
        if getattr(self.hooks,'is_local12',False):files=(*files,'local12_aggregation.py','tonight_aggregation.py','tools/run_evidence_candidate.py','configs/e03_local12_400.json')
        if getattr(args,'deterministic_execution',False):files=(*files,'utils/reproducibility.py')
        if getattr(self.hooks,'is_native_groups',False):
            files=(*files,'native_group_aggregation.py','semcons_scope.py','tools/run_evidence_candidate.py',
                'configs/g03_refinements10_400.json' if args.method.startswith('p') else 'configs/native_groups10_400.json')
        if getattr(args,'_final40_context',None):
            files=(*files,'capability_final40.py','tools/run_final40_job.py','tools/final40_data.py','configs/capability_final40_400.json',
                   'configs/native_groups10_400.json','configs/g03_refinements10_400.json')
        self.identity['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
        if args.method_resume:
            # Local, user-selected research checkpoint includes RNG tuples.
            state=torch.load(args.method_resume,map_location='cpu',weights_only=False)
            if state['identity']!=self.identity:
                raise ValueError('Checkpoint source, method, data identity or training options differ')
            self.hooks.load_state_dict(state['method_state'],models[-1].state_dict())
            for model in models:
                device=next(model.parameters()).device
                full={k:v.to(device) for k,v in state['global'].items()}
                model.load_state_dict(split_model(full,model.state_dict()))
            self.rows=state['rows']; self.next_round=state['next_round']; self.total_time=state['total_time']
            self.elapsed_before=state['elapsed_seconds']
            csv_path=self.path/'accuracy_each_round.csv'
            with csv_path.open(encoding='utf-8') as f:
                existing=list(csv.DictReader(f))
            with (self.path/'method_diagnostics.csv').open(encoding='utf-8') as f:
                diagnostics=list(csv.DictReader(f))
            if len(existing)!=self.next_round or len(diagnostics)!=self.next_round:
                raise ValueError('Only aligned round-boundary checkpoints can resume; journal differs')
            if self.rows:
                best=torch.load(self.path/'best_full.pt',map_location='cpu',weights_only=False)
                peak=max(self.rows,key=lambda r:r['full_accuracy'])
                if best['identity']!=self.identity or best['round']!=peak['round'] or best['full_accuracy']!=peak['full_accuracy']:
                    raise ValueError('Peak snapshot and latest checkpoint disagree')
            restore_rng(state['rng'])
        else:
            manifest={'identity':self.identity,'command':sys.argv,'torch':torch.__version__,
                      'python':sys.version,'device':str(args.device),'profiles':[(w,d) for w in args.width_ration for d in args.depth_saved if w!=1 or d==args.depth_saved[0]],
                      'initial_state_sha256':[hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes() for k,v in m.state_dict().items())).hexdigest() for m in models],
                      'partition_definition':'intersection of all seven actual parameter prefixes vs complement; no original fixed two-part topology',
                      'evaluated_profiles':[f'P{i}' for i in self.evaluated_indices],
                      'cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                      'training_configuration':{k:str(v) if isinstance(v,(torch.device,Path)) else v for k,v in vars(args).items() if not k.startswith('_')},
                      'rng_compatibility':'isolated evaluation changes the legacy training RNG trajectory; not bitwise identical to historical 300-round runs' if self.eval_rng_policy=='isolated' else 'legacy evaluation RNG consumption',
                      'save_pt':self.save_pt,
                      'checkpoint_selection':'latest.pt for resume; best_full.pt is an exploratory official-test-oracle selection, not validation-selected' if self.save_pt else 'No checkpoint files by user request; test-oracle peak recorded as scalar only'}
            if getattr(self.hooks,'is_scope',False):
                manifest['structural_regions']=self.hooks.regions
                manifest['scope_geometry_sha256']=self.hooks.geometry_hash
            if getattr(self.hooks,'is_native_groups',False):manifest['structural_regions']=self.hooks.regions
            (self.path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        self.accuracy=(self.path/'accuracy_each_round.csv').open('a' if args.method_resume else 'x',newline='',encoding='utf-8')
        self.diagnostics=(self.path/'method_diagnostics.csv').open('a' if args.method_resume else 'x',newline='',encoding='utf-8')
        self.accuracy_writer=None
        self.diagnostic_writer=csv.DictWriter(self.diagnostics,fieldnames=['round','method','metrics_json'])
        if not args.method_resume:
            self.diagnostic_writer.writeheader()

    def record(self,round_index,models,evaluations,max_time,total_time,wall,server_seconds,eval_seconds,slots,local_logs):
        scores=evaluations if isinstance(evaluations,dict) else dict(enumerate(evaluations))
        if set(scores)!=set(self.evaluated_indices):
            raise ValueError('Evaluation profile IDs do not match configured logging mode')
        row={'round':round_index+1,'full_accuracy':scores[6][0],'full_ce':scores[6][1],
             'original_max_client_seconds':max_time,'original_cumulative_max_client_seconds':total_time,
             'round_wall_seconds':wall,'server_seconds':server_seconds,'evaluation_seconds':eval_seconds}
        peak=max(self.rows,key=lambda r:r['full_accuracy']) if self.rows else row
        if row['full_accuracy']>peak['full_accuracy']:
            peak=row
        row['best_full_accuracy']=peak['full_accuracy']
        row['best_round']=peak['round']
        for i,(acc,ce) in scores.items():
            row[f'P{i}_accuracy']=acc; row[f'P{i}_ce']=ce
        if self.accuracy_writer is None:
            self.accuracy_writer=csv.DictWriter(self.accuracy,fieldnames=list(row))
            if not self.args.method_resume:
                self.accuracy_writer.writeheader()
        self.accuracy_writer.writerow(row);self.accuracy.flush()
        metrics={'server':self.hooks.round_diagnostics,'clients':local_logs,'slots':slots}
        for warning in self.hooks.round_diagnostics.get('warning_codes',[]):
            print(f'METHOD_DIAGNOSTIC_WARNING method={self.args.method} round={round_index+1} code={warning}',flush=True)
        self.diagnostic_writer.writerow({'round':round_index+1,'method':self.args.method,'metrics_json':json.dumps(metrics,allow_nan=False)})
        self.diagnostics.flush()
        is_best=not self.rows or row['full_accuracy']>max(r['full_accuracy'] for r in self.rows)
        self.rows.append(row);self.total_time=total_time;self.next_round=round_index+1
        if not self.save_pt:
            if self.next_round==self.args.epochs:
                full=models[-1].state_dict()
                check=dict(round=self.next_round,full_numel=sum(v.numel() for v in full.values()),
                    full_shapes={k:list(v.shape) for k,v in full.items()},
                    finite_full_parameters=all(bool(torch.isfinite(v).all()) for v in full.values()),
                    source='In-memory final Full state; no tensors or checkpoint saved')
                (self.path/'final_model_check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
                if not check['finite_full_parameters']:raise FloatingPointError('Nonfinite final Full state')
            self.write_summary()
            return
        checkpoint={'identity':self.identity,'next_round':self.next_round,'total_time':total_time,
                    'elapsed_seconds':self.elapsed_before+time.perf_counter()-self.started,'rows':self.rows,
                    'global':cpu_copy(models[-1].state_dict()),'method_state':self.hooks.state_dict(),'rng':rng_state()}
        if is_best:
            best={'identity':self.identity,'round':round_index+1,'full_accuracy':row['full_accuracy'],
                  'full_ce':row['full_ce'],'global':checkpoint['global'],
                  'selection':'test-oracle peak on official test set; exploratory only, not an unbiased validation-selected model'}
            temp=self.path/'best_full.pending'
            torch.save(best,temp);temp.replace(self.path/'best_full.pt')
        temp=self.path/'latest.pending'
        torch.save(checkpoint,temp); temp.replace(self.path/'latest.pt')
        self.write_summary()

    def write_summary(self):
        if not self.rows:
            return
        values=[r['full_accuracy'] for r in self.rows]
        best=max(self.rows,key=lambda r:r['full_accuracy'])
        summary={'method':self.args.method,'configuration':self.identity,'rounds_completed':len(values),
                 'requested_epochs':self.args.epochs,'best_full_accuracy':best['full_accuracy'],
                 'best_round':best['round'],'peak_label':'test-oracle peak',
                 'final_full_accuracy':values[-1],'final_full_ce':self.rows[-1]['full_ce'],
                 'last20_full_mean':statistics.mean(values[-20:]),
                 'last20_full_std':statistics.pstdev(values[-20:]),'last20_count':len(values[-20:]),
                 'last50_full_mean':statistics.mean(values[-50:]),
                 'elapsed_seconds':self.elapsed_before+time.perf_counter()-self.started,
                 'selection':'best_full.pt is exploratory official-test selection; latest.pt is the resume checkpoint' if self.save_pt else 'Checkpoint files disabled by user; scalar test-oracle peak only'}
        pending=self.path/'summary.pending'
        pending.write_text(json.dumps(summary,indent=2),encoding='utf-8');pending.replace(self.path/'summary.json')

    def finish(self):
        rows=[]
        for i in self.evaluated_indices:
            values=[r[f'P{i}_accuracy'] for r in self.rows]
            if values:
                rows.append({'profile':f'P{i}','rounds_completed':len(values),'final_accuracy':values[-1],
                             'final_ce':self.rows[-1][f'P{i}_ce'],'last20_count':len(values[-20:]),
                             'last20_mean':statistics.mean(values[-20:]),'last20_std_population':statistics.pstdev(values[-20:])})
        if rows:
            with (self.path/'accuracy_summary.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        completed={'rounds_completed':len(self.rows),'requested_epochs':self.args.epochs,
                   'success':len(self.rows)==self.args.epochs,
                   'formal_300_success':len(self.rows)==self.args.epochs==300,
                   'elapsed_seconds':self.elapsed_before+time.perf_counter()-self.started,
                   'checkpoint':'latest.pt' if self.save_pt else None,
                   'checkpoint_policy':'saved' if self.save_pt else 'disabled_by_user',
                   'selection':'latest/final round, no test selection'}
        (self.path/'completed.json').write_text(json.dumps(completed,indent=2),encoding='utf-8')
        self.write_summary()
        self.accuracy.close();self.diagnostics.close()
