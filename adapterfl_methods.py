"""Opt-in hooks for the ORIGINAL seven-model AdapterFL loop.

No model, sampler, optimizer, test function or aggregation mapping is replaced.
The two named regions below are COVERAGE regions, not invented ResNet stages.
"""
import copy
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
           'partition_alpha_shared','partition_alpha_remainder','shared_prox_lambda',
           'partition_norm_multiplier','partition_norm_min_clients','history_beta')


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
        if any(not math.isfinite(float(v)) for v in c.values()):
            raise ValueError('Method options must be finite')
        if (c['coverage_gamma'] < 0 or not 0 < c['coverage_eta_min'] <= 1 <= c['coverage_eta_max']
                or min(c['conflict_lambda'],c['shared_prox_lambda'],c['partition_alpha_shared'],c['partition_alpha_remainder']) < 0
                or c['partition_norm_multiplier'] <= 0 or c['partition_norm_min_clients'] < 2
                or not 0 <= c['history_beta'] < 1):
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
        self.round_diagnostics = {}

    def state_dict(self):
        return cpu_copy({'method':self.method,'config':self.config,'shared_shapes':self.shared_shapes,
                         'velocity':self.velocity,'previous_active':self.previous_active,
                         'previous_norms':self.previous_norms})

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

    def region(self, key, tensor, name):
        mask = torch.zeros_like(tensor,dtype=torch.bool)
        sl = tuple(slice(0,min(a,b)) for a,b in zip(tensor.shape,self.shared_shapes[key]))
        mask[sl] = True
        return mask if name=='shared' else ~mask

    def configure_local(self, local):
        if self.method=='m4' and self.config['shared_prox_lambda'] > 0:
            local.adapterfl_prox = (self.shared_shapes,self.config['shared_prox_lambda'])

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
                    'covered_fraction':float(covered.float().mean()),
                    'slot_coverage':{'min':float(counts[k][covered].min()),'mean':float(reference),
                                     'max':float(counts[k].max())},
                    'coverage_histogram':coverage_histogram,
                    'eta':{'min':float(eta.min()),'mean':float(eta.mean()),'max':float(eta.max()),'histogram':eta_histogram}}
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
        result, dot, old_square, current_square, applied_square = {},0.,0.,0.,0.
        reactivated, covered_count = 0,0
        beta = self.config['history_beta']
        for k,old in previous.items():
            active = counts[k]>0
            prior = self.previous_active.get(k,torch.zeros_like(active))
            both = active & prior
            delta = base[k]-old
            velocity = torch.zeros_like(old)
            velocity[active] = delta[active]  # First observation or after a gap.
            if k in self.velocity:
                prev = self.velocity[k]
                dot += float((delta[both].double()*prev[both].double()).sum())
                old_square += float(prev[both].double().square().sum())
                current_square += float(delta[both].double().square().sum())
                if beta:
                    velocity[both] = beta*prev[both]+(1-beta)*delta[both]
            result[k] = base[k].clone()
            if beta and torch.any(both):
                result[k][both] = old[both]+velocity[both]
            result[k][~active] = old[~active]  # No historical drift without coverage.
            self.velocity[k],self.previous_active[k] = velocity,active.clone()
            covered_count += int(active.sum()); reactivated += int((active & ~prior).sum())
            applied_square += float(velocity.double().square().sum())
        raw = math.sqrt(sum(float((base[k]-v).double().square().sum()) for k,v in previous.items()))
        self.round_diagnostics = {'raw_update_l2':raw,'smoothed_update_l2':math.sqrt(applied_square),
            'delta_vs_previous_velocity_cosine':dot/math.sqrt(old_square*current_square) if min(old_square,current_square)>EPS**2 else None,
            'fresh_or_reactivated_coordinates':reactivated,'covered_coordinates':covered_count,'beta':beta}
        return result

    def conflict(self, uploads, lens, previous, dispatched):
        strength = self.config['conflict_lambda']
        if len(uploads)<2 or strength==0:
            self.round_diagnostics = {'fallback':'single_slot_or_zero_strength'}
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
        self.round_diagnostics = {'cosine':similarities,'conflict_weights':weights,
            'unnormalized_log_weights':logits,
            'negative_fraction':sum(s is not None and s<0 for s in similarities)/len(similarities),
            'common_coordinates':sum(d.numel() for d in deltas[0].values()),
            'fallback_slots':sum(s is None for s in similarities)}
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
                for i,factor in zip(group,factors):
                    if factor==1:
                        continue
                    if changed[i] is uploads[i]:
                        changed[i]={k:v.clone() for k,v in uploads[i].items()}
                    for k,v in uploads[i].items():
                        mask=self.region(k,v,region)
                        sent=dispatched[i][k]
                        changed[i][k][mask]=sent[mask]+factor*(v[mask]-sent[mask])
                diagnostics.append({'partition':region,'slots':group,'norms_before':norms,
                    'threshold':threshold,'factors':factors,'clipped_fraction':sum(f<1 for f in factors)/len(factors),
                    'norms_after':[n*f for n,f in zip(norms,factors)],'fallback':None if enough else 'insufficient_equal_support_slots'})
        self.round_diagnostics={'groups':diagnostics}
        return Aggregation_AdaptiveFL(changed,lens,previous)


def prepare_prox(net, setup):
    """Detached actual-dispatch snapshot of shared PARAMETER prefixes only."""
    shapes,strength=setup
    targets={}
    for k,p in net.named_parameters():
        sl=tuple(slice(0,d) for d in shapes[k])
        targets[k]=(sl,p[sl].detach().clone())
    return targets,sum(v.numel() for _,v in targets.values()),strength


def prox_loss(net, targets, count):
    result=next(net.parameters()).new_zeros(())
    for k,p in net.named_parameters():
        sl,initial=targets[k]
        result=result+(p[sl]-initial).square().sum()
    return .5*result/count


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


class RunLog:
    """Journal/checkpoint hook called by the native loop, not a training engine."""
    def __init__(self,args,models):
        self.args=args
        self.path=prepare_output(args)
        self.hooks=MethodHooks(args,models)
        self.rows=[]; self.next_round=0; self.total_time=0.
        self.started=time.perf_counter()
        self.elapsed_before=0.
        if args.client_chosen_mode!='available':
            raise ValueError('Reviewed experiments use original available scheduling only')
        if args.model!='resnet' or args.width_ration!=[.4,.66,1.] or args.depth_saved!=[2,3,4]:
            raise ValueError('Keep original seven ResNet profiles')
        self.identity={k:getattr(args,k) for k in ('seed','epochs','num_users','frac','local_ep','local_bs','bs','lr',
            'lr_decay','momentum','weight_decay','optimizer','dataset','model','iid','data_beta','client_hetero_ration','client_chosen_mode')}
        self.identity['method']=args.method; self.identity['options']=self.hooks.config
        self.identity['partition_sha256']=getattr(args,'_method_partition_sha256',None)
        files=('main_fed_ori.py','Algorithm/Training_AdaptiveFL.py','models/Update.py','models/Fed.py','models/resnet.py','models/test.py','utils/options.py','adapterfl_methods.py','utils/get_datatset_new.py','utils/set_seed.py')
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
            restore_rng(state['rng'])
        else:
            manifest={'identity':self.identity,'command':sys.argv,'torch':torch.__version__,
                      'python':sys.version,'device':str(args.device),'profiles':[(w,d) for w in args.width_ration for d in args.depth_saved if w!=1 or d==args.depth_saved[0]],
                      'partition_definition':'intersection of all seven actual parameter prefixes vs complement; no original fixed two-part topology',
                      'checkpoint_selection':'latest round only; never selected on official test accuracy'}
            (self.path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        self.accuracy=(self.path/'accuracy_each_round.csv').open('a' if args.method_resume else 'x',newline='',encoding='utf-8')
        self.diagnostics=(self.path/'method_diagnostics.csv').open('a' if args.method_resume else 'x',newline='',encoding='utf-8')
        self.accuracy_writer=None
        self.diagnostic_writer=csv.DictWriter(self.diagnostics,fieldnames=['round','method','metrics_json'])
        if not args.method_resume:
            self.diagnostic_writer.writeheader()

    def record(self,round_index,models,evaluations,max_time,total_time,wall,server_seconds,eval_seconds,slots,local_logs):
        row={'round':round_index+1,'full_accuracy':evaluations[-1][0],'full_ce':evaluations[-1][1],
             'original_max_client_seconds':max_time,'original_cumulative_max_client_seconds':total_time,
             'round_wall_seconds':wall,'server_seconds':server_seconds,'evaluation_seconds':eval_seconds}
        for i,(acc,ce) in enumerate(evaluations):
            row[f'P{i}_accuracy']=acc; row[f'P{i}_ce']=ce
        if self.accuracy_writer is None:
            self.accuracy_writer=csv.DictWriter(self.accuracy,fieldnames=list(row))
            if not self.args.method_resume:
                self.accuracy_writer.writeheader()
        self.accuracy_writer.writerow(row);self.accuracy.flush()
        metrics={'server':self.hooks.round_diagnostics,'clients':local_logs,'slots':slots}
        self.diagnostic_writer.writerow({'round':round_index+1,'method':self.args.method,'metrics_json':json.dumps(metrics,allow_nan=False)})
        self.diagnostics.flush()
        self.rows.append(row);self.total_time=total_time;self.next_round=round_index+1
        checkpoint={'identity':self.identity,'next_round':self.next_round,'total_time':total_time,
                    'elapsed_seconds':self.elapsed_before+time.perf_counter()-self.started,'rows':self.rows,
                    'global':cpu_copy(models[-1].state_dict()),'method_state':self.hooks.state_dict(),'rng':rng_state()}
        temp=self.path/'latest.pending'
        torch.save(checkpoint,temp); temp.replace(self.path/'latest.pt')

    def finish(self):
        rows=[]
        for i in range(7):
            values=[r[f'P{i}_accuracy'] for r in self.rows]
            if values:
                rows.append({'profile':f'P{i}','rounds_completed':len(values),'final_accuracy':values[-1],
                             'final_ce':self.rows[-1][f'P{i}_ce'],'last20_count':len(values[-20:]),
                             'last20_mean':statistics.mean(values[-20:]),'last20_std_population':statistics.pstdev(values[-20:])})
        if rows:
            with (self.path/'accuracy_summary.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        completed={'rounds_completed':len(self.rows),'requested_epochs':self.args.epochs,
                   'formal_300_success':len(self.rows)==self.args.epochs==300,
                   'elapsed_seconds':self.elapsed_before+time.perf_counter()-self.started,
                   'checkpoint':'latest.pt','selection':'latest/final round, no test selection'}
        (self.path/'completed.json').write_text(json.dumps(completed,indent=2),encoding='utf-8')
        self.accuracy.close();self.diagnostics.close()
