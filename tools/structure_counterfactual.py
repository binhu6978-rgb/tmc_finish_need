"""Fixed offline A0-A4 comparison on saved mature uploads. No client SGD."""
import argparse
from collections import defaultdict, Counter
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import statistics
import sys
import time
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapterfl_methods import MethodHooks, isolated_evaluation_rng
from models.Fed import Aggregation_AdaptiveFL, get_model_list
from semcons_scope import ScopeHooks, slices
from tools.receiver_ce_probe import ce_vector


def input_token(states):
    return [tuple((k,id(v),v._version) for k,v in state.items()) for state in states]


def evidence_pool(slots, records):
    """Same region evidence as A4; physical-client pooling follows S05's pattern,
    using direction only. Zero risks contribute to the denominator too.
    """
    values=defaultdict(list)
    for record in records:
        grouped=defaultdict(list)
        for i,risk in record.get('direction_risks',{}).items():
            grouped[slots[int(i)][0]].append(risk)
        for cid,risks in grouped.items():
            values[cid].append((record['numel'],sum(risks)/len(risks)))
    return {cid:sum(n*r for n,r in pairs)/sum(n for n,_ in pairs) for cid,pairs in values.items()}


@torch.no_grad()
def whole_client_control(hook, uploads, lens, dispatched, base, records, strength=4.):
    risks=evidence_pool(hook.slots,records)
    output={k:v.clone() for k,v in base.items()}; details=[]
    for region,record in zip(hook.regions,records):
        members=record['slots']; gates={i:math.exp(-strength*risks.get(hook.slots[i][0],0.)) for i in members}
        row={'region_id':region['id'],'slots':members,'global_risks':{str(i):risks.get(hook.slots[i][0],0.) for i in members},
             'gates':[gates[i] for i in members],'physical_clients':record['physical_clients'],
             'extrapolated_slots':[i for i in members if gates[i]<1-1e-12 and record.get('direction_risks',{}).get(str(i),0.)==0.]}
        if not members:
            row.update(returned_native_exactly=True,changed_coordinates=0,common_gate_cancellation=False)
            details.append(row); continue
        triggered=any(v<1-1e-12 for v in gates.values())
        common=max(gates.values())-min(gates.values())<=1e-14
        row['common_gate_cancellation']=triggered and common
        if not triggered or common:
            row.update(returned_native_exactly=True,changed_coordinates=0)
            details.append(row); continue
        deltas=hook.region_deltas(region,uploads,dispatched,members)
        native=sum(deltas[i]*lens[i] for i in members)/sum(lens[i] for i in members)
        applied=sum(deltas[i]*(lens[i]*gates[i]) for i in members)/sum(lens[i]*gates[i] for i in members)
        offset=0; changed=0
        for cell in region['cells']:
            key=cell['key']; sl=slices(cell); n=cell['numel']
            output[key][sl]=base[key][sl]+(applied-native)[offset:offset+n].reshape(base[key][sl].shape).to(base[key].dtype)
            changed+=int((output[key][sl]!=base[key][sl]).sum()); offset+=n
        row.update(changed_coordinates=changed,returned_native_exactly=changed==0)
        details.append(row)
    return output,{'physical_client_risks':risks,'regions':details}


@torch.no_grad()
def aggregates(models, args, anchor, snap):
    u=snap['uploads']; n=snap['lens']; sent=snap['dispatched']
    before=input_token([anchor,*u,*sent])
    base=Aggregation_AdaptiveFL(u,n,anchor); states={'A0':base}; diagnostics={}
    for name,method in [('A1','m2'),('A2','m3')]:
        opt=copy.copy(args); opt.method=method; opt.conflict_lambda=4.
        opt.partition_alpha_shared=1.; opt.partition_alpha_remainder=.98
        h=MethodHooks(opt,models); states[name]=h.aggregate(u,n,anchor,sent)
        diagnostics[name]=h.round_diagnostics
    opt=copy.copy(args); opt.method='s03'; opt.conflict_lambda=4.
    h=ScopeHooks(opt,models); h.start_round(0); h.slots=snap['slots']
    # S03 uses only direction; no Q measurement or Q decision is needed.
    h.observations=[None]*len(u)
    if h.regions!=snap['regions']: raise ValueError('Saved support regions differ')
    states['A4']=h.aggregate(u,n,anchor,sent); diagnostics['A4']=h.round_diagnostics
    records=diagnostics['A4']['regions']
    states['A3'],diagnostics['A3']=whole_client_control(h,u,n,sent,base,records)
    if input_token([anchor,*u,*sent])!=before: raise AssertionError('Counterfactual mutated input')
    return states,diagnostics,h


def geometry(state, anchor, base):
    square=0.; diff=0.; dot=0.; reference=0.; changed=0; total=0
    for k,v in state.items():
        d=(v-anchor[k]).double(); b=(base[k]-anchor[k]).double(); correction=(v-base[k]).double()
        square+=float(d.square().sum()); reference+=float(b.square().sum()); dot+=float((d*b).sum())
        diff+=float(correction.square().sum()); changed+=int(v.ne(base[k]).sum()); total+=v.numel()
    cosine=dot/math.sqrt(square*reference) if min(square,reference)>0 else None
    return {'update_l2':math.sqrt(square),'difference_from_A0_l2':math.sqrt(diff),'changed_from_A0_fraction':changed/total,
            'update_norm_ratio_A0':math.sqrt(square/reference) if reference else None,'update_cosine_A0':cosine,
            'orthogonal_update_fraction':math.sqrt(max(0.,1-cosine*cosine)) if cosine is not None else None}


@torch.no_grad()
def rescale(state, anchor, ratio):
    return {k:anchor[k]+(v-anchor[k])*ratio for k,v in state.items()}


@torch.no_grad()
def evaluate(net,state,probes):
    net.load_state_dict(state,strict=True); net.eval()
    with isolated_evaluation_rng():
        return {int(cid):float(ce_vector(net,probe).mean()) for cid,probe in sorted(probes.items())}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def summarize(output):
    """Offline CSV reduction only; never reevaluates models or selects settings."""
    def read(name):
        with (output/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    metrics=read('cohort_metrics.csv'); clients=read('client_ce.csv'); regions=read('regions.csv')
    methods=sorted({r['method'] for r in metrics})
    fields=[k for k in metrics[0] if k not in ('cohort','method')]
    mean=lambda rows,key:statistics.mean(float(r[key]) for r in rows)
    counts=lambda values:dict(Counter('improved' if x>1e-6 else 'worse' if x< -1e-6 else 'neutral' for x in values))
    result={'method_mean':{m:{k:mean([r for r in metrics if r['method']==m],k) for k in fields} for m in methods},
            'client_groups':{},'A4_comparisons':{},'region_summary':{}}
    groups={'all':lambda r:True,'low':lambda r:r['capacity']=='low','medium':lambda r:r['capacity']=='medium',
            'high':lambda r:r['capacity']=='high','participating':lambda r:r['participating']=='True',
            'nonparticipating':lambda r:r['participating']=='False'}
    for group,predicate in groups.items():
        result['client_groups'][group]={}
        for m in methods:
            rows=[r for r in clients if r['method']==m and predicate(r)]
            result['client_groups'][group][m]={'client_cohort_count':len(rows),'full_ce':mean(rows,'full_ce'),
                'improvement_vs_A0':mean(rows,'improvement_vs_A0'),
                'vs_A0_counts':counts([float(r['improvement_vs_A0']) for r in rows])}
    lookup={(int(r['cohort']),r['method']):float(r['full_probe_ce']) for r in metrics}
    for target in ('A0','A1','A2','A3'):
        differences=[lookup[(c,target)]-lookup[(c,'A4')] for c in range(1,6)]
        result['A4_comparisons'][target]={'mean_ce_improvement':statistics.mean(differences),
            'by_cohort_improvement':differences,'cohort_counts':counts(differences)}
    result['A4_comparisons']['A3_equal_A0_norm']={'mean_ce_improvement':statistics.mean(
        lookup[(c,'A3_at_A0_norm')]-lookup[(c,'A4_at_A0_norm')] for c in range(1,6))}
    for m in ('A3','A4'):
        result['region_summary'][m]={'gated_region_slots':sum(int(r[m+'_gated_slots']) for r in regions),
            'gated_region_cohorts':sum(int(r[m+'_gated_slots'])>0 for r in regions),
            'common_gate_cancel_regions':sum(r[m+'_common_gate_cancel']=='True' for r in regions),
            'actually_changed_regions':sum(int(r[m+'_changed_coordinates'])>0 for r in regions)}
    result['region_summary']['direction_valid_comparisons']=sum(int(r['direction_comparisons']) for r in regions)
    result['region_summary']['direction_conflicts']=sum(int(r['direction_conflicts']) for r in regions)
    result['region_summary']['A3_extrapolated_region_slots']=sum(int(r['A3_extrapolated_slots']) for r in regions)
    result['anchor_full_ce']=mean([r for r in clients if r['method']=='A0'],'anchor_ce')
    result['completed']=json.loads((output/'completed.json').read_text())
    with (output/'summary.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2)
    table=[dict(method=m,**result['method_mean'][m]) for m in ('A0','A1','A2','A3','A4')]
    write_csv(output/'comparison.csv',table)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--inputs',type=Path,default=ROOT/'runs/mature_receiver_q_final5_seed1_20261019')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/mature_q_inputs/E03_latest400.pt')
    p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args(); protocol=json.loads((ROOT/'configs/structure_counterfactual5.json').read_text())
    complete=json.loads((opt.inputs/'diagnostic_completed.json').read_text())
    checkpoint=torch.load(opt.checkpoint,map_location='cpu',weights_only=False)
    if not complete['success'] or checkpoint['next_round']!=400: raise ValueError('Final mature inputs required')
    opt.output.mkdir(parents=True,exist_ok=False)
    (opt.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    args=SimpleNamespace(**json.loads((opt.inputs/'manifest.json').read_text())['training_configuration'])
    args.device=torch.device('cpu'); torch.set_num_threads(2)
    models,_=get_model_list(args)
    anchor=checkpoint['global']; original=copy.deepcopy(anchor)
    net=copy.deepcopy(models[-1]).cuda().eval()
    probes=torch.load(opt.inputs/'fixed_probes.pt',map_location='cpu',weights_only=False)
    if len(probes)!=100 or any(p is None or len(p['y'])!=32 for p in probes.values()):
        raise ValueError('Expected original complete 100-client probe panel')
    (opt.output/'models').mkdir()
    client_rows=[]; summary=[]; region_rows=[]; diagnostic_rows=[]
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    anchor_ce=evaluate(net,anchor,probes)
    zero_a=ce_vector(net,probes[0]); zero_b=ce_vector(net,probes[0])
    zero_control=float((zero_a-zero_b).abs().max())
    if zero_control>protocol['ce_tolerance']: raise ValueError('Fixed Full probe zero control failed')
    for cohort in protocol['cohorts']:
        snap=torch.load(opt.inputs/f'raw_round_{cohort:02d}.pt',map_location='cpu',weights_only=False)
        # Saved dispatches must be mature prefixes of this SAME Full anchor.
        for sent in snap['dispatched']:
            for k,v in sent.items():
                if not torch.equal(v,anchor[k][tuple(slice(0,n) for n in v.shape[:2])]):
                    raise ValueError('Upload cohort has different mature anchor')
        states,diagnostics,hook=aggregates(models,args,anchor,snap)
        metrics={name:geometry(state,anchor,states['A0']) for name,state in states.items()}
        variants=dict(states)
        for name in ('A1','A2','A3','A4'):
            if metrics[name]['update_l2']==0: raise ValueError('Cannot norm-match zero update')
            variants[name+'_at_A0_norm']=rescale(states[name],anchor,metrics['A0']['update_l2']/metrics[name]['update_l2'])
        variants['A0_at_A4_norm']=rescale(states['A0'],anchor,metrics['A4']['update_l2']/metrics['A0']['update_l2'])
        evaluations={name:evaluate(net,state,probes) for name,state in variants.items()}
        participating={cid for cid,_ in snap['slots']}
        for name,state in variants.items():
            actual=geometry(state,anchor,states['A0'])
            ce=evaluations[name]; mean=sum(ce.values())/100
            changes=[evaluations['A0'][cid]-ce[cid] for cid in ce]
            summary.append(dict(cohort=cohort,method=name,full_probe_ce=mean,improvement_vs_A0=sum(changes)/100,
                                **actual))
            for cid,value in ce.items():
                client_rows.append(dict(cohort=cohort,method=name,client=cid,capacity='low' if cid<40 else 'medium' if cid<70 else 'high',
                    participating=cid in participating,full_ce=value,anchor_ce=anchor_ce[cid],
                    improvement_vs_A0=evaluations['A0'][cid]-value,improvement_vs_A3=evaluations['A3'][cid]-value))
            if name in states: torch.save({'cohort':cohort,'method':name,'global':state},opt.output/'models'/f'{cohort:02d}_{name}.pt')
        for region,record,global_record in zip(hook.regions,diagnostics['A4']['regions'],diagnostics['A3']['regions']):
            row=dict(cohort=cohort,region=region['id'],numel=region['numel'],covering_slots=len(record['slots']),
                physical_clients=record['client_count'],direction_comparisons=len(record.get('direction_risks',{})),
                direction_conflicts=sum(x>0 for x in record.get('direction_risks',{}).values()))
            for name,gates in [('A3',global_record['gates']),('A4',record['gates'])]:
                row[name+'_gated_slots']=sum(g<1-1e-12 for g in gates)
                row[name+'_common_gate_cancel']=bool(gates and min(gates)<1-1e-12 and max(gates)-min(gates)<=1e-14)
            row['A3_extrapolated_slots']=len(global_record['extrapolated_slots'])
            for name,state in states.items():
                differences=[(state[c['key']][slices(c)]-states['A0'][c['key']][slices(c)]).double() for c in region['cells']]
                row[name+'_changed_coordinates']=sum(int(x.ne(0).sum()) for x in differences)
                row[name+'_difference_l2']=math.sqrt(sum(float(x.square().sum()) for x in differences))
            region_rows.append(row)
        diagnostic_rows.append({'cohort':cohort,'slots':snap['slots'],'lens':snap['lens'],'methods':diagnostics})
        if not all(torch.equal(anchor[k],original[k]) for k in anchor): raise AssertionError('Full anchor changed')
    write_csv(opt.output/'cohort_metrics.csv',summary); write_csv(opt.output/'client_ce.csv',client_rows)
    write_csv(opt.output/'regions.csv',region_rows)
    (opt.output/'diagnostics.json').write_text(json.dumps(diagnostic_rows,indent=2))
    (opt.output/'completed.json').write_text(json.dumps({'success':True,'cohorts':5,'counterfactual_models':25,
        'probe_clients':100,'probe_examples_per_client':32,'client_SGD_calls':0,'official_test_access':False,
        'input_tensors_unmodified':True,'anchor_unmodified':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'fixed_batch_zero_control_max_abs_ce':zero_control,
        'wall_seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2))
    print(json.dumps({'completed':True,'output':str(opt.output),'cohort_method_rows':len(summary)},indent=2))
    summarize(opt.output)


if __name__=='__main__': main()
