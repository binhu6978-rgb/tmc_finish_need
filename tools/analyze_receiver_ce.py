"""Describe fixed recipient CE effects. Never fits or updates any Gate."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch


def stats(values):
    x=np.array(values,dtype=float)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,'median':float(np.median(x)) if len(x) else None,
            'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None}


def auc(scores,labels):
    positive=[s for s,y in zip(scores,labels) if y];negative=[s for s,y in zip(scores,labels) if not y]
    if not positive or not negative:return None
    return sum((a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative))


def ce_sign(value,tolerance):
    return 'worse' if value>tolerance else 'better' if value< -tolerance else 'neutral'


def incremental(rows,controlled=False):
    # Predetermined OLS; scaling uses training folds only. No tuning, no Gate change.
    controls=['alpha','scaled_delta_l2','ce_before','sender_own_gain'] if controlled else []
    def features(row,add_q):return [-row['direction_cosine']]+[row[k] for k in controls]+([row['q']] if add_q else [])
    predictions={False:[],True:[]};truth=[];folds=[]
    for round_index in sorted({r['round'] for r in rows}):
        train=[r for r in rows if r['round']!=round_index];test=[r for r in rows if r['round']==round_index]
        if len(train)<len(controls)+5:return {'status':'无法判断','reason':'too_few_training_pairs_for_fixed_OLS'}
        truth.extend(r['delta_ce'] for r in test);folds.append({'round':round_index,'train':len(train),'test':len(test)})
        for add_q in (False,True):
            x=np.array([features(r,add_q) for r in train]);z=np.array([features(r,add_q) for r in test])
            mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
            x=np.column_stack([np.ones(len(x)),(x-mean)/scale]);z=np.column_stack([np.ones(len(z)),(z-mean)/scale])
            # Same float64 OLS objective. Torch's CPU LAPACK also works on the
            # local Windows environment whose NumPy lstsq DLL fails to load.
            coefficient=torch.linalg.lstsq(torch.from_numpy(x),
                torch.tensor([r['delta_ce'] for r in train],dtype=torch.float64),driver='gelsd').solution.numpy()
            predictions[add_q].extend((z@coefficient).tolist())
    y=np.array(truth);a=float(np.mean((y-predictions[False])**2));b=float(np.mean((y-predictions[True])**2))
    return {'status':'descriptive_leave_round_out','folds':folds,'controls':controls,'direction_mse':a,'direction_plus_q_mse':b,
            'relative_mse_reduction':(a-b)/a if a>0 else None,'warning':'single-seed dependent pairs; exploratory association, not Gate effectiveness'}


def summarize(rows,protocol):
    tolerance=protocol['ce_tolerance'];valid=[r for r in rows if r.get('valid')]
    signed=[r for r in valid if ce_sign(r['delta_ce'],tolerance)!='neutral' and r['q'] is not None]
    counts=Counter(ce_sign(r['delta_ce'],tolerance) for r in valid)
    floors=protocol['minimum_incremental_sample'];senders={r['sender'] for r in signed};rounds={r['round'] for r in signed}
    signed_counts=Counter(ce_sign(r['delta_ce'],tolerance) for r in signed)
    eligible=(signed_counts['better']>=floors['each_ce_sign_pairs'] and signed_counts['worse']>=floors['each_ce_sign_pairs']
              and len(senders)>=floors['distinct_senders'] and len(rounds)>=floors['rounds'])
    contexts=defaultdict(list)
    for r in valid:contexts[(r['round'],r['region'],r['sender_slot'])].append(r)
    mixed=sum({'better','worse'}<=set(ce_sign(r['delta_ce'],tolerance) for r in values) for values in contexts.values())
    scores={'q':lambda r:r['q'],'negative_signed_LOO_cosine':lambda r:-r['direction_cosine'],
            'clipped_direction_risk':lambda r:r['direction_risk'],
            'negative_pair_cosine':lambda r:-r['pair_direction_cosine'] if r['pair_direction_cosine'] is not None else None}
    result={'pairs':len(rows),'valid_pairs':len(valid),'invalid_reasons':dict(Counter(r.get('reason') for r in rows if not r.get('valid'))),
        'outcomes':dict(counts),'delta_ce':stats([r['delta_ce'] for r in valid]),'physical_senders':len({r['sender'] for r in valid}),
        'physical_receivers':len({r['receiver'] for r in valid}),'sign_analysis_distinct_senders':len(senders),
        'regions':sorted({r['region'] for r in valid}),'profile_pairs':dict(Counter(f"P{r['sender_profile']}->P{r['receiver_profile']}" for r in valid)),
        'q_by_outcome':{name:stats([r['q'] for r in valid if r['q'] is not None and ce_sign(r['delta_ce'],tolerance)==name]) for name in ('better','worse','neutral')},
        'sender_own_gain':stats([r['sender_own_gain'] for r in valid]),
        'sender_own_gain_positive_receiver_worse':sum(r['sender_own_gain']>tolerance and ce_sign(r['delta_ce'],tolerance)=='worse' for r in valid),
        'mixed_receiver_sign_contexts':int(mixed),'sender_region_contexts':len(contexts),
        'multiple_receiver_contexts':sum(len({r['receiver'] for r in values})>=2 for values in contexts.values()),'classes':{}}
    for scope in ('all','sender_training_classes','q_compared_classes'):
        cells=[c for r in valid for c in r['classes'] if scope=='all'
               or (scope=='sender_training_classes' and c['sender_has_training_class'])
               or (scope=='q_compared_classes' and c['in_sender_q_classes'])]
        result['classes'][scope]={'count':len(cells),'outcomes':dict(Counter(ce_sign(c['delta_ce'],tolerance) for c in cells)),'delta_ce':stats([c['delta_ce'] for c in cells])}
    result['auc']={}
    for name,score in scores.items():
        selected=[r for r in signed if score(r) is not None]
        result['auc'][name]={'value':auc([score(r) for r in selected],[r['delta_ce']>tolerance for r in selected]),'pairs':len(selected)}
    result['incremental_status']='descriptive_only' if eligible else '无法判断'
    result['minimum_sample_requirements']=floors
    result['OLS']=incremental(signed) if eligible else None
    result['OLS_same_magnitude_own_gain_controls']=incremental(signed,True) if eligible else None
    return result


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    rounds=[json.loads(line) for line in (root/'receiver_ce.jsonl').read_text().splitlines()]
    if len(rounds)!=5 or [r['round'] for r in rounds]!=list(range(1,6)) or not (root/'diagnostic_completed.json').is_file():
        raise ValueError('Exactly five completed diagnostic rounds required')
    records=[r for row in rounds for r in row['records']]
    primary=[r for r in records if r['round'] in protocol['primary_rounds']]
    result={'protocol':protocol,'all5':summarize(records,protocol),'primary2_5':summarize(primary,protocol),
        'round1':summarize([r for r in records if r['round']==1],protocol),
        'by_stratum':{s:summarize([r for r in primary if r['stratum']==s],protocol) for s in ('conflict','nonconflict')},
        'all5_by_stratum':{s:summarize([r for r in records if r['stratum']==s],protocol) for s in ('conflict','nonconflict')},
        'inventory':[{'round':r['round'],'regions':r['inventory']} for r in rounds]}
    matched=[]
    for r in primary:
        if not r.get('valid'):continue
        cells=[c for c in r['classes'] if c['in_sender_q_classes']]
        if not cells:continue
        row=dict(r,classes=cells,delta_ce=sum(c['delta_ce']*c['n'] for c in cells)/sum(c['n'] for c in cells),
                 ce_before=sum(c['ce_before']*c['n'] for c in cells)/sum(c['n'] for c in cells));matched.append(row)
    result['primary_q_compared_classes_only']=summarize(matched,protocol)
    total_coordinates=11424356
    coverage=[]
    for row in rounds:
        sampled={r['region']:r['region_numel'] for r in row['records']}
        coverage.append({'round':row['round'],'regions':sorted(sampled),'union_coordinate_fraction_full':sum(sampled.values())/total_coordinates,
                         'pairs_by_stratum':dict(Counter(r['stratum'] for r in row['records']))})
    result['coverage']=coverage
    result['cost']={'pairs':len(records),'injection_forwards':sum(r.get('valid') is not None and 'delta_ce' in r for r in records),
        'baseline_zero_control_forwards':2*sum(row['baseline_slots'] for row in rounds),
        'sender_post_forwards':sum(row['sender_full_post_forwards'] for row in rounds),
        'diagnostic_seconds_including_snapshot_io':sum(row['diagnostic_seconds'] for row in rounds),
        'raw_tensor_bytes':sum(row['raw_tensor_bytes'] for row in rounds),
        'online_extra_delta_downlink_float32_proxy':sum(r['region_numel']*4 for r in records),
        'online_feedback_class_delta_count_uint32_float32_proxy':sum(len(r.get('classes',[]))*12 for r in records)}
    receiver_load=Counter((r['round'],r['receiver']) for r in records)
    result['cost']['max_injection_queries_per_round_physical_receiver']=max(receiver_load.values(),default=0)
    result['cost']['unique_receiver_round_slot_baseline_forwards']=len({(r['round'],r['receiver_slot']) for r in records})
    result['cost']['native_one_direction_slot_parameter_bytes']=result['cost']['raw_tensor_bytes']//2
    return result,records


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);opt=p.parse_args()
    result,records=analyze(opt.root);opt.output.mkdir(parents=True,exist_ok=False)
    (opt.output/'summary.json').write_text(json.dumps(result,indent=2))
    fields=['round','region','stratum','sender','receiver','sender_slot','receiver_slot','sender_profile','receiver_profile',
            'region_numel','covering_slot_count','covering_physical_clients','native_mass','sender_weight','alpha',
            'direction_cosine','direction_risk','pair_direction_cosine','q','sender_own_gain','ce_before','ce_after','delta_ce',
            'outcome','valid','reason','scaled_delta_l2','receiver_probe_examples','receiver_probe_sha256']
    with (opt.output/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(records)
    class_fields=['round','region','stratum','sender','receiver','sender_profile','receiver_profile','class','n',
                  'ce_before','ce_after','delta_ce','sender_has_training_class','in_sender_q_classes']
    with (opt.output/'classes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=class_fields,extrasaction='ignore',lineterminator='\n');writer.writeheader()
        for row in records:
            if not row.get('valid'):continue
            for cell in row['classes']:
                writer.writerow({**{k:row[k] for k in class_fields[:7]},**cell})
    print(json.dumps({'all5':result['all5'],'primary':result['primary2_5'],'strata':result['by_stratum'],'cost':result['cost']},indent=2))


if __name__=='__main__':main()
