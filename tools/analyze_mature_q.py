"""Fixed final Q analysis. All effects are training-probe associations, not Gates."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_receiver_ce import summarize, ce_sign, auc


def blocked_ols(rows, key, controlled=False):
    """Leave one complete physical ID/region/cohort out, no row-wise random split."""
    controls = ['alpha','scaled_delta_l2','ce_before','sender_own_gain'] if controlled else []
    valid = [r for r in rows if r.get('valid') and r.get('q') is not None and abs(r['delta_ce'])>1e-6]
    keys = sorted({r[key] for r in valid})
    if len(keys)<3:
        return {'status':'无法判断','reason':'fewer_than_three_blocks','block':key,'groups':len(keys)}
    total = [0., 0.]; folds = []; truth = 0
    for group in keys:
        train = [r for r in valid if r[key]!=group]; test = [r for r in valid if r[key]==group]
        if len(train)<len(controls)+5:
            return {'status':'无法判断','reason':'insufficient_training_blocks','block':key}
        errors = []
        for add_q in (False, True):
            names = ['direction_cosine'] + controls + (['q'] if add_q else [])
            x = torch.tensor([[r[n] for n in names] for r in train],dtype=torch.float64)
            z = torch.tensor([[r[n] for n in names] for r in test],dtype=torch.float64)
            mean = x.mean(0); scale = x.std(0,correction=0); scale[scale<1e-12] = 1.
            x = torch.cat([torch.ones(len(x),1,dtype=torch.float64),(x-mean)/scale],1)
            z = torch.cat([torch.ones(len(z),1,dtype=torch.float64),(z-mean)/scale],1)
            y = torch.tensor([r['delta_ce'] for r in train],dtype=torch.float64)
            target = torch.tensor([r['delta_ce'] for r in test],dtype=torch.float64)
            coefficient = torch.linalg.lstsq(x,y,driver='gelsd').solution
            errors.append(float(((z@coefficient-target)**2).sum()))
        truth += len(test)
        total = [a+b for a,b in zip(total,errors)]
        folds.append({'held_out':group,'train_pairs':len(train),'test_pairs':len(test),
                      'direction_mse':errors[0]/len(test),'direction_plus_q_mse':errors[1]/len(test)})
    return {'status':'dependent_pair_sensitivity_only','block':key,'controls':controls,'groups':len(keys),
            'pairs':truth,'direction_mse':total[0]/truth,'direction_plus_q_mse':total[1]/truth,
            'relative_mse_reduction':(total[0]-total[1])/total[0] if total[0]>0 else None,
            'folds':folds,'warning':'Crossed sender/receiver/region dependencies remain; no iid-pair significance claim'}


def describe(rows, protocol):
    out = summarize(rows,protocol)
    out['by_cohort'] = {str(c): dict(Counter(ce_sign(r['delta_ce'],protocol['ce_tolerance']) if r.get('valid') else 'invalid'
                                          for r in rows if r['round']==c))
                        for c in protocol['primary_rounds']}
    out['by_region'] = {str(g): dict(Counter(ce_sign(r['delta_ce'],protocol['ce_tolerance']) if r.get('valid') else 'invalid'
                                          for r in rows if r['region']==g))
                        for g in sorted({r['region'] for r in rows})}
    out['blocked_OLS'] = {k:{'direction':blocked_ols(rows,k), 'magnitude_own_gain_controlled':blocked_ols(rows,k,True)}
                          for k in ('round','sender','receiver','region')}
    contexts = defaultdict(list)
    for r in rows:
        if r.get('valid') and r.get('q') is not None:
            contexts[(r['round'],r['region'],r['sender'])].append(r)
    # Equal context weighting is a descriptive sensitivity, not an increased N.
    context_rows = []
    for values in contexts.values():
        row = dict(values[0]); row['delta_ce'] = sum(v['delta_ce'] for v in values)/len(values)
        context_rows.append(row)
    out['sender_region_cohort_contexts'] = len(contexts)
    out['context_mean_ce_outcomes'] = dict(Counter(ce_sign(r['delta_ce'],1e-6) for r in context_rows))
    out['context_mean_q_auc'] = auc([r['q'] for r in context_rows if abs(r['delta_ce'])>1e-6],
                                   [r['delta_ce']>1e-6 for r in context_rows if abs(r['delta_ce'])>1e-6])
    out['incremental_interpretation'] = ('无法判断：未达到预先固定的独立发送者/CE符号/批次数量下限'
                                        if out['incremental_status']=='无法判断' else
                                        '仅探索关联；需同时查看方向控制及交叉依赖分块敏感性，不能当作独立统计验证')
    return out


def analyze(root):
    protocol = json.loads((root/'protocol.json').read_text())
    complete = json.loads((root/'diagnostic_completed.json').read_text())
    rounds = [json.loads(line) for line in (root/'receiver_ce.jsonl').read_text().splitlines()]
    if (not complete.get('success') or complete.get('global_updates')!=0 or len(rounds)!=5
        or [r['round'] for r in rounds]!=protocol['primary_rounds']
        or not all(r['anchor_rng_history_unchanged'] for r in rounds)):
        raise ValueError('Five completed read-only mature cohorts required')
    records = [r for row in rounds for r in row['records']]
    matched = []
    for r in records:
        if not r.get('valid'): continue
        cells = [c for c in r['classes'] if c['in_sender_q_classes']]
        if not cells: continue
        n = sum(c['n'] for c in cells)
        matched.append(dict(r,classes=cells,delta_ce=sum(c['delta_ce']*c['n'] for c in cells)/n,
                            ce_before=sum(c['ce_before']*c['n'] for c in cells)/n))
    result = {'protocol':protocol,'all':describe(records,protocol),
              'by_stratum':{s:describe([r for r in records if r['stratum']==s],protocol) for s in ('conflict','nonconflict')},
              'q_compared_class_sensitivity':{s:describe([r for r in matched if r['stratum']==s],protocol)
                                             for s in ('conflict','nonconflict')},
              'inventory':[{'cohort':r['round'],'regions':r['inventory']} for r in rounds],
              'resources':json.loads((root/'resources.json').read_text()),
              'verification':complete,
              'cost':{'native_sgd_seconds':sum(r['native_train_seconds'] for r in rounds),
                      'probe_seconds_including_snapshot_io':sum(r['diagnostic_seconds'] for r in rounds),
                      'raw_tensor_bytes':sum(r['raw_tensor_bytes'] for r in rounds),
                      'extra_downlink_float32_bytes_proxy':sum(r['region_numel']*4 for r in records),
                      'class_feedback_float32_delta_uint32_class_count_bytes_proxy':sum(len(r.get('classes',[]))*12 for r in records),
                      'injection_batches':len([r for r in records if 'delta_ce' in r]),
                      'baseline_zero_control_batches':2*sum(r['baseline_slots'] for r in rounds),
                      'sender_post_batches':sum(r['sender_full_post_forwards'] for r in rounds)}}
    result['offline_verification'] = json.loads((root/'offline_verification.json').read_text()) if (root/'offline_verification.json').exists() else None
    result['analysis_source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('tools/analyze_mature_q.py','tools/analyze_receiver_ce.py','tools/verify_mature_q.py')}
    files = ['protocol.json','manifest.json','receiver_ce.jsonl','probe_plans.json','fixed_probes.pt'] + [f'raw_round_{i:02d}.pt' for i in range(1,6)]
    result['raw_evidence_sha256'] = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    result['analysis_note'] = 'Restricted-class sensitivity cohort/region labels use restricted delta_ce, not original all-class outcome. No raw training, pairing, Q or fixed protocol change.'
    return result, records


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('root',type=Path); p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args(); result,records=analyze(opt.root)
    opt.output.mkdir(parents=True,exist_ok=False)
    (opt.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    fields = sorted({k for r in records for k in r if k not in ('classes','q_valid_classes')})
    with (opt.output/'pairs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    cells = [{**{k:r[k] for k in ('round','region','stratum','sender','receiver','sender_slot','receiver_slot')},**c}
             for r in records for c in r.get('classes',[])]
    with (opt.output/'classes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for c in cells for k in c}),lineterminator='\n')
        writer.writeheader(); writer.writerows(cells)
    print(json.dumps({'pairs':len(records),'strata':{k:{n:v[n] for n in ('valid_pairs','outcomes','auc','incremental_status')}
                      for k,v in result['by_stratum'].items()}},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
