"""Offline E03, semantic meaning, confidence and tensor-direction diagnostics.

Never reads official-test scores or changes aggregation decisions.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def mean(xs):
    return statistics.fmean(xs) if xs else None


def stats(xs):
    x = sorted(xs)
    return {'n':len(x),'mean':mean(x),'min':x[0] if x else None,'max':x[-1] if x else None,
            'median':statistics.median(x) if x else None}


def journal(path):
    with path.open() as f:
        return [dict(round=int(r['round']), **json.loads(r['metrics_json'])) for r in csv.DictReader(f)]


def correlation(x, y):
    if len(x)<3: return None
    a=mean(x);b=mean(y);aa=sum((v-a)**2 for v in x);bb=sum((v-b)**2 for v in y)
    return sum((v-a)*(w-b) for v,w in zip(x,y))/(aa*bb)**.5 if min(aa,bb)>0 else None


def within_context_correlation(records, keys, score='q'):
    groups=defaultdict(list)
    for r in records:groups[tuple(tuple(r[k]) if isinstance(r[k],list) else r[k] for k in keys)].append(r)
    xs=[];ys=[];kept=0
    for values in groups.values():
        if len(values)<3:continue
        kept+=1;a=mean([r[score] for r in values]);b=mean([r['local_true_probability_gain'] for r in values])
        xs.extend(r[score]-a for r in values);ys.extend(r['local_true_probability_gain']-b for r in values)
    return {'pearson':correlation(xs,ys),'records':len(xs),'groups':kept,'context':list(keys)}


def m2_summary(rows):
    cs=[c for r in rows for c in r['server']['cosine'] if c is not None]
    weights=[w for r in rows for w in r['server']['conflict_weights']]
    neg=[v for v in cs if v<0]; active=[w for w in weights if w<1-1e-12]
    return {'rounds':len(rows),'negative_cosines':len(neg),'valid_cosines':len(cs),
            'negative_fraction':len(neg)/len(cs) if cs else None,'negative_cosine':stats(neg),
            'nonuniform_weight_rounds':sum(max(r['server']['conflict_weights'])-min(r['server']['conflict_weights'])>1e-12 for r in rows),
            'active_weight_count':len(active),'weight_count':len(weights),'weights':stats(weights),'active_weights':stats(active),
            'mean_weight_suppression':mean([1-w for w in weights]),
            'duplicate_client_rounds':sum(len(set(c for c,p in r['slots']))<len(r['slots']) for r in rows),
            'common_support_fraction_full':stats([r['server']['common_support_fraction_full'] for r in rows])}


def history_summary(rows):
    history={};participants=set();counts=Counter();events=[];rounds=[]
    slots_total=0;all_clients=set();consistent=True
    for row in rows:
        pending=set();rd=Counter();conf=[]
        cids={c for c,p in row['slots']};all_clients.update(cids);slots_total+=len(row['slots'])
        for region in row['server']['regions']:
            d=region.get('direction_risks',{});q=region.get('semantic_risks',{});eligible=set(region.get('eligible_slots',[]))
            for i in region['slots']:
                cid=row['slots'][i][0];key=(cid,region['region_id']);n=history.get(key,0)
                c=region.get('confidence',{}).get(str(i))
                if c is None:continue
                consistent &= abs(c-n/(n+1))<1e-12
                conf.append(c);rd['observed_region_slots']+=1
                if i in eligible:
                    rd['cold_valid_slots' if n==0 else 'warm_valid_slots']+=1;pending.add(key)
                    if n==0:rd['cold_first_participation_slots' if cid not in participants else 'cold_previously_participated_slots']+=1
                if d.get(str(i),0)>0 and q.get(str(cid),0)>0:
                    rd['joint_positive_slots']+=1;rd['joint_cold_slots' if n==0 else 'joint_warm_slots']+=1
                    category='first_client_participation' if cid not in participants else 'previous_client_but_first_region_observation' if n==0 else 'warm_region'
                    events.append({'round':row['round'],'client':cid,'region':region['region_id'],'prior_observations':n,'category':category,
                                   'direction_risk':d[str(i)],'semantic_risk':q[str(cid)],'confidence':c})
        for key in pending:history[key]=history.get(key,0)+1
        consistent &= sum(history.values())==row['server']['history_total_events']
        participants.update(cids);counts.update(rd)
        rounds.append({'round':row['round'],**dict(rd),'confidence':stats(conf),'unique_history_events':len(pending)})
    return {'counts':dict(counts),'joint_events':events,'rounds':rounds,'history_reconstruction_matches':bool(consistent),
            'physical_clients':len(all_clients),'training_slots':slots_total,'distinct_client_region_keys':len(history),
            'total_deduplicated_history_events':sum(history.values()),'history_counts_per_key':stats(list(history.values()))}


def semantic_summary(rows):
    records=[]
    for row in rows:
        local=defaultdict(list)
        for (cid,profile),entry in zip(row['slots'],row['clients']):
            local[cid].append((profile,{x['class']:x for x in entry.get('functional_class_changes',[])}))
        for region in row['server']['regions']:
            sem=region.get('semantic',{})
            for key,q in sem.get('scores',{}).items():
                cid=int(key);classes=sem['valid_classes'][key];gains=[];norms=[]
                for profile,changes in local[cid]:
                    if profile not in region['support_profiles']:continue
                    for y in classes:
                        if y in changes:gains.append(changes[y]['true_class_probability_change']);norms.append(changes[y]['response_delta_l2'])
                records.append({'round':row['round'],'region':region['region_id'],'client':cid,'q':q,
                                'q_original':sem.get('original_scores',{}).get(key,q),
                                'profiles':sorted({p for p,changes in local[cid] if p in region['support_profiles']}),
                                'shared_classes':len(classes),'local_true_probability_gain':mean(gains),
                                'positive_class_gain_fraction':sum(g>0 for g in gains)/len(gains) if gains else None,
                                'response_delta_norm':mean(norms)})
    valid=[r for r in records if r['local_true_probability_gain'] is not None]
    positive=[r for r in valid if r['q']>0];zero=[r for r in valid if r['q']==0]
    return {'records':records,'valid_records':len(valid),'positive_q_count':len(positive),
            'positive_q_and_positive_own_probability_gain':sum(r['local_true_probability_gain']>0 for r in positive),
            'positive_q_own_gain':stats([r['local_true_probability_gain'] for r in positive]),
            'zero_q_own_gain':stats([r['local_true_probability_gain'] for r in zero]),
            'q_vs_own_gain_pearson':correlation([r['q'] for r in valid],[r['local_true_probability_gain'] for r in valid]),
            'q_vs_shared_classes_pearson':correlation([r['q'] for r in valid],[r['shared_classes'] for r in valid]),
            'within_round_region':within_context_correlation(valid,('round','region')),
            'within_round_region_profile':within_context_correlation(valid,('round','region','profiles')),
            'original_q_within_round_region':within_context_correlation(valid,('round','region'),score='q_original'),
            'original_q_within_round_region_profile':within_context_correlation(valid,('round','region','profiles'),score='q_original'),
            'note':'Descriptive dependent region-client observations; own probability gain is neither CE nor cross-client harm.'}


def tensor_summary(rows):
    def summarize(records):
        valid=[r for r in records if r['cosine'] is not None];positive=[r for r in valid if r['cosine']>=0]
        cells=[c for r in valid for c in r['cells'] if c['cosine'] is not None]
        neg=[c for c in cells if c['cosine']<0];hidden=[c for r in positive for c in r['cells'] if c['cosine'] is not None and c['cosine']<0]
        all_hidden_context=[c for r in positive for c in r['cells']]
        pos_dot=sum(max(0,c['dot']) for c in cells);neg_dot=sum(-min(0,c['dot']) for c in cells)
        groups={}
        for family in sorted({c['family'] for c in cells}):
            x=[c for c in cells if c['family']==family];n=[c for c in x if c['cosine']<0]
            groups[family]={'valid':len(x),'negative':len(n),'negative_dot_abs':sum(-c['dot'] for c in n),
                            'negative_delta_energy':sum(c['delta_square'] for c in n),'total_delta_energy':sum(c['delta_square'] for c in x)}
        return {'region_slot_comparisons':len(valid),'negative_regions':sum(r['cosine']<0 for r in valid),
                'nonnegative_regions_with_negative_cells':sum(any(c['cosine'] is not None and c['cosine']<0 for c in r['cells']) for r in positive),
                'nonnegative_regions':len(positive),'valid_tensor_cells':len(cells),'negative_tensor_cells':len(neg),
                'cell_negative_cosines':stats([c['cosine'] for c in neg]),
                'negative_dot_abs':neg_dot,'positive_dot':pos_dot,'negative_to_positive_dot_ratio':neg_dot/pos_dot if pos_dot else None,
                'hidden_negative_delta_energy_fraction':sum(c['delta_square'] for c in hidden)/sum(c['delta_square'] for c in all_hidden_context) if all_hidden_context and sum(c['delta_square'] for c in all_hidden_context)>0 else None,
                'families':groups}
    raw=[dict(r,round=row['round']) for row in rows for r in row['records']]
    # A support region may contain several disjoint rectangles of the SAME tensor.
    # Join them before counting tensor-level conflict; do not inflate counts by tiling.
    records=[]
    for record in raw:
        tensors={}
        for index,cell in enumerate(record['cells']):
            key=cell.get('key',f'fixture_cell_{index}')
            if key not in tensors:tensors[key]=dict(cell,dot=0.,delta_square=0.,reference_square=0.)
            target=tensors[key]
            for name in ('dot','delta_square','reference_square'):target[name]+=cell[name]
        for cell in tensors.values():
            aa=cell['delta_square'];bb=cell['reference_square']
            cell['cosine']=max(-1.,min(1.,cell['dot']/(aa*bb)**.5)) if min(aa,bb)>1e-24 else None
        records.append(dict(record,cells=list(tensors.values())))
    return {'all':summarize(records),'post_initialization_rounds':summarize([r for r in records if r['round']>1]),
            'raw_rectangle_post_initialization':summarize([r for r in raw if r['round']>1]),
            'by_round':{str(row['round']):summarize(row['records']) for row in rows},
            'by_region':{str(k):summarize([r for r in records if r['region']==k]) for k in sorted({r['region'] for r in records})}}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--project',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);p.add_argument('--tensor-probe',type=Path)
    opt=p.parse_args();root=opt.project;opt.output.mkdir(parents=True,exist_ok=False)
    e03=root/'runs/full_only_400_seed1/experiments/m2/E03_full_only_400_seed1/method_diagnostics.csv'
    rows=journal(e03)
    if len(rows)!=400 or [r['round'] for r in rows]!=list(range(1,401)):raise ValueError('E03 must contain exactly 400 completed rounds')
    result={'e03_all':m2_summary(rows),'e03_windows':{f'{a}-{b}':m2_summary(rows[a-1:b]) for a,b in [(1,10),(11,100),(101,200),(201,300),(301,400)]},
            'e03_negative_rounds':[r['round'] for r in rows if any(c is not None and c<0 for c in r['server']['cosine'])],
            'e03_each_round':[dict(round=r['round'],**m2_summary([r])) for r in rows], 'scope':{},'source_sha256':{str(e03):hashlib.sha256(e03.read_bytes()).hexdigest()}}
    base=root/'runs/semcons_activation_full10_seed1'
    for mode in ('s03','s04','s06','s07','s08','s09'):
        path=base/mode/'activation_10/method_diagnostics.csv';x=journal(path)
        result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        result['scope'][mode]={'semantic':semantic_summary(x)}
        if mode in ('s07','s08','s09'):result['scope'][mode]['history']=history_summary(x)
    if opt.tensor_probe:
        path=opt.tensor_probe/'tensor_directions.jsonl';x=[json.loads(s) for s in path.read_text().splitlines()]
        if len(x)!=3 or not (opt.tensor_probe/'s07/activation_3/completed.json').is_file():raise ValueError('Three completed probe rounds required')
        result['tensor_probe']=tensor_summary(x);result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (opt.output/'mechanisms.json').write_text(json.dumps(result,indent=2))
    with (opt.output/'e03_direction_each_round.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['round','negative_cosines','valid_cosines','negative_fraction',
            'nonuniform_weights','active_weights','mean_weight','min_weight','mean_active_weight'],lineterminator='\n')
        writer.writeheader()
        for r in result['e03_each_round']:
            writer.writerow({'round':r['round'],'negative_cosines':r['negative_cosines'],'valid_cosines':r['valid_cosines'],
                'negative_fraction':r['negative_fraction'],'nonuniform_weights':r['nonuniform_weight_rounds'],
                'active_weights':r['active_weight_count'],'mean_weight':r['weights']['mean'],
                'min_weight':r['weights']['min'],'mean_active_weight':r['active_weights']['mean']})
    print(json.dumps({'e03_all':result['e03_all'],'e03_windows':result['e03_windows'],
                      'tensor_probe':result.get('tensor_probe',{}).get('all')},indent=2))


if __name__=='__main__':main()
