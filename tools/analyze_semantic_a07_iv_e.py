"""Prespecified descriptive analysis; no fitting of U, no training launch."""
import argparse
from collections import defaultdict,Counter
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def mean(values):return float(np.mean(values)) if len(values) else None


def rank(x):
    order=np.argsort(x,kind='stable');r=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2;i=j
    return r


def corr(x,y,w=None,spearman=False):
    x=np.array(x,float);y=np.array(y,float);w=np.ones(len(x)) if w is None else np.array(w,float)
    if len(x)<3 or w.sum()==0:return None
    if spearman:x=rank(x);y=rank(y)
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    den=np.sqrt(np.dot(w,x*x)*np.dot(w,y*y))
    return float(np.dot(w,x*y)/den) if den>1e-15 else None


def auc(x,y):
    a=[v for v,z in zip(x,y) if z>0];b=[v for v,z in zip(x,y) if z<=0]
    return mean([(u>v)+.5*(u==v) for u in a for v in b]) if a and b else None


def interval(values):
    x=[v for v in values if v is not None and math.isfinite(v)]
    return [float(v) for v in np.quantile(x,[.025,.975])] if x else [None,None]


def cluster_weights(rows,rng):
    rounds={r['round'] for r in rows};people={r['receiver'] for r in rows}
    people.update(c for r in rows for c in r.get('source_clients',[]))
    rw={r:rng.poisson(1) for r in sorted(rounds)};pw={c:rng.poisson(1) for c in sorted(people)}
    return np.array([rw[r['round']]*pw[r['receiver']]*
        (np.mean([pw[c] for c in r['source_clients']]) if r.get('source_clients') else 1.) for r in rows],float)


def points_from(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['round'],r['source_tier']].append(r)
    points=[]
    for (rd,g),values in sorted(groups.items()):
        regions=defaultdict(list)
        for v in values:regions[v['region']].append(v['common_utility'])
        points.append(dict(round=rd,source_tier=g,utility=mean([mean(v) for v in regions.values()]),
            **{k:values[0][k] for k in ('D','M','U','rho','permuted_U')},pairs=len(values),regions=len(regions),
            receivers=sorted({v['receiver'] for v in values}),
            source_clients=sorted({c for v in values for c in v['source_clients']})))
    return points


def point_weights(points,rng):
    rd={r:rng.poisson(1) for r in sorted({p['round'] for p in points})}
    ids=sorted({c for p in points for c in p['source_clients']+p['receivers']});pw={c:rng.poisson(1) for c in ids}
    return np.array([rd[p['round']]*mean([pw[c] for c in p['source_clients']])*mean([pw[c] for c in p['receivers']]) for p in points])


def associations(points,protocol):
    result={};rng=np.random.default_rng(protocol['analysis_seed'])
    for score in ('D','M','U','permuted_U'):
        x=[p[score] for p in points];y=[p['utility'] for p in points]
        draws=[]
        for _ in range(protocol['bootstrap_repetitions']):draws.append(corr(x,y,point_weights(points,rng),True))
        result[score]=dict(pearson=corr(x,y),spearman=corr(x,y,spearman=True),spearman_cluster95=interval(draws),
            benefit_auc=auc(x,y),source_round_units=len(points))
    return result


def incremental(rows,fold_kind,fold_getter=None):
    # Raw responses are weighted inversely by repeated source-round count.
    counts=Counter((r['round'],r['source_tier']) for r in rows)
    weights=np.array([1/counts[r['round'],r['source_tier']] for r in rows])
    folds=np.array([fold_getter(r) if fold_getter else r['round'] if fold_kind=='round' else r['receiver']%5 for r in rows])
    if len(set(folds))<3:return dict(status='unable_to_judge',reason='fewer_than_three_nonempty_group_folds')
    def features(r):
        return [math.log1p(r['source_delta_l2']),r['direction_cosine'] or 0.,float(r['direction_cosine'] is None),
            float(r['source_tier']==1),float(r['source_tier']==2),r['receiver_tier'],r['receiver_profile'],
            math.log1p(r['source_mass']),math.log1p(r['region_numel']),r['source_receiver_label_js'],
            r['source_profile_mean'],r['source_own_gain'],r['before_ce']]
    y=np.array([r['common_utility'] for r in rows]);outputs={};details=[]
    for name,extra in [('controls',[]),('current_D',['D']),('D_plus_memory',['D','M']),
                       ('D_M_plus_reliability',['D','M','U']),('D_M_plus_permuted',['D','M','permuted_U'])]:
        x=np.array([features(r)+[r[k] for k in extra] for r in rows]);prediction=np.zeros(len(rows))
        for fold in sorted(set(folds)):
            tr=folds!=fold;te=~tr
            if tr.sum()<3: return dict(status='unable_to_judge',reason='too_few_training_rows')
            mu=np.average(x[tr],axis=0,weights=weights[tr]);sd=np.sqrt(np.average((x[tr]-mu)**2,axis=0,weights=weights[tr]));sd[sd<1e-12]=1
            a=np.column_stack([np.ones(tr.sum()),(x[tr]-mu)/sd]);b=np.column_stack([np.ones(te.sum()),(x[te]-mu)/sd])
            at=torch.tensor(a,dtype=torch.float64);wt=torch.tensor(weights[tr],dtype=torch.float64)
            penalty=torch.eye(a.shape[1],dtype=torch.float64);penalty[0,0]=0
            coefficient=torch.linalg.solve(at.T@(at*wt[:,None])+penalty,
                at.T@(torch.tensor(y[tr],dtype=torch.float64)*wt)).numpy()
            prediction[te]=b@coefficient
        mse=float(np.average((prediction-y)**2,weights=weights))
        outputs[name]=dict(weighted_mse=mse)
        for i,r in enumerate(rows):details.append(dict(round=r['round'],source_tier=r['source_tier'],receiver=r['receiver'],
            region=r['region'],fold_kind=fold_kind,model=name,observed=y[i],predicted=float(prediction[i]),weight=float(weights[i])))
    baseline=outputs['controls']['weighted_mse']
    for v in outputs.values():v['reduction_vs_controls']=(baseline-v['weighted_mse'])/baseline if baseline>0 else None
    return dict(status='exploratory_only_few_independent_rounds',fold_kind=fold_kind,models=outputs,
        pairs=len(rows),source_round_units=len(counts),rounds=len({r['round'] for r in rows}),
        warning='weighted dependent responses; not independent validation of a score',predictions=details)


def source_shares(records,training):
    regression=[json.loads(l) for l in (training/'regression.jsonl').read_text().splitlines()]
    metrics=list(csv.DictReader((training/'method_diagnostics.csv').open()))
    result=[]
    for record,raw in zip(records,metrics):
        rd=record['round'];slots=regression[rd]['slots']
        base=json.loads(raw['metrics_json'])['server']['a07_baseline']['regions']
        for method in ('A07','V2','no_sm','no_rm','permuted','A07_L2_matched'):
            change=record['regions'] if method=='V2' else record['counterfactual_fusion'].get(method,{})
            changed={v['region']:v for v in change.get('regions',[]) if v['reason']=='semantic_source_blend'} if change else {}
            sizes={v['region']:v['feature_numel'] for v in record['inventory']}
            for r in base:
                if not r['slots']:continue
                pi=defaultdict(float)
                for i,w in zip(r['slots'],r['applied_slot_shares']):pi[sum(slots[i][0]>=b for b in (40,70))]+=w
                new=changed.get(r['region_id'],{}).get('pi_new',dict(pi))
                for g,value in pi.items():
                    after=new.get(str(g),new.get(g,value))
                    result.append(dict(round=rd,method=method,region=r['region_id'],tier=g,
                        feature_numel=sizes.get(r['region_id'],r['weighted_parameter_numel']),
                        pi_a07=value,pi_after=after,delta_pi=after-value))
    return result


def analyze(run,output):
    complete=json.loads((run/'diagnostic_completed.json').read_text());assert complete['success'] and complete['rounds']==20
    training=run/'training/a07/active_20_seed1';records=[json.loads(l) for l in (training/'iv_e_rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in records]==list(range(1,21))
    protocol=json.loads((run/'protocol.json').read_text());output.mkdir(parents=True,exist_ok=True)
    figure7=[row for r in records for row in r['figure7']];figure8=[]
    active=[r['round'] for r in records if r['norms']['V2']['difference_from_a07_l2']>0]
    for r in records:
        for row in r['figure8']:figure8.append(row|r['norms'][row['method']]|dict(v2_nonneutral=r['round'] in active))
    primary=[r for r in figure7 if r['U'] is not None and r['common_utility'] is not None]
    points=points_from(primary);shares=source_shares(records,training)
    # Strict removal of either required memory is provably the existing neutral
    # path, not a new forward or an imputed utility: the evaluated model is A07.
    baseline_rows=[r for r in figure8 if r['method']=='A07']
    baseline_shares=[r for r in shares if r['method']=='A07']
    for name in ('no_sm_strict_fallback','no_rm_strict_fallback'):
        figure8.extend(dict(r,method=name,evaluation_reused_from='A07 exact missing-memory neutral result') for r in baseline_rows)
        shares.extend(dict(r,method=name) for r in baseline_shares)
    report=dict(protocol=protocol,completion=complete,source=json.loads((run/'diagnostic_manifest.json').read_text()),
        figure7=dict(total_pairs=len(figure7),semantic_valid_pairs=sum(r['U'] is not None for r in figure7),
            primary_pairs=len(primary),missing_common_probe_classes=sum(r['U'] is not None and r['common_utility'] is None for r in figure7),
            source_round_units=len(points),effective_rounds=sorted({r['round'] for r in primary}),
            distinct_source_clients=len({c for r in primary for c in r['source_clients']}),
            distinct_receivers=len({r['receiver'] for r in primary}),
            primary_benefit_pairs=sum(r['common_utility']>0 for r in primary),
            primary_harm_pairs=sum(r['common_utility']<0 for r in primary),
            source_round_mean_utility=mean([p['utility'] for p in points]),
            direction_negative_pairs=sum(r['direction_cosine'] is not None and r['direction_cosine']<0 for r in primary),
            association=associations(points,protocol),bins=[],incremental={},
            by_source_tier={str(g):dict(units=len([p for p in points if p['source_tier']==g]),
                mean_U=mean([p['U'] for p in points if p['source_tier']==g]),
                mean_utility=mean([p['utility'] for p in points if p['source_tier']==g]),
                within_tier_spearman=corr([p['U'] for p in points if p['source_tier']==g],
                    [p['utility'] for p in points if p['source_tier']==g],spearman=True)) for g in range(3)},
            all_probe_secondary_association=associations(points_from([dict(r,common_utility=r['utility'])
                for r in figure7 if r['U'] is not None]),protocol),
            U_round_spearman=corr([p['U'] for p in points],[p['round'] for p in points],spearman=True),
            common_probe_class_counts=dict(Counter(r['common_class_count'] for r in primary))),
        figure8=dict(nonneutral_rounds=active,neutral_fraction=(20-len(active))/20,methods={}),
        strict_memory_controls='exact original missing-memory neutral A07; reuse same evaluated model, not new GPU forwards',
        costs=dict(diagnostic_seconds=sum(r['seconds'] for r in records),
            recipient_forward_examples=sum(r['receiver_forward_examples'] for r in records),
            hypothetical_region_downlink_bytes=sum(r['hypothetical_tier_delta_downlink_bytes'] for r in records),
            hypothetical_class_feedback_bytes=sum(r['hypothetical_class_feedback_bytes'] for r in records)))
    rng=np.random.default_rng(protocol['analysis_seed'])
    for lo,hi in zip(protocol['bins'][:-1],protocol['bins'][1:]):
        selected=[p for p in points if lo<=p['U']<hi or (hi==1 and p['U']==1)]
        draws=[]
        for _ in range(protocol['bootstrap_repetitions']):
            w=point_weights(selected,rng)
            if len(w) and w.sum()>0:draws.append(float(np.average([p['utility'] for p in selected],weights=w)))
        report['figure7']['bins'].append(dict(lower=lo,upper=hi,n=len(selected),mean=mean([p['utility'] for p in selected]),ci95=interval(draws)))
    predictions=[]
    for fold in ('round','receiver'):
        v=incremental(primary,fold);predictions.extend(v.pop('predictions',[]));report['figure7']['incremental'][fold]=v
    for method in sorted({r['method'] for r in figure8}):
        all_rows=[r for r in figure8 if r['method']==method];selected=[r for r in all_rows if r['v2_nonneutral']]
        draws=[]
        for _ in range(protocol['bootstrap_repetitions']):
            w=cluster_weights(selected,rng)
            if len(w) and w.sum()>0:draws.append(float(np.average([p['utility_vs_a07'] for p in selected],weights=w)))
        report['figure8']['methods'][method]=dict(all20_mean_utility_vs_a07=mean([r['utility_vs_a07'] for r in all_rows]),
            active_mean_utility_vs_a07=mean([r['utility_vs_a07'] for r in selected]),active_cluster95=interval(draws),
            active_benefit_probes=sum(r['utility_vs_a07']>0 for r in selected),active_harm_probes=sum(r['utility_vs_a07']<0 for r in selected),
            mean_active_difference_l2=mean([r['difference_from_a07_l2'] for r in selected]),
            mean_active_update_l2=mean([r['update_l2'] for r in selected]))
    write_csv(output/'figure7_raw.csv',figure7);write_csv(output/'figure7_source_round.csv',points)
    write_csv(output/'figure7_source_round_all_probe.csv',points_from([dict(r,common_utility=r['utility'])
        for r in figure7 if r['U'] is not None]))
    write_csv(output/'figure7_predictions.csv',predictions);write_csv(output/'figure8_raw.csv',figure8)
    write_csv(output/'figure8_source_shares.csv',shares)
    (output/'diagnostic_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();result=analyze(opt.run,opt.output);print(json.dumps({k:result[k] for k in ('figure7','figure8')},indent=2))
