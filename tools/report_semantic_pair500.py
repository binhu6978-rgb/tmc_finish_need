"""Fixed descriptive analysis of completed 500-round paired trajectories."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.analyze_semantic_a07_iv_e import write_csv,mean,points_from,associations,incremental,interval,cluster_weights
import numpy as np
from semcons_a07_v2.pair500 import verify_pair,sampled_rounds


def shares(records,path):
    regression={r['round']:r for r in map(json.loads,(path/'regression.jsonl').read_text().splitlines())}
    metrics={int(r['round']):json.loads(r['metrics_json']) for r in csv.DictReader((path/'method_diagnostics.csv').open())}
    result=[]
    for record in records:
        rd=record['round'];slots=regression[rd]['slots']
        for method,change in [('V2',record['regions']),('permuted',record['counterfactual_fusion']['permuted'])]:
            changed={r['region']:r for r in change.get('regions',[]) if 'pi_new' in r}
            for region in metrics[rd]['server']['a07_baseline']['regions']:
                ids=region['slots']
                if not ids:continue
                pi=defaultdict(float)
                for i,w in zip(ids,region['applied_slot_shares']):pi[sum(slots[i][0]>=b for b in (40,70))]+=w
                after=changed.get(region['region_id'],{}).get('pi_new',pi)
                for g,v in pi.items():result.append(dict(round=rd,method=method,region=region['region_id'],tier=g,
                    pi_a07=v,pi_after=after.get(str(g),after.get(g,v)),
                    delta_pi=after.get(str(g),after.get(g,v))-v))
    return result


def utility_summary(rows,protocol):
    result={};rng=np.random.default_rng(protocol['analysis_seed'])
    for method in ('V2','no_sm','no_rm','permuted','A07_L2_matched'):
        selected=[r for r in rows if r['method']==method]
        active=[r for r in selected if r['v2_nonneutral']]
        for label,items in [('all',selected),('active',active)]:
            # Repeated recipients are averaged within round before descriptive means.
            grouped=defaultdict(list)
            for r in items:grouped[r['round']].append(r['utility_vs_a07'])
            draws=[]
            for _ in range(protocol['bootstrap_repetitions']):
                w=cluster_weights(items,rng)
                if len(w) and w.sum():draws.append(float(np.average([r['utility_vs_a07'] for r in items],weights=w)))
            result.setdefault(method,{})[label]=dict(rounds=len(grouped),responses=len(items),
                mean_utility=mean([mean(v) for v in grouped.values()]),cluster95=interval(draws),
                positive_responses=sum(r['utility_vs_a07']>0 for r in items),negative_responses=sum(r['utility_vs_a07']<0 for r in items))
    return result


def analyze(run,out,protocol):
    out.mkdir(parents=True,exist_ok=False)
    records=[json.loads(l) for l in (run/'iv_e_rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in records]==sampled_rounds(protocol)
    f7=[v for r in records for v in r['figure7']]
    f8=[v|r['norms'][v['method']]|dict(v2_nonneutral=r['norms']['V2']['difference_from_a07_l2']>0)
        for r in records for v in r['figure8']]
    write_csv(out/'figure7_raw.csv',f7);write_csv(out/'figure8_raw.csv',f8)
    write_csv(out/'figure8_source_shares.csv',shares(records,run))
    result=dict(raw_pairs=len(f7),sampled_rounds=len(records),probe_sgd_independent=False,cohorts={})
    points=[]
    for part in ('confirmation','development'):
        matched=[r for r in f7 if r['cohort']==part and r['U'] is not None and r['common_utility'] is not None]
        allclasses=[dict(r,common_utility=r['utility']) for r in f7 if r['cohort']==part and r['U'] is not None]
        pp=points_from(matched);aa=points_from(allclasses)
        points.extend(dict(p,cohort=part,scope='common') for p in pp)
        points.extend(dict(p,cohort=part,scope='all') for p in aa)
        stage={}
        for name,(a,b) in protocol['stages'].items():
            subset=[p for p in pp if a<=p['round']<=b]
            stage[name]=dict(source_round_units=len(subset),rounds=len({p['round'] for p in subset}),
                tiers=dict(Counter(p['source_tier'] for p in subset)),association=associations(subset,protocol))
        # Same-injection common/noncommon decomposition, not post-hoc class selection.
        errors=[]
        for r in matched:
            fraction=r['common_sample_fraction'];other=r['noncommon_utility']
            if other is not None:errors.append(abs(r['utility']-fraction*r['common_utility']-(1-fraction)*other))
        result['cohorts'][part]=dict(source_round_units=len(pp),rounds=len({p['round'] for p in pp}),
            pairs=len(matched),common_association=associations(pp,protocol),all_association=associations(aa,protocol),
            stages=stage,decomposition_max_error=max(errors,default=0),
            common_sample_fraction_mean=mean([r['common_sample_fraction'] for r in matched]),
            current_gain_vs_evaluation_gain=dict(
                source_own_training_mean=mean([r['source_own_training_gain'] for r in matched]),
                source_own_evaluation_mean=mean([r['source_own_evaluation_gain'] for r in matched if r['source_own_evaluation_gain'] is not None])),
            incremental_leave_round=incremental(matched,'round'),incremental_receiver_groups=incremental(matched,'receiver'),
            same_input_full=utility_summary([r for r in f8 if r['cohort']==part],protocol))
    write_csv(out/'figure7_source_round.csv',points)
    semantic=[json.loads(l) for l in (run/'semantic_v2_rounds.jsonl').read_text().splitlines()]
    assert len(semantic)==500
    result.update(semantic_valid_rounds=sum(r['decision']['valid'] for r in semantic),
        applied_nonneutral_rounds=sum(r['fusion']['difference_from_a07_l2']>0 for r in semantic),
        applied_fallback_reasons=dict(Counter(r['fusion']['reason'] for r in semantic)),
        sampled_virtual_v2_active_rounds=sum(r['norms']['V2']['difference_from_a07_l2']>0 for r in records),
        receiver_forward_examples=sum(r['receiver_forward_examples'] for r in records),
        diagnostic_seconds=sum(r['seconds'] for r in records))
    (out/'diagnostic_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def draw(out,performance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    points=list(csv.DictReader((out/'active/figure7_source_round.csv').open()))
    fig,ax=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for a,scope in zip(ax,('common','all')):
        for g,color in enumerate(('#3579A8','#DC8B2C','#3A8B66')):
            values=[r for r in points if r['scope']==scope and r['cohort']=='confirmation' and int(r['source_tier'])==g]
            a.scatter([float(r['U']) for r in values],[float(r['utility']) for r in values],s=15,alpha=.65,c=color,label=['Low','Middle','High'][g])
        a.axhline(0,color='.5',lw=.7,ls='--');a.set(xlabel='Frozen complementarity U',ylabel='Recipient CE decrease',title=scope.capitalize()+' evaluation classes')
        a.legend(frameon=False,fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'figure7_semantic_utility.{ext}')
    plt.close(fig)
    f8=list(csv.DictReader((out/'active/figure8_raw.csv').open()))
    fig,ax=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    rows=list(csv.DictReader((out/'active/figure8_source_shares.csv').open()))
    for g in range(3):
        selected=[r for r in rows if r['method']=='V2' and int(r['tier'])==g]
        ax[0].scatter([int(r['round']) for r in selected],[100*float(r['delta_pi']) for r in selected],s=9,alpha=.45,label=['Low','Middle','High'][g])
    ax[0].set(xlabel='Round (fixed diagnostic windows)',ylabel='Regional tier-share change (pp)',title='Same-input V2 share adjustment');ax[0].legend(frameon=False,fontsize=7)
    for method in ('V2','permuted','A07_L2_matched'):
        selected=[r for r in f8 if r['method']==method and r['cohort']=='confirmation'];group=defaultdict(list)
        for r in selected:group[int(r['round'])].append(float(r['utility_vs_a07']))
        ax[1].plot(sorted(group),[mean(group[r]) for r in sorted(group)],'.',ms=3,label=method)
    ax[1].axhline(0,color='.5',lw=.7,ls='--');ax[1].set(xlabel='Round',ylabel='CE(A07) − CE(counterfactual)',title='Paired Full-probe utility');ax[1].legend(frameon=False,fontsize=7)
    for ext in ('png','pdf'):fig.savefig(out/f'figure8_aggregation_behavior.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for mode,rows in performance.items():
        for a,key in zip(ax,('full_accuracy','full_ce')):a.plot([r['round'] for r in rows],[r[key] for r in rows],lw=.8,label=mode)
    for a,label in zip(ax,('Full Accuracy (%)','Full CE')):a.set(xlabel='Round',ylabel=label);a.legend(frameon=False)
    fig.savefig(out/'accuracy_ce.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);opt=p.parse_args()
    protocol=json.loads((ROOT/'configs/semantic_a07_pair500.json').read_text());root=opt.root
    verification=verify_pair(root,500);out=root/'report';out.mkdir(exist_ok=False)
    results={};performance={};table=[]
    for mode in ('neutral','active'):
        path=root/mode/'training/a07'/f'{mode}_500_seed1'
        results[mode]=analyze(path,out/mode,protocol)
        performance[mode]=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((path/'accuracy_each_round.csv').open())]
        summary=json.loads((path/'summary.json').read_text())
        table.append(dict(method='A07-only' if mode=='neutral' else 'Semantic-A07 V2',**{k:summary[k] for k in (
            'best_full_accuracy','best_round','final_full_accuracy','final_full_ce','last20_full_mean','last50_full_mean')},
            peak_label='test-oracle peak'))
    write_csv(out/'performance_comparison.csv',table);draw(out,performance)
    (out/'summary.json').write_text(json.dumps(dict(verification=verification,results=results),indent=2,allow_nan=False))
    text=['# Frozen A07 / V2: 500-round mechanism evidence','',
        'Single seed1, official-test exploratory trajectories; training-side utility probes are disjoint from semantic measurement but NOT from SGD. CRD omitted by user.','',
        '|Method|test-oracle peak|Round|Final|Last20|Last50|Final CE|','|---|---:|---:|---:|---:|---:|---:|']
    for r in table:text.append(f"|{r['method']}|{r['best_full_accuracy']:.4f}|{r['best_round']}|{r['final_full_accuracy']:.4f}|{r['last20_full_mean']:.4f}|{r['last50_full_mean']:.4f}|{r['final_full_ce']:.6f}|")
    text+=['','Three separate questions must be read separately: score association (source-tier/round units), same-input counterfactual utility, and whole-trajectory Accuracy. CSVs preserve neutral and adverse cases. Memory ablations retain the full-V2 validity mask, so they do not establish the effect of removing all history. Adapter remains auxiliary. No score was tuned.','']
    for mode,s in results.items():
        c=s['cohorts']['confirmation'];u=c['common_association']['U'];v=c['same_input_full']['V2']['active']
        text += [f"## {mode}",f"Independent semantic source-round units: {c['source_round_units']} across {c['rounds']} rounds. Matched-class Spearman={u['spearman']}, clustered95={u['spearman_cluster95']}. All-class Spearman={c['all_association']['U']['spearman']}.",
            f"Same-input V2 active-round Full CE utility={v['mean_utility']}, clustered95={v['cluster95']}. Applied nonneutral rounds={s['applied_nonneutral_rounds']}/500. See diagnostic_summary.json for current/memory/reliability ablations and held-out-round/receiver-group prediction errors.",'']
    text+=['These observations do not establish multi-seed generalization. Temporal and physical-client dependence, training-side evaluation bias and source-versus-region attribution remain explicit limitations. No further run is scheduled.']
    (out/'IV_E_500_REPORT.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':main()
