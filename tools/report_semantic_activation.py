"""Activation-only plots: task/memory/weight data, never select by utility or accuracy."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def extract(records,geometry,protocol):
    timeline=[];categories=[];sources=[];shares=[]
    total=sum(r['numel'] for r in geometry);allshared={r['id'] for r in geometry if len(r['profiles'])==7}
    for row in records:
        rd=row['round'];decision=row['decision'];fusion=row['fusion'];group=defaultdict(list)
        timeline.append(dict(round=rd,valid_evidence=decision['valid'],reason=fusion['reason'],
            common_classes=len(decision['classes']),changed_parameters=fusion['changed_parameters'],
            full_parameter_numel=total,changed_fraction=fusion['changed_parameters']/total,
            difference_l2=fusion['difference_from_a07_l2'],active=fusion['difference_from_a07_l2']>0,
            anchor_count=row['tier_anchor_count'],history_records=row['tier_history_records']))
        if decision['valid']:
            for d in decision['details']:
                assert d['reference_round']<rd
                assert math.isclose(d['u'],d['rho']*min(d['d'],d['a']),rel_tol=1e-12,abs_tol=1e-14)
                record=dict(round=rd,**d);categories.append(record);group[d['tier']].append(record)
            for g,values in group.items():
                record=dict(round=rd,tier=g,classes=len(values))
                for key in ('q_pre','q_post','d','a','rho','u'):
                    record[key]=float(np.mean([d[key] for d in values]))
                record['memory_limited']=float(np.mean([min(d['d'],d['a']) for d in values]))
                record['U']=decision['scores'].get(str(g),decision['scores'].get(g))
                assert math.isclose(record['U'],record['u'],abs_tol=1e-14)
                sources.append(record)
        for region in fusion['regions']:
            if 'pi_new' not in region:continue
            pi={int(g):v for g,v in region['pi_a07'].items()};new={int(g):v for g,v in region['pi_new'].items()}
            score={int(g):v for g,v in decision['scores'].items()}
            center=sum(pi[g]*score[g] for g in pi)
            assert abs(sum(new.values())-1)<1e-12 and all(v>=0 for v in new.values())
            for g in pi:
                expected=(1-protocol['semantic_lambda'])*pi[g]+protocol['semantic_lambda']*pi[g]*(1+score[g])/(1+center)
                assert math.isclose(new[g],expected,abs_tol=1e-14)
                shares.append(dict(round=rd,region=region['region'],tier=g,U=score[g],
                    centered_U=score[g]-center,pi_before=pi[g],pi_after=new[g],
                    delta_pp=100*(new[g]-pi[g]),relative_change_percent=100*(new[g]/pi[g]-1),
                    all_profiles_shared=region['region'] in allshared,
                    feature_numel=region['feature_numel'],reason=region['reason']))
    cases=[]
    for stage,(a,b) in protocol['stages'].items():
        chosen=next((r for r in timeline if a<=r['round']<=b and r['active']),None)
        eligible=[r for r in shares if chosen and r['round']==chosen['round'] and r['all_profiles_shared']]
        if not eligible:cases.append(dict(stage=stage,round=None,reason='no_qualified_case_in_fixed_window'));continue
        region=min(r['region'] for r in eligible)
        cases.extend(dict(stage=stage,**r) for r in eligible if r['region']==region)
    summary=dict(rounds=len(timeline),active_rounds=sum(r['active'] for r in timeline),
        valid_rounds=sum(r['valid_evidence'] for r in timeline),fallback_counts=dict(Counter(r['reason'] for r in timeline)),
        source_round_observations=len(sources),used_category_observations=len(categories),
        semantic_memory_numerically_limits=sum(min(r['d'],r['a'])<r['d'] for r in categories),
        reliability_numerically_modifies=sum(r['u']!=min(r['d'],r['a']) for r in categories),
        changed_parameter_max=max(r['changed_parameters'] for r in timeline),
        difference_l2_max=max(r['difference_l2'] for r in timeline),
        region_share_change_pp_max=max((abs(r['delta_pp']) for r in shares),default=0),
        case_rule=protocol['illustrative_case_rule'],
        scope='computational participation and intervention, no utility or accuracy attribution')
    return timeline,categories,sources,shares,cases,summary


def draw(output,timeline,categories,sources,shares):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#367BA8','#DC922E','#26866E'];labels=['Low','Middle','High']
    fig,panels=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained');ax=panels.flatten()
    for g,color in enumerate(colors):
        vals=[r for r in sources if r['tier']==g]
        if not vals:continue
        for key,offset,marker in [('q_pre',-.12,'o'),('q_post',.12,'s')]:
            values=[r[key] for r in vals];lo,hi=np.quantile(values,[.25,.75]);m=np.mean(values)
            ax[0].plot([lo,hi],[g+offset,g+offset],c=color,lw=3,alpha=.55)
            ax[0].scatter(m,g+offset,marker=marker,c=color,s=28,label={'q_pre':'Before local SGD','q_post':'After local SGD'}[key] if g==0 else None)
        for i,key in enumerate(('d','memory_limited','U')):
            ax[1].bar(g+(i-1)*.23,np.mean([r[key] for r in vals]),width=.21,
                color=['#98BED4','#5E99B9','#236986'][i],label=['Current gain d','Memory-limited m','Calibrated U'][i] if g==0 else None)
    ax[0].set(yticks=range(3),yticklabels=labels,xlim=(0,1),xlabel='Correct-class task probability',
        title='(a) Fixed pre/post task measurement')
    ax[0].legend(frameon=False,fontsize=7,loc='upper left');ax[0].text(.03,.42,'Markers: source-round means\nBars: interquartile range',
        transform=ax[0].transAxes,ha='left',va='top',fontsize=6.7)
    ax[1].set(xticks=range(3),xticklabels=labels,ylabel='Mean source-round score',title='(b) Actual score construction')
    ax[1].legend(frameon=False,fontsize=7)
    if categories:
        ages=[max(r['own_reliability']['age'],r['reference_reliability']['age']) for r in categories]
        hb=ax[2].hexbin(ages,[r['rho'] for r in categories],gridsize=(25,15),bins='log',mincnt=1,cmap='Blues',linewidths=0)
        fig.colorbar(hb,ax=ax[2],label='Used tier/category observations',pad=.02)
    ax[2].set(xlabel='Oldest supporting history age (rounds)',ylabel='Evidence-use coefficient rho',
        title='(c) Past-only reliability observations',ylim=(0,1))
    n=len(timeline)
    for g,color in enumerate(colors):
        lookup={r['round']:r['U'] for r in sources if r['tier']==g}
        y=[lookup.get(rd,np.nan) for rd in range(1,n+1)]
        ax[3].plot(range(1,n+1),y,'.-',color=color,ms=1.8,lw=.55,label=labels[g])
    ax[3].set_yscale('symlog',linthresh=1e-4)
    ax[3].set(xlabel='Communication round',ylabel='U (symlog; linear below 1e-4)',title='(d) Complete source-score trajectory')
    ax[3].legend(frameon=False,fontsize=7,ncol=3)
    for ext in ('png','pdf'):fig.savefig(output/f'figure7_score_formation.{ext}')
    plt.close(fig)

    fig,panels=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained');ax=panels.flatten()
    for g,color in enumerate(colors):
        vals=[r for r in shares if r['tier']==g]
        ax[0].scatter([r['centered_U'] for r in vals],[r['relative_change_percent'] for r in vals],
            c=color,s=5,alpha=.35,label=labels[g],rasterized=True)
    ax[0].axhline(0,color='.5',lw=.5);ax[0].axvline(0,color='.5',lw=.5)
    ax[0].set(xlabel='U minus region-weighted mean U',ylabel='Relative source-share change (%)',
        title='(a) Score-to-contribution mapping');ax[0].legend(frameon=False,fontsize=7)
    for label,color,test in [('Adjusted','#26866E',lambda r:r['active']),('Neutral','#A8AFB8',lambda r:not r['active'])]:
        vals=[r for r in timeline if test(r)]
        ax[1].scatter([r['round'] for r in vals],[int(r['active']) for r in vals],c=color,s=9,label=f'{label}: {len(vals)}')
    ax[1].set(yticks=[0,1],yticklabels=['Neutral','Adjusted'],ylim=(-.35,1.4),xlabel='Communication round',
        title='(b) All-round execution record');ax[1].legend(frameon=False,fontsize=7,ncol=2)
    ax[2].plot([r['round'] for r in timeline],[100*r['changed_fraction'] for r in timeline],lw=.65,c='#367BA8')
    ax[2].set(xlabel='Communication round',ylabel='Full coordinates changed (%)',title='(c) Effective feature-parameter intervention')
    ax[3].plot([r['round'] for r in timeline],[r['difference_l2'] for r in timeline],lw=.65,c='#26866E')
    ax[3].set_yscale('symlog',linthresh=1e-6)
    ax[3].set(xlabel='Communication round',ylabel='Fusion-induced L2 (symlog; linear <1e-6)',title='(d) Fusion-induced displacement')
    for ext in ('png','pdf'):fig.savefig(output/f'figure8_aggregation_activation.{ext}')
    plt.close(fig)


def read_plot_csv(path):
    """Restore typed scalar/nested values without selecting or changing rows."""
    def parse(value):
        if value == '':return None
        if value in ('True','False'):return value == 'True'
        try:return json.loads(value)
        except (json.JSONDecodeError,TypeError):return value
    with path.open(newline='',encoding='utf-8') as stream:
        return [{k:parse(v) for k,v in row.items()} for row in csv.DictReader(stream)]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--redraw-csv',action='store_true',help='Redraw all original CSV rows only; write a new output directory')
    opt=p.parse_args();root=opt.root
    if opt.redraw_csv:
        assert opt.output is not None
        opt.output.mkdir(parents=True,exist_ok=False)
        draw(opt.output,*[read_plot_csv(root/f'{name}.csv') for name in ('all_rounds','used_categories','source_rounds','region_shares')])
        return
    manifest=json.loads((root/'manifest.json').read_text())
    protocol=manifest['identity']['protocol'];n=protocol['rounds']
    run=root/'training/a07'/f'active_{n}_seed1'
    complete=json.loads((run/'completed.json').read_text());assert complete['success'] and complete['rounds_completed']==n
    metrics=list(csv.DictReader((run/'accuracy_each_round.csv').open()));assert len(metrics)==n
    assert all(math.isfinite(float(r[k])) for r in metrics for k in ('full_accuracy','full_ce'))
    records=[json.loads(l) for l in (run/'semantic_v2_rounds.jsonl').read_text().splitlines()]
    assert [r['round'] for r in records]==list(range(1,n+1))
    regression=[json.loads(l) for l in (run/'regression.jsonl').read_text().splitlines()]
    assert [r['round'] for r in regression]==list(range(n+1))
    diagnostics=[json.loads(l) for l in (run/'iv_e_rounds.jsonl').read_text().splitlines()]
    expected=sorted(set(protocol['diagnostic_rounds'])|{r for a,b in protocol['stages'].values() for r in range(a,b+1,protocol['window_stride'])})
    assert [r['round'] for r in diagnostics]==expected
    assert json.loads((run/'recovery_latest.json').read_text())['round']==n and (run/'recovery_latest.pt').is_file()
    geometry=json.loads((run/'manifest.json').read_text())['structural_regions']
    data=extract(records,geometry,protocol)
    out=opt.output or root/'activation_report';out.mkdir(parents=True,exist_ok=False)
    for name,rows in zip(('all_rounds','used_categories','source_rounds','region_shares','fixed_examples'),data[:5]):
        write_csv(out/f'{name}.csv',rows)
    summary=data[-1]|dict(source_commit=manifest['commit'],config_sha256=manifest['configuration_sha256'],
        partition_sha256=manifest['identity']['partition'],adapter_in_score=False,training_side_probes=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));draw(out,*data[:4])
    (out/'MECHANISM_ACTIVATION_REPORT.md').write_text(
        f"# Mechanism operation over{n} real rounds\n\n"
        f"Actual adjustment rounds: {summary['active_rounds']}/{n}. "
        f"Semantic Memory numerically limits {summary['semantic_memory_numerically_limits']} used category observations; "
        f"Reliability modifies {summary['reliability_numerically_modifies']} observations.\n\n"
        "Figure7 documents task evidence, historical limitation, reliability and U. Figure8 documents internal source-share changes, all-round activation, actual coordinate changes and fusion displacement relative to the same-input pre-fusion state. These are execution measurements; they do not establish update utility or performance superiority. Adapter is auxiliary. All neutral/missing records are retained, and case selection never uses accuracy or utility.\n",
        encoding='utf-8')


if __name__=='__main__':main()
