"""Standalone publication-candidate plots from IV-E CSV/JSON; no selection."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return list(csv.DictReader(path.open(encoding='utf-8')))


def draw(root):
    summary=json.loads((root/'diagnostic_summary.json').read_text())
    points=read(root/'figure7_source_round.csv');shares=read(root/'figure8_source_shares.csv');rows=read(root/'figure8_raw.csv')
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8.5,'axes.labelsize':9,
        'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'xtick.labelsize':8,'ytick.labelsize':8,'savefig.dpi':300})
    colors=['#3579A8','#DC8B2C','#3A8B66'];names=['Low','Middle','High']
    all_points=read(root/'figure7_source_round_all_probe.csv')
    fig,ax=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained')
    for panel,items in zip(ax[:2],(points,all_points)):
        for g in range(3):
            subset=[p for p in items if int(p['source_tier'])==g]
            if subset:panel.scatter([float(p['U']) for p in subset],[float(p['utility']) for p in subset],
                c=colors[g],s=26,marker=['s','o','^'][g],edgecolors='white',linewidths=.4,label=names[g],zorder=3)
        panel.axhline(0,color='.35',lw=.7,ls='--');panel.set_xlabel(r'Complementarity $U_g$')
        panel.set_xticks([.05,.10,.15,.20,.25]);panel.tick_params(axis='x',labelsize=7)
    ax[0].set(ylabel='Recipient utility (CE decrease)',title='(a) Matched probe classes')
    ax[1].set(ylabel='Recipient utility (CE decrease)',title='(b) All fixed probe classes')
    if points:ax[0].legend(frameon=False,loc='upper left',fontsize=6.5)
    for panel,key,items in zip(ax[:2],('association','all_probe_secondary_association'),(points,all_points)):
        rho=summary['figure7'][key]['U']['spearman'];text='undefined' if rho is None else f'{rho:.3f}'
        panel.text(.98,.02,f"Spearman = {text}\nn={len(items)}; rounds={len({p['round'] for p in items})}",fontsize=6.6,
                   va='bottom',ha='right',transform=panel.transAxes,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    bins=summary['figure7']['bins']
    for i,b in enumerate(bins):
        if b['mean'] is None:continue
        lo,hi=b['ci95'];m=b['mean']
        ax[2].plot([i,i],[lo,hi],color='#3579A8',lw=1.3)
        ax[2].plot([i-.06,i+.06],[lo,lo],color='#3579A8',lw=1);ax[2].plot([i-.06,i+.06],[hi,hi],color='#3579A8',lw=1)
        ax[2].scatter(i,m,color='#3579A8',s=28,zorder=3)
        ax[2].annotate(f"n={b['n']}",(i,hi),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
    ax[2].axhline(0,color='.35',lw=.7,ls='--');ax[2].set_xticks(range(len(bins)),['[0, .1)','[.1, .2)','[.2, 1]'])
    ax[2].tick_params(axis='x',labelsize=7)
    ax[2].set(xlabel='Prespecified U interval',ylabel='Mean matched-class utility',title='(c) Fixed U-bin means')
    ax[2].margins(x=.25,y=.25)
    for a in ax:a.grid(axis='y',color='.9',lw=.5,zorder=0)
    for ext in ('png','pdf'):fig.savefig(root/f'figure7_semantic_utility.{ext}')
    plt.close(fig)

    fig,ax=plt.subplots(2,2,figsize=(7.2,5.0),layout='constrained')
    for g in range(3):
        values=[]
        for rd in range(1,21):
            selected=[s for s in shares if s['method']=='V2' and int(s['round'])==rd and int(s['tier'])==g]
            # Same area denominator for every tier, including absent-tier zero
            # changes, so the three plotted share changes still sum to zero.
            region_sizes={s['region']:float(s['feature_numel']) for s in shares if s['method']=='V2' and int(s['round'])==rd}
            denominator=sum(region_sizes.values())
            values.append(100*sum(float(s['delta_pi'])*float(s['feature_numel']) for s in selected)/denominator if denominator else 0.)
        ax[0,0].plot(range(1,21),values,color=colors[g],marker=['s','o','^'][g],ms=3,lw=1,label=names[g])
    ax[0,0].axhline(0,color='.4',lw=.6);ax[0,0].set(xlabel='Communication round',ylabel='Mean source-share change (pp)',title='(a) V2 feature-area-weighted share changes')
    ax[0,0].legend(frameon=False,ncol=3,fontsize=7,loc='upper left');ax[0,0].set_xticks([1,5,10,15,20])
    methods=['V2','no_sm','no_rm','permuted','A07_L2_matched'];labels=['V2','No anchor','No rho','Shuffled','L2 only']
    palette=['#3579A8','#A17DB8','#DC8B2C','#8C8C8C','#3A8B66']
    for i,(method,color) in enumerate(zip(methods,palette)):
        v=summary['figure8']['methods'][method];m=v['active_mean_utility_vs_a07'];lo,hi=v['active_cluster95']
        if m is None:continue
        ax[0,1].plot([i,i],[lo,hi],color=color,lw=1.3);ax[0,1].scatter(i,m,color=color,s=25,zorder=3)
    ax[0,1].axhline(0,color='.35',lw=.7,ls='--');ax[0,1].set_xticks(range(5),labels,rotation=18)
    ax[0,1].set(ylabel=r'$CE_{A07}-CE_{counterfactual}$',title='(b) Paired utility on V2-active rounds')
    for method,color,label in zip(('V2','permuted','A07_L2_matched'),('#3579A8','#8C8C8C','#3A8B66'),('V2','Shuffled','L2 only')):
        values=[]
        for rd in range(1,21):
            selected=[r for r in rows if r['method']==method and int(r['round'])==rd]
            values.append(float(selected[0]['difference_from_a07_l2']))
        ax[1,0].plot(range(1,21),values,color=color,marker='o',ms=2.5,lw=1,label=label)
    ax[1,0].set(xlabel='Communication round',ylabel='Parameter difference from A07 (L2)',title='(c) Actual counterfactual adjustment sizes')
    ax[1,0].legend(frameon=False,fontsize=7);ax[1,0].set_xticks([1,5,10,15,20])
    for rd in range(1,21):
        selected=[float(r['utility_vs_a07']) for r in rows if r['method']=='V2' and int(r['round'])==rd]
        m=np.mean(selected);lo=min(selected);hi=max(selected)
        ax[1,1].plot([rd,rd],[lo,hi],color='#3579A8',lw=.8);ax[1,1].scatter(rd,m,s=18,color='#3579A8',zorder=3)
    ax[1,1].axhline(0,color='.35',lw=.7,ls='--');ax[1,1].set(xlabel='Communication round',ylabel='V2 utility relative to A07',title='(d) Independent recipient mean and range')
    ax[1,1].set_xticks([1,5,10,15,20])
    for a in ax.flat:
        a.grid(axis='y',color='.9',lw=.5,zorder=0)
        a.ticklabel_format(axis='y',style='sci',scilimits=(-2,3),useMathText=True)
    for ext in ('png','pdf'):fig.savefig(root/f'figure8_aggregation_behavior.{ext}')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);draw(p.parse_args().root)
