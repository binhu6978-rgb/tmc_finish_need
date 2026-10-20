"""Read-only diagnostics and fixed contrasts after the local twelve-run suite."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_tonight_results import decomposition
from tools.report_tonight10 import diagnostic_summary


def analyze(root,output):
    import torch
    state=json.loads((root/'status.json').read_text())
    if state['state']!='completed' or len(state['completed'])!=12 or state['failed']:
        raise ValueError('All twelve actual400-round local runs must finish first')
    output.mkdir(exist_ok=False,parents=True)
    records={};table=[];costs=[];initial=[];schedules=[];curves={}
    full=torch.load(root/'m2/REF_full_only_400_seed1/latest.pt',map_location='cpu',weights_only=False)['global']
    total=sum(v.numel() for v in full.values())
    for name in ['REF']+[f'L{i:02d}' for i in range(1,12)]:
        method='m2' if name=='REF' else name.lower();p=root/method/(name+'_full_only_400_seed1')
        manifest=json.loads((p/'manifest.json').read_text());initial.append(manifest['initial_state_sha256'])
        summary,schedule=diagnostic_summary(p/'method_diagnostics.csv');schedules.append(schedule)
        with (p/'method_diagnostics.csv').open() as f:metrics=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        servers=[m['server'] for m in metrics]
        with (p/'accuracy_each_round.csv').open() as f:rows=list(csv.DictReader(f))
        if len(rows)!=400 or len(servers)!=400:raise ValueError('Missing400 rows')
        curves[name]=[float(r['full_accuracy']) for r in rows]
        summary.update(later_change_decomposition=decomposition(servers[1:]),
            actual_e03_changed_rounds=sum(s['difference_from_e03_l2']>0 for s in servers),
            later_actual_update_l2_mean=statistics.mean(s['actual_update_l2'] for s in servers[1:]),
            full_training_slots=sum(profile==6 for m in metrics for _,profile in m['slots']),
            repeated_physical_client_rounds=sum(len({cid for cid,_ in m['slots']})<10 for m in metrics),
            slots_with_changed_global_normalized_share=sum(s.get('normalized_share_changed_slots',0) for s in servers))
        original=[s.get('original_e03_cosine') for s in servers]
        if all(v is not None for v in original):
            summary['same_input_e03_negative_slots']=sum(c is not None and c<0 for v in original for c in v)
            summary['changed_risk_sign_slots']=sum((a is not None and a<0)!=(b is not None and b<0) for s,v in zip(servers,original) for a,b in zip(s['cosine'],v))
        protected=[(s,i) for s in servers for i in s.get('protected_slot_indices',[])]
        summary['capacity_protected_slots']=len(protected)
        summary['capacity_protected_original_negative_slots']=sum(s['original_e03_cosine'][i] is not None and s['original_e03_cosine'][i]<0 for s,i in protected)
        role_keys=servers[0].get('native_protected_keys',[])
        if role_keys:
            summary['native_protected_numel']=sum(full[k].numel() for k in role_keys)
            summary['native_protected_full_fraction']=summary['native_protected_numel']/total
            summary['max_actual_changed_full_fraction']=max(s['changed_from_e03_fraction'] for s in servers)
            if summary['max_actual_changed_full_fraction']>summary['native_protected_full_fraction']:
                raise ValueError('Changed parameter scope exceeds protected role')
        if name in ('L01','L02'):
            summary['later_synced_dispatch_rounds']=sum(s.get('dispatch_matches_full_anchor',False) for s in servers[1:])
            summary['later_nonzero_e03_difference_rounds']=sum(s['difference_from_e03_l2']>0 for s in servers[1:])
        if name=='L11':
            pairs=[(s['amplitude_risk'][i],max(0.,-c)) for s in servers for i,c in enumerate(s['cosine']) if c is not None and c<0]
            summary['mean_amplitude_to_cosine_risk_ratio']=statistics.mean(a/b for a,b in pairs)
        records[name]=summary
        table.append(dict(method=name,negative_direction_slots=summary['conflict_slots'],
            negative_percent=100*summary['conflict_slots']/4000,nonuniform_weight_rounds=summary['rounds_with_nonuniform_weights'],
            mean_weight=summary['weight_mean'],min_weight=summary['weight_min'],actual_same_input_e03_changed_rounds=summary['actual_e03_changed_rounds'],
            later_difference_from_same_input_e03_l2=summary['later_same_input_e03_difference_l2_mean'],
            later_l2_ratio_same_input_e03=summary['later_l2_ratio_e03_mean'],later_update_cosine_same_input_e03=summary['later_update_cosine_e03_mean'],
            perpendicular_change_energy_fraction=summary['later_change_decomposition']['perpendicular_fraction']))
        resource=json.loads((p/'resources.json').read_text())
        costs.append(dict(method=name,wall_seconds=resource['wall_seconds'],peak_allocated_mib=resource['peak_allocated_bytes']/2**20,
            peak_reserved_mib=resource['peak_reserved_bytes']/2**20,sum_server_seconds=sum(float(r['server_seconds']) for r in rows),
            sum_evaluation_seconds=sum(float(r['evaluation_seconds']) for r in rows)))
    if any(s!=schedules[0] for s in schedules) or any(v!=initial[0] for v in initial):raise ValueError('Local pairing differs')
    records['_verification']={'all_twelve_initializations_and_400_schedules_match':True,'full_model_numel':total,
        'L02_all_rounds_exact_same_input_e03_aggregation':records['L02']['actual_e03_changed_rounds']==0,
        'note':'L02 and REF are the same mathematical algorithm. Different observed trajectories are not new-method gains. One neutral repeat does not estimate a variance distribution.'}
    (output/'extended_mechanisms.json').write_text(json.dumps(records,indent=2))
    for filename,data in [('mechanisms.csv',table),('costs.csv',costs)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=[('Fusion and numeric control',['REF','L01','L02']),('Parameter-role protection',['REF','L03','L04','L05']),
        ('Capacity protection',['REF','L06','L07']),('Direction evidence',['REF','L08','L09','L10','L11'])]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(title,names) in zip(axes.flat,panels):
        for i,name in enumerate(names):ax.plot(range(201,401),curves[name][200:],label=name,color='black' if name=='REF' else f'C{i-1}',
            linestyle='--' if name=='REF' else '-',linewidth=1.5 if name=='REF' else 1.)
        ax.set(title=title,xlabel='Communication round',ylabel='Full test Accuracy (%)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(output/'accuracy_late_fixed_contrasts.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();analyze(a.root,a.output)
