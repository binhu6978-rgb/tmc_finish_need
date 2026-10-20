"""Read-only scalar-log analysis of the completed frozen server suite."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_tonight10 import diagnostic_summary


def decomposition(servers):
    parallel=perpendicular=0.
    for s in servers:
        if s.get('effective_step_along_e03') is None:continue
        b=s['e03_update_l2'];a=s['actual_update_l2'];step=s['effective_step_along_e03']
        parallel+=(step-1)**2*b*b
        perpendicular+=max(0.,a*a-step*step*b*b)
    total=parallel+perpendicular
    return dict(parallel_difference_energy=parallel,perpendicular_difference_energy=perpendicular,
        perpendicular_fraction=perpendicular/total if total>1e-20 else None,
        note='Reconstructed from logged actual/base update norms and inner products; excludes initialization when caller passes rounds2-400')


def analyze(root,output):
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='finished' or status['failed'] or len(status['completed'])!=11:
        raise ValueError('Expected completed C2 plus ten fixed server runs')
    output.mkdir(exist_ok=False,parents=True)
    all_data={};table=[];costs=[];manifests=[];schedules=[]
    for method in ['c1',*status['order']]:
        name='C2_full_only_400_seed1_after_update' if method=='c2' else method.upper()+'_full_only_400_seed1'
        p=root/method/name
        done=json.loads((p/'completed.json').read_text())
        if not done['success'] or done['rounds_completed']!=400:raise ValueError('Incomplete run')
        with (p/'accuracy_each_round.csv').open() as f:rounds=list(csv.DictReader(f))
        with (p/'method_diagnostics.csv').open() as f:metrics=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        if len(rounds)!=400 or len(metrics)!=400:raise ValueError('Missing rounds')
        servers=[m['server'] for m in metrics]
        summary,schedule=diagnostic_summary(p/'method_diagnostics.csv');schedules.append(schedule)
        manifest=json.loads((p/'manifest.json').read_text());manifests.append(manifest)
        summary['later_change_decomposition']=decomposition(servers[1:])
        summary['rounds_with_actual_e03_change']=sum(s['difference_from_e03_l2']>0 for s in servers)
        summary['duplicate_physical_slot_rounds']=sum(len({slot[0] for slot in m['slots']})<10 for m in metrics)
        original=[s.get('original_e03_cosine') for s in servers]
        if all(v is not None for v in original):
            summary['same_input_original_e03_negative_slots']=sum(c is not None and c<0 for v in original for c in v)
            summary['direction_risk_sign_changed_slots']=sum((a is not None and a<0)!=(b is not None and b<0)
                for s,v in zip(servers,original) for a,b in zip(s['cosine'],v))
        projections=[c for s in servers for c in s.get('projection_slots',[]) if c['negative_projection_l2']>0]
        if projections:
            summary['c2_negative_slot_remaining_energy_mean']=statistics.mean(c['remaining_component_l2']**2/c['delta_l2']**2 for c in projections)
            summary['c2_remaining_components_m2_would_downweight']=sum(c['e03_normalized_fraction']<c['native_sample_fraction'] for c in projections)
        protected=[c for s in servers for c in s.get('protected_components',[])]
        if protected:
            summary['t09_remaining_energy_mean']=statistics.mean(c['remaining_l2']**2/(c['remaining_l2']**2+c['negative_projection_l2']**2) for c in protected)
            summary['t09_restored_coefficient_mean']=statistics.mean(c['restored_coefficient'] for c in protected)
            summary['t09_max_restoration_bound_excess']=max(s.get('restoration_actual_l2',0)-s.get('restoration_triangle_bound_l2',0) for s in servers)
        summary['effective_scale_rounds']=sum(bool(s.get('scale_applied')) for s in servers)
        all_data[method.upper()]=summary
        table.append(dict(method=method.upper(),conflict_slots=summary['conflict_slots'],
            conflict_slot_percent=100*summary['conflict_slots']/4000,conflict_rounds=summary['conflict_rounds'],
            nonuniform_weight_rounds=summary['rounds_with_nonuniform_weights'],
            mean_weight=summary['weight_mean'],min_weight=summary['weight_min'],
            later_difference_from_same_input_e03_l2=summary['later_same_input_e03_difference_l2_mean'],
            later_l2_ratio_same_input_e03=summary['later_l2_ratio_e03_mean'],
            later_update_cosine_same_input_e03=summary['later_update_cosine_e03_mean'],
            perpendicular_change_energy_fraction=summary['later_change_decomposition']['perpendicular_fraction'],
            t09_restored_slots=summary['protected_components'],c2_adjusted_projection_slots=summary['negative_projection_adjusted_slots']))
        resources=json.loads((p/'resources.json').read_text())
        costs.append(dict(method=method.upper(),wall_seconds=resources['wall_seconds'],
            peak_allocated_mib=resources['peak_allocated_bytes']/2**20,peak_reserved_mib=resources['peak_reserved_bytes']/2**20,
            sum_round_wall_seconds=sum(float(r['round_wall_seconds']) for r in rounds),
            sum_server_seconds=sum(float(r['server_seconds']) for r in rounds),
            sum_evaluation_seconds=sum(float(r['evaluation_seconds']) for r in rounds)))
    if any(s!=schedules[0] for s in schedules[1:]) or any(m['initial_state_sha256']!=manifests[0]['initial_state_sha256'] for m in manifests[1:]):
        raise ValueError('C1/C2/T initialization or client-profile schedules differ')
    old=json.loads((root/'historical/E03_manifest.json').read_text())
    all_data['_verification']=dict(actual_completed_new_runs=11,C1_completed_rounds=400,
        all_twelve_initializations_and_400_schedules_match=True,
        historical_E03_initialization_match=old['initial_state_sha256']==manifests[0]['initial_state_sha256'],
        original_aggregation_source_match=old['identity']['source_sha256']['models/Fed.py']==manifests[0]['identity']['source_sha256']['models/Fed.py'],
        note='Checkpoint400 and first best-peak alignment were validated on server; only scalar files were downloaded. Same-input E03 diagnostics are not historical trajectory comparisons.')
    (output/'extended_mechanisms.json').write_text(json.dumps(all_data,indent=2))
    for name,data in [('mechanisms.csv',table),('costs.csv',costs)]:
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (root/'report/accuracy_all_rounds.csv').open() as f:curves=list(csv.DictReader(f))
    panels=[('Initialization control',['E03','T04','B00','N08']),
        ('Fixed scale controls',['E03','T01','T02','T03','T10']),
        ('Direction evidence',['E03','T05','T06','T07','T08']),
        ('Scope and component protection',['E03','C1','C2','T09'])]
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,(title,names) in zip(axes.flat,panels):
        for i,name in enumerate(names):
            ax.plot(range(201,401),[float(r[name]) for r in curves[200:]],label=name,
                color='black' if name=='E03' else f'C{i-1}',linestyle='--' if name=='E03' else '-',linewidth=1.4 if name=='E03' else 1.)
        ax.set(title=title,xlabel='Communication round',ylabel='Full test Accuracy (%)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(output/'accuracy_late_fixed_contrasts.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();analyze(a.root,a.output)
