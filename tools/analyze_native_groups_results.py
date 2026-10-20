"""Read-only analysis of completed fixed G01-G10 scalar logs and geometry."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_next10_m2_m3 import read_rounds,describe


def mean(values):
    values=list(values)
    return statistics.mean(values) if values else None


def geometry_change(servers):
    parallel=perpendicular=0.
    for s in servers:
        a=s['actual_update_l2'];b=s['native_update_l2'];d=s['difference_from_native_l2']
        if not b:continue
        dot=(a*a+b*b-d*d)/2
        parallel+=(dot/(b*b)-1)**2*b*b
        perpendicular+=max(0.,a*a-dot*dot/(b*b))
    total=parallel+perpendicular
    return dict(parallel_difference_energy=parallel,perpendicular_difference_energy=perpendicular,
        perpendicular_fraction=perpendicular/total if total else None,
        note='Approximate reconstruction from logged float32 displacement norms; difference energy, not total update energy')


def analyze(root,output):
    status=json.loads((root/'status.json').read_text())
    if status['state']!='completed' or status['failed'] or len(status['completed'])!=10:raise ValueError('Ten completed runs required')
    remote=json.loads((root/'checkpoint_verification.json').read_text())
    original=json.loads((root/'report/summary.json').read_text())
    baseline_path=root/'baseline/accuracy_each_round.csv'
    baseline,baseline_curve=describe('B00-D','baseline',baseline_path,2)
    base_manifest=json.loads((root/'baseline/manifest.json').read_text())
    with (root/'baseline/method_diagnostics.csv').open() as f:base_schedule=[json.loads(r['metrics_json'])['slots'] for r in csv.DictReader(f)]
    identity={k:v for k,v in base_manifest['identity'].items() if k not in ('method','options','source_sha256')}
    partition=json.loads((ROOT/'data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json').read_text())['train_data']
    data_mass=[sum(len(partition[str(cid)]) for cid in range(a,b)) for a,b in [(0,40),(40,70),(70,100)]]
    curves={'B00-D':baseline_curve};rows=[];summaries={};region_rows=[];weight_rows=[];costs=[];all_coverage=None
    output.mkdir(parents=True,exist_ok=False)
    for number in range(1,11):
        name=f'G{number:02d}';p=root/'raw'/name
        done=json.loads((p/'completed.json').read_text());manifest=json.loads((p/'manifest.json').read_text())
        item,curve=describe(name,name.lower(),p/'accuracy_each_round.csv',2);rounds,_,ce=read_rounds(p/'accuracy_each_round.csv')
        if not done['success'] or done['rounds_completed']!=400:raise ValueError('Incomplete '+name)
        for key,value in item.items():
            if value!=next(r for r in original['comparison'] if r['id']==name)[key]:raise ValueError('Original summary differs')
        check=remote['runs'][name]
        if (check['rounds']!=400 or check['rows']!=400 or check['best_round']!=item['best_round']
            or check['best_accuracy']!=item['test_oracle_peak']
            or check['latest_accuracy']!=item['final_accuracy'] or check['latest_ce']!=item['final_ce']
            or not check['finite_full_parameters'] or not check['source_matches_other_nine']):
            raise ValueError('Checkpoint verification differs')
        own={k:v for k,v in manifest['identity'].items() if k not in ('method','options','source_sha256')}
        if own!=identity or manifest['initial_state_sha256']!=base_manifest['initial_state_sha256'] or manifest['torch']!=base_manifest['torch']:
            raise ValueError('Unpaired training settings')
        with (p/'method_diagnostics.csv').open() as f:logs=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        if len(logs)!=400 or [r['slots'] for r in logs]!=base_schedule:raise ValueError('Slot schedule differs')
        servers=[m['server'] for m in logs];regions=manifest['structural_regions']
        if len(regions)!=9 or servers[0]['difference_from_native_l2']!=0:raise ValueError('Geometry or warmup differs')
        with baseline_path.open() as f:first=next(csv.DictReader(f))
        if curve[0]!=float(first['full_accuracy']) or ce[0]!=float(first['full_ce']):raise ValueError('First Native evaluation differs')
        shared=next(r['id'] for r in regions if len(r['profiles'])==7)
        full_only=next(r['id'] for r in regions if r['profiles']==[6])
        coverage=[0]*9
        for record,s in zip(logs,servers):
            for r,detail in zip(regions,s['regions']):
                members=[i for i,(_,profile) in enumerate(record['slots']) if profile in r['profiles']]
                physical={record['slots'][i][0] for i in members}
                if members!=detail['slots'] or len(physical)!=detail['distinct_physical_clients']:raise ValueError('Coverage bookkeeping differs')
                coverage[r['id']]+=len(physical)
                if not members and detail['actual_changed_coordinates']!=0:raise ValueError('Uncovered region changed')
                if name=='G04' and r['id']!=shared and detail['actual_changed_coordinates']!=0:raise ValueError('G04 outside common scope')
                if name=='G05' and r['id']==shared and detail['actual_changed_coordinates']!=0:raise ValueError('G05 changed common scope')
            if name in ('G07','G08') and coverage!=s['cumulative_physical_coverage']:raise ValueError('Cumulative physical coverage differs')
        if all_coverage is None:all_coverage=list(coverage)
        elif coverage!=all_coverage:raise ValueError('Different coverage opportunities')
        if name in ('G01','G03','G04','G05') and any(s['regions'][full_only]['actual_changed_coordinates'] for s in servers):
            raise ValueError('Single-source Full-only region should be Native')
        group_native=[];group_applied=[];profile_native=[];profile_applied=[]
        for log,s in zip(logs[1:],servers[1:]):
            d=s['regions'][shared];native=[0.]*3;applied=[0.]*3;pn=[0.]*7;pa=[0.]*7
            for j,slot in enumerate(d['slots']):
                cid,profile=log['slots'][slot];tier=int(cid>=40)+int(cid>=70)
                native[tier]+=d['native_slot_shares'][j];applied[tier]+=d['applied_slot_shares'][j]
                pn[profile]+=d['native_slot_shares'][j];pa[profile]+=d['applied_slot_shares'][j]
            group_native.append(native);group_applied.append(applied);profile_native.append(pn);profile_applied.append(pa)
        tier_native=[mean(row[i] for row in group_native) for i in range(3)]
        tier_applied=[mean(row[i] for row in group_applied) for i in range(3)]
        scale_error=[s['scale_norm_relative_error'] for s in servers if s['scale_norm_relative_error'] is not None]
        worst_scale=max((s for s in servers if s['scale_norm_relative_error'] is not None),
            key=lambda s:s['scale_norm_relative_error'],default=None)
        summary=dict(changed_rounds=sum(s['difference_from_native_l2']>0 for s in servers),
            later_native_difference_l2_mean=mean(s['difference_from_native_l2'] for s in servers[1:]),
            later_update_l2_ratio_native_mean=mean(s['actual_update_l2_ratio_native'] for s in servers[1:] if s['actual_update_l2_ratio_native'] is not None),
            later_global_scale_mean=mean(s['global_scale'] for s in servers[1:]),
            max_scale_norm_relative_error=max(scale_error) if scale_error else None,
            worst_scale_match={k:worst_scale[k] for k in ('round','actual_update_l2','virtual_regional_update_l2',
                'scale_norm_absolute_error','scale_norm_relative_error')} if worst_scale else None,
            later_geometry=geometry_change(servers[1:]),coverage_unique_physical_totals=coverage,
            full_training_slots=sum(profile==6 for m in logs for _,profile in m['slots']),
            duplicate_client_rounds=sum(len({c for c,_ in m['slots']})<10 for m in logs),
            shared_tier_native_shares=tier_native,shared_tier_applied_shares=tier_applied,
            shared_profile_native_shares=[mean(row[i] for row in profile_native) for i in range(7)],
            shared_profile_applied_shares=[mean(row[i] for row in profile_applied) for i in range(7)],
            full_only_directly_changed_rounds=sum(s['regions'][full_only]['actual_changed_coordinates']>0 for s in servers))
        for r in regions:
            items=[s['regions'][r['id']] for s in servers[1:]]
            covered=[d for d in items if d['slots']]
            estimated=[d for d in items if 'estimated_mean_noise_energy' in d]
            row=dict(id=name,region_id=r['id'],support_profiles=','.join(map(str,r['profiles'])),numel=r['numel'],
                total_unique_physical_coverages=coverage[r['id']],covered_later_rounds=len(covered),
                actual_changed_later_rounds=sum(d['actual_changed_coordinates']>0 for d in items),
                slot_share_l1_change_mean=mean(d['weight_share_l1_change'] for d in items),
                gain_is_virtual=name in ('G08','G10'),gain_mean=mean(d['gain'] for d in covered),
                lower_compensation_clip_rounds=sum(d['gain']==.8 for d in covered),
                upper_compensation_clip_rounds=sum(d['gain']==1.2 for d in covered),
                zero_gain_rounds=sum(d['gain']==0 for d in covered),
                single_physical_neutral_rounds=sum(d['reason']=='one_physical_client' for d in items),
                mean_noise_to_observed_mean_energy=mean(d['estimated_mean_noise_energy']/d['mean_energy'] for d in estimated if d['mean_energy']>0))
            region_rows.append(row)
        if name in ('G09','G10'):
            zero_fractions=[];weighted_gains=[]
            for s in servers[1:]:
                covered=[d for d in s['regions'] if d['slots']];mass=sum(d['numel'] for d in covered)
                zero_fractions.append(sum(d['numel'] for d in covered if d['gain']==0)/mass)
                weighted_gains.append(sum(d['numel']*d['gain'] for d in covered)/mass)
            summary['mean_zero_gain_fraction_covered_coordinates']=mean(zero_fractions)
            summary['mean_parameter_weighted_regional_gain']=mean(weighted_gains)
        summaries[name]=summary
        weight_rows.append(dict(id=name,**{f'{tier}_native_share':tier_native[i] for i,tier in enumerate(('low','mid','high'))},
            **{f'{tier}_applied_share':tier_applied[i] for i,tier in enumerate(('low','mid','high'))}))
        for key in ('test_oracle_peak','final_accuracy','last20_mean','last50_mean','final_ce'):item[key+'_minus_B00_D']=item[key]-baseline[key]
        rows.append(item);curves[name]=curve
        resource=json.loads((p/'resources.json').read_text())
        costs.append(dict(id=name,wall_minutes=resource['wall_seconds']/60,
            server_seconds=sum(float(r['server_seconds']) for r in rounds),
            peak_allocated_mib=resource['peak_allocated_bytes']/2**20,peak_reserved_mib=resource['peak_reserved_bytes']/2**20))
    model_numel=[sum(r['numel'] for r in regions if profile in r['profiles']) for profile in range(7)]
    uplink=sum(model_numel[p]*4 for slots in base_schedule for _,p in slots)
    contrasts={}
    by_id={r['id']:r for r in rows}
    for a,b in [('G03','G06'),('G05','G01'),('G05','G04'),('G07','G08'),('G09','G10')]:
        contrasts[a+'-'+b]={key:by_id[a][key]-by_id[b][key] for key in ('test_oracle_peak','final_accuracy','last20_mean','last50_mean','final_ce')}
    result=dict(baseline=baseline,comparison=rows,mechanisms=summaries,contrasts=contrasts,
        data_mass_by_physical_tier=data_mass,data_mass_share_by_tier=[n/sum(data_mass) for n in data_mass],
        physical_coverage_totals=all_coverage,model_numel=model_numel,
        communication=dict(each_run_uplink_float32_state_bytes=uplink,each_run_downlink_float32_state_bytes=uplink,
            note='Identical actual slot/profile schedule, no added client payload; excludes transport/serialization headers'),
        verification=dict(all_ten_actual_400=True,checkpoint_finite_shape_and_peak_validated_on_server=True,
            all_baseline_init_protocol_schedule_and_first_eval_match=True,nine_real_coverage_regions=True,
            uncovered_regions_native=True,G04_G05_scopes_correct=True,G07_G08_counter_histories_correct=True),
        interpretation='G08/G10 regional gains are virtual, applied coefficient is the global scale. Difference norms compare each own trajectory with same-input Native, not with historical baseline models.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2))
    for filename,data in [('comparison.csv',rows),('regions.csv',region_rows),('shared_source_weights.csv',weight_rows),('costs.csv',costs)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    plots(output,result,curves)
    return result


def plots(output,result,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,names in zip(axes.flat,[['G03','G06'],['G01','G04','G05'],['G07','G08'],['G09','G10']]):
        ax.plot(range(301,401),curves['B00-D'][300:],label='B00-D',color='black',linestyle='--',linewidth=1.2)
        for name in names:ax.plot(range(301,401),curves[name][300:],label=name,linewidth=1)
        ax.set(xlabel='Round',ylabel='Full test Accuracy (%)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'late_fixed_contrasts.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));x=list(range(3));width=.25
    g=result['mechanisms']['G03']
    for offset,values,label in [(-width,result['data_mass_share_by_tier'],'All training data mass'),
        (0,g['shared_tier_native_shares'],'Native shared contribution'),(width,g['shared_tier_applied_shares'],'G03 shared contribution')]:
        ax.bar([i+offset for i in x],[v*100 for v in values],width=width,label=label)
    ax.set(xticks=x,xticklabels=['Low capability','Mid capability','High capability'],ylabel='Share (%)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(output/'g03_source_weights.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
