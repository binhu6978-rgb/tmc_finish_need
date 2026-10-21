"""Final fixed comparisons and actual diagnostics, without model/data tuning."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_tonight10 import validate
from tools.report_next10_m2_m3 import describe


def validate_reference(path):
    """Historical comparisons use retained scalar results, not deleted weights.

    User-authorized checkpoint cleanup does not invalidate completed CSVs.
    Current experiments still use the strict checkpoint-aware validate().
    """
    done=json.loads((path/'completed.json').read_text())
    if not done['success'] or done['rounds_completed']!=400:
        raise ValueError('Historical reference did not complete 400 rounds: '+str(path))
    item,_=describe(path.name,path.parent.name,path/'accuracy_each_round.csv',2)
    return item


def validate_run(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('save_pt',True):return validate(path)
    item=validate_reference(path)
    done=json.loads((path/'completed.json').read_text())
    if done.get('checkpoint_policy')!='disabled_by_user' or done['checkpoint'] is not None:
        raise ValueError('Checkpoint-free completion policy mismatch')
    if list(path.glob('*.pt')) or list(path.glob('*.pending')):
        raise ValueError('Checkpoint-free run contains unexpected model/pending files')
    with (path/'method_diagnostics.csv').open() as f:diags=list(csv.DictReader(f))
    if [int(r['round']) for r in diags]!=list(range(1,401)):
        raise ValueError('Incomplete actual aggregation diagnostics')
    check=json.loads((path/'final_model_check.json').read_text())
    if check['round']!=400 or not check['finite_full_parameters'] or check['full_numel']!=sum(r['numel'] for r in manifest['structural_regions']):
        raise ValueError('Final in-memory Full validity check differs')
    summary=json.loads((path/'summary.json').read_text())
    if summary['rounds_completed']!=400 or summary['best_round']!=item['best_round'] or summary['best_full_accuracy']!=item['test_oracle_peak']:
        raise ValueError('Final scalar summary differs from 400-round logs')
    return item


def diagnostic_summary(path):
    with path.open() as f:logs=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
    servers=[m['server'] for m in logs]
    if len(servers)!=400:raise ValueError('Expected 400 diagnostic rows')
    regions={}
    for rid in range(len(servers[0]['regions'])):
        items=[s['regions'][rid] for s in servers]
        regions[rid]=dict(numel=items[0]['numel'],support_profiles=items[0]['support_profiles'],
            covered_rounds=sum(bool(r['slots']) for r in items),
            total_unique_physical_coverages=sum(r['distinct_physical_clients'] for r in items),
            mean_covered_distinct_profiles=statistics.mean(len(r['actual_profiles']) for r in items),
            mean_distinct_physical_clients=statistics.mean(r['distinct_physical_clients'] for r in items),
            gain_min=min(r['gain'] for r in items),gain_mean=statistics.mean(r['gain'] for r in items),
            gain_max=max(r['gain'] for r in items),
            mean_weight_share_l1_change=statistics.mean(r['weight_share_l1_change'] for r in items),
            actually_changed_rounds=sum(r['actual_changed_coordinates']>0 for r in items),
            single_client_neutral_rounds=sum(r['reason']=='one_physical_client' for r in items),
            estimated_noise_energy_mean=statistics.mean(r['estimated_mean_noise_energy'] for r in items if 'estimated_mean_noise_energy' in r) if any('estimated_mean_noise_energy' in r for r in items) else None)
    error=[s['scale_norm_relative_error'] for s in servers if s['scale_norm_relative_error'] is not None]
    ratios=[s['actual_update_l2_ratio_native'] for s in servers[1:] if s['actual_update_l2_ratio_native'] is not None]
    if servers[0]['difference_from_native_l2']!=0:raise ValueError('First-round Native control differs')
    summary=dict(first_round=servers[0],regions=regions,
        changed_rounds=sum(s['difference_from_native_l2']>0 for s in servers),
        later_difference_from_native_l2_mean=statistics.mean(s['difference_from_native_l2'] for s in servers[1:]),
        later_update_l2_ratio_native_mean=statistics.mean(ratios) if ratios else None,
        scale_norm_relative_error_max=max(error) if error else None,
        cumulative_final=servers[-1]['cumulative_physical_coverage'])
    if 'difference_from_g03_l2' in servers[-1]:
        for key in ('actual_update_cosine_native','actual_update_cosine_g03','difference_from_g03_l2','global_scale'):
            values=[s[key] for s in servers[1:] if s[key] is not None]
            summary['later_'+key+'_mean']=statistics.mean(values) if values else None
        values=[s['matched_norm_relative_error'] for s in servers if s['matched_norm_relative_error'] is not None]
        summary['matched_norm_relative_error_max']=max(values) if values else None
        summary['source_shares']={}
        for rid in regions:
            shares={}
            for key in ('native_slot_shares','applied_slot_shares','virtual_g03_slot_shares'):
                triples=[]
                for log,s in zip(logs[1:],servers[1:]):
                    detail=s['regions'][rid]
                    if key not in detail:continue
                    tier=[0.,0.,0.]
                    for i,w in zip(detail['slots'],detail[key]):
                        cid=log['slots'][i][0];tier[int(cid>=40)+int(cid>=70)]+=w
                    triples.append(tier)
                shares[key]=[statistics.mean(x[t] for x in triples) for t in range(3)] if triples else None
            summary['source_shares'][rid]=shares
        summary['source_share_scope']='P10 applied shares refer to nonclassifier cells; classifier is Native. P01 applied shares are Native, virtual G03 shares are counterfactual. Global scales act on anchored increments.'
    return summary,[m['slots'] for m in logs]


def report(root,config_path=None):
    config=json.loads((config_path or ROOT/'configs/native_groups10_400.json').read_text())
    baseline_path=ROOT/config['baseline_csv'];base_manifest=json.loads((ROOT/config['baseline_manifest']).read_text())
    reference,reference_curve=describe('B00-D','baseline',baseline_path,2)
    with (baseline_path.parent/'method_diagnostics.csv').open() as f:base_schedule=[json.loads(r['metrics_json'])['slots'] for r in csv.DictReader(f)]
    with baseline_path.open() as f:base_ce=[float(r['full_ce']) for r in csv.DictReader(f)]
    curves={'B00-D':reference_curve};ce_curves={'B00-D':base_ce};records=[];mechanisms={};costs=[]
    common_identity={k:v for k,v in base_manifest['identity'].items() if k not in ('method','options','source_sha256')}
    competitor=None
    if 'reference_csv' in config:
        p=ROOT/config['reference_csv'];validate_reference(p.parent)
        competitor,curve=describe(config['reference_id'],config['reference_id'].lower(),p,2)
        curves[competitor['id']]=curve
        with p.open() as f:ce_curves[competitor['id']]=[float(r['full_ce']) for r in csv.DictReader(f)]
        m=json.loads((p.parent/'manifest.json').read_text())
        with (p.parent/'method_diagnostics.csv').open() as f:schedule=[json.loads(r['metrics_json'])['slots'] for r in csv.DictReader(f)]
        if ({k:v for k,v in m['identity'].items() if k not in ('method','options','source_sha256')}!=common_identity
            or m['initial_state_sha256']!=base_manifest['initial_state_sha256'] or schedule!=base_schedule):
            raise ValueError('Existing G03 reference protocol or initial schedule differs')
    for name in sum(config['batches'],[]):
        p=root/name.lower()/(name+'_full_only_400_seed1_deterministic');validate_run(p)
        manifest=json.loads((p/'manifest.json').read_text())
        identity={k:v for k,v in manifest['identity'].items() if k not in ('method','options','source_sha256')}
        if identity!=common_identity or manifest['initial_state_sha256']!=base_manifest['initial_state_sha256'] or manifest['torch']!=base_manifest['torch']:
            raise ValueError('Training, numerical execution or initialization differs from B00-D')
        item,curve=describe(name,name.lower(),p/'accuracy_each_round.csv',2)
        for key in ('test_oracle_peak','final_accuracy','last20_mean','last50_mean','final_ce'):
            item[key+'_minus_B00_D']=item[key]-reference[key]
            if competitor:item[key+'_minus_'+competitor['id']]=item[key]-competitor[key]
        records.append(item);curves[name]=curve
        with (p/'accuracy_each_round.csv').open() as f:rounds=list(csv.DictReader(f))
        ce_curves[name]=[float(r['full_ce']) for r in rounds]
        mechanisms[name],schedule=diagnostic_summary(p/'method_diagnostics.csv')
        if schedule!=base_schedule:raise ValueError('400-round actual slots differ from B00-D')
        if curve[0]!=reference_curve[0] or ce_curves[name][0]!=base_ce[0]:raise ValueError('First Native evaluation differs from B00-D')
        resource=json.loads((p/'resources.json').read_text())
        costs.append(dict(id=name,wall_seconds=resource['wall_seconds'],peak_allocated_mib=resource['peak_allocated_bytes']/2**20,
            peak_reserved_mib=resource['peak_reserved_bytes']/2**20,
            server_seconds=sum(float(r['server_seconds']) for r in rounds),evaluation_seconds=sum(float(r['evaluation_seconds']) for r in rounds)))
    output=root/'report';output.mkdir(exist_ok=False)
    for filename,rows in [('comparison.csv',records),('costs.csv',costs)]:
        with (output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(comparison=records,baseline=reference,competitor=competitor,mechanisms=mechanisms,
        pairing='all completed initial models, original partition,400 actual schedules and strict execution match B00-D; first Native evaluation identical',
        scale_control='same-input virtual norm per own trajectory; G07 vs G08 and G09 vs G10 do not share subsequent trained models',
        peak_label='official test-oracle peak, single seed exploration'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for values,label,filename in [(curves,'Full test Accuracy (%)','accuracy'),(ce_curves,'Full test CE','ce')]:
        with (output/(filename+'_all_rounds.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['round',*values]);w.writerows([i+1,*[v[i] for v in values.values()]] for i in range(400))
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        for ax,batch in zip(axes.flat,config['batches']):
            ax.plot(range(1,401),values['B00-D'],label='B00-D',color='black',linewidth=1,linestyle='--')
            if competitor:ax.plot(range(1,401),values[competitor['id']],label=competitor['id'],color='purple',linewidth=1.2,linestyle='--')
            for name in batch:ax.plot(range(1,401),values[name],label=name,linewidth=1)
            ax.set(xlabel='Round',ylabel=label);ax.grid(alpha=.2);ax.legend()
        axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(output/(filename+'.png'),dpi=150);plt.close(fig)
    lines=['# Native coverage/source experiments','', '| ID | test-oracle peak | round | Final | Last20 | Last50 | CE |','|---|---:|---:|---:|---:|---:|---:|']
    for r in [reference,*([competitor] if competitor else []),*records]:lines.append(f"|{r['id']}|{r['test_oracle_peak']:.4f}|{r['best_round']}|{r['final_accuracy']:.4f}|{r['last20_mean']:.4f}|{r['last50_mean']:.4f}|{r['final_ce']:.6f}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
