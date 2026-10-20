"""Offline scalar evidence checks and interpretation inputs for the five runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyze_tonight_results import decomposition
from tools.report_l04_control import factorial_contrasts


def analyze(root,output):
    evidence=json.loads((root/'checkpoint_verification.json').read_text())
    expected=json.loads((root/'report/summary.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    results=[];mechanisms={};costs=[];initial=None;schedule=None;identity=None
    first_scores=[]
    for name in ('B00-D','R0','R1','R2','R3'):
        p=root/'raw'/name
        m=json.loads((p/'manifest.json').read_text());done=json.loads((p/'completed.json').read_text())
        with (p/'accuracy_each_round.csv').open() as f:rounds=list(csv.DictReader(f))
        with (p/'method_diagnostics.csv').open() as f:logs=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        if not done['success'] or done['rounds_completed']!=400 or len(rounds)!=400 or len(logs)!=400:
            raise ValueError('Incomplete '+name)
        if [int(r['round']) for r in rounds]!=list(range(1,401)):raise ValueError('Round sequence differs')
        own_schedule=[x['slots'] for x in logs]
        own_identity={k:v for k,v in m['identity'].items() if k not in ('method','options','source_sha256')}
        if initial is None:initial=m['initial_state_sha256'];schedule=own_schedule;identity=own_identity
        elif initial!=m['initial_state_sha256'] or schedule!=own_schedule or identity!=own_identity:
            raise ValueError('Five-run pairing differs')
        values=[float(r['full_accuracy']) for r in rounds];ce=[float(r['full_ce']) for r in rounds]
        peak=max(range(400),key=lambda i:values[i])
        row=dict(id=name,test_oracle_peak=values[peak],best_round=peak+1,final=values[-1],
            last20=statistics.mean(values[-20:]),last50=statistics.mean(values[-50:]),final_ce=ce[-1],peak_ce=ce[peak])
        original=next(r for r in expected['comparison'] if r['id']==name)
        if any(row[k]!=original[k] for k in row):raise ValueError('Summary differs from original CSV')
        check=evidence['runs'][name]
        if check['validated_completed_rounds']!=400 or not check['latest_parameters_finite'] or check['best_round']!=peak+1:
            raise ValueError('Remote checkpoint verification differs')
        results.append(row)
        servers=[x['server'] for x in logs];later=servers[1:]
        active=[i+1 for i,s in enumerate(servers) if s.get('classifier_protection_active')]
        has_scores=all('conflict_weights' in s for s in servers)
        def avg(key,data=later):
            v=[s[key] for s in data if s.get(key) is not None]
            return statistics.mean(v) if v else None
        if has_scores:first_scores.append((servers[0]['cosine'],servers[0]['conflict_weights']))
        summary=dict(classifier_active_rounds=len(active),first_classifier_active_round=active[0] if active else None,
            last_classifier_active_round=active[-1] if active else None,
            actual_changed_from_same_input_m2_rounds=sum(s['difference_from_e03_l2']>0 for s in servers) if has_scores else None,
            first_difference_from_same_input_m2_l2=servers[0].get('difference_from_e03_l2'),
            first_actual_update_l2=servers[0].get('actual_update_l2'),
            later_difference_from_same_input_m2_l2=avg('difference_from_e03_l2'),
            later_actual_update_l2=avg('actual_update_l2'),later_m2_update_l2=avg('e03_update_l2'),
            later_l2_ratio_m2=avg('update_l2_ratio_e03'),
            changed_from_same_input_m2_fraction_mean=avg('changed_from_e03_fraction',servers),
            changed_from_same_input_m2_fraction_max=max(s['changed_from_e03_fraction'] for s in servers) if has_scores else None,
            negative_slots=sum(s.get('conflict_slot_count',0) for s in servers) if has_scores else None,
            nonuniform_weight_rounds=sum(len(set(s['conflict_weights']))>1 for s in servers) if has_scores else None,
            mean_conflict_weight=statistics.mean(w for s in servers for w in s['conflict_weights']) if has_scores else None,
            min_conflict_weight=min(w for s in servers for w in s['conflict_weights']) if has_scores else None,
            later_change_decomposition=decomposition(later) if has_scores else None,
            full_training_slots=sum(p==6 for x in logs for _,p in x['slots']),
            rounds_with_duplicate_clients=sum(len({c for c,_ in x['slots']})<10 for x in logs))
        protected={k for s in servers for k in s.get('native_protected_keys',[])}
        if any(not k.startswith('classifier.') for k in protected):raise ValueError('Protected a non-classifier key')
        if has_scores and summary['changed_from_same_input_m2_fraction_max']>check['classifier_numel']/check['full_numel']:
            raise ValueError('Intervention exceeds classifier scope')
        summary['protected_keys']=sorted(protected)
        if name=='R3' and (active!=[1] or any(s['difference_from_e03_l2']!=0 for s in later)):
            raise ValueError('R3 differs from original M2 after round1')
        mechanisms[name]=summary
        resource=json.loads((p/'resources.json').read_text())
        costs.append(dict(id=name,wall_seconds=resource['wall_seconds'],
            peak_allocated_mib=resource['peak_allocated_bytes']/2**20,peak_reserved_mib=resource['peak_reserved_bytes']/2**20,
            sum_server_seconds=sum(float(r['server_seconds']) for r in rounds),
            sum_evaluation_seconds=sum(float(r['evaluation_seconds']) for r in rounds)))
    if any(v!=first_scores[0] for v in first_scores):raise ValueError('Initial M2 evidence differs')
    baseline=results[0];r0=next(r for r in results if r['id']=='R0')
    for row in results:
        for key in ('test_oracle_peak','final','last20','last50','final_ce'):
            row[key+'_minus_B00_D']=row[key]-baseline[key];row[key+'_minus_R0']=row[key]-r0[key]
    result=dict(comparison=results,mechanisms=mechanisms,costs=costs,factorial_contrasts=factorial_contrasts(results),
        verification=dict(all_five_400_rounds=True,all_initializations_schedules_execution_settings_match=True,
            initial_m2_cosines_and_weights_match=True,remote_latest_best_validated=True,R3_later_399_rounds_same_input_exact_M2=True),
        note='Same-input M2 differences are diagnostics on each own trajectory. Native baseline has no direction diagnostic; missing values remain null.')
    (output/'analysis.json').write_text(json.dumps(result,indent=2))
    for filename,rows in [('comparison.csv',results),('costs.csv',costs)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
