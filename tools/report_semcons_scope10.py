"""Post-training mechanism report; no algorithm decisions or official-test tuning."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch
try:
    from .report_next10_m2_m3 import report_next10
except ImportError:
    from report_next10_m2_m3 import report_next10


def mean(values):
    v=[float(x) for x in values if x is not None]
    return statistics.mean(v) if v else None


def rng_equal(a,b):
    return (a['python']==b['python'] and a['numpy'][0]==b['numpy'][0]
        and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
        and torch.equal(a['torch'],b['torch']) and len(a['cuda'])==len(b['cuda'])
        and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])))


def report_scope(batches,output,config):
    report_next10(batches,output,config)
    jobs={j['id']:j for b in batches for j in b};records=[];schedules={};initial={}
    for id,job in jobs.items():
        root=Path(job['output_dir'])
        with (root/'method_diagnostics.csv').open() as f:rows=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        servers=[r['server'] for r in rows];clients=[c for r in rows for c in r['clients']]
        schedules[id]=hashlib.sha256(json.dumps([r['slots'] for r in rows],separators=(',',':')).encode()).hexdigest()
        manifest=json.loads((root/'manifest.json').read_text());initial[id]=manifest['initial_state_sha256']
        if id=='S01':
            with (Path(output)/'S01_structural_coverage.jsonl').open('w') as f:
                for index,row in enumerate(rows,1):
                    regions=[]
                    for region in manifest['structural_regions']:
                        members=[i for i,(_,p) in enumerate(row['slots']) if p in region['profiles']]
                        regions.append({'region_id':region['id'],'numel':region['numel'],'slots':members,
                            'physical_clients':sorted({row['slots'][i][0] for i in members})})
                    f.write(json.dumps({'round':index,'regions':regions})+'\n')
        record={'id':id,'rounds':len(rows),
            'mean_direction_conflict_fraction':mean([s.get('direction_conflict_fraction',s.get('negative_fraction')) for s in servers]),
            'mean_semantic_valid_fraction':mean([s.get('semantic_valid_comparison_fraction') for s in servers]),
            'mean_semantic_missing_fraction':mean([s.get('semantic_missing_comparison_fraction') for s in servers]),
            'mean_gate_region_slot_fraction':mean([s.get('gate_trigger_region_slot_fraction') for s in servers]),
            'mean_gate_target_parameter_fraction':mean([s.get('gate_target_covered_parameter_fraction') for s in servers]),
            'mean_actual_changed_parameter_fraction':mean([s.get('actual_changed_parameter_fraction_full') for s in servers]),
            'mean_insufficient_evidence_region_fraction':mean([s.get('insufficient_evidence_region_fraction') for s in servers]),
            'mean_native_learning_update_l2':mean([s.get('comparable_native_learning_update_l2') for s in servers]),
            'mean_applied_learning_update_l2':mean([s.get('comparable_applied_learning_update_l2') for s in servers]),
            'measurement_seconds':sum(c.get('semantic_measurement_seconds',0.) for c in clients),
            'semantic_payload_bytes_proxy':sum(c.get('semantic_payload_bytes_proxy',0) for c in clients)}
        semantic=[r.get('semantic',{}) for server in servers for r in server.get('regions',[])]
        record['semantic_permuted_client_regions']=sum(s.get('permuted_client_count',0) for s in semantic)
        record['semantic_changed_score_client_regions']=sum(s.get('changed_score_count',0) for s in semantic)
        record['mean_neutral_or_fallback_region_fraction']=mean([s.get('neutral_or_fallback_region_fraction') for s in servers])
        records.append(record)
    paired={'all_schedules_equal':len(set(schedules.values()))==1,'all_initializations_equal':all(v==initial['S00'] for v in initial.values())}
    a=torch.load(Path(jobs['S00']['output_dir'])/'latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(Path(jobs['S02']['output_dir'])/'latest.pt',map_location='cpu',weights_only=False)
    paired.update(S00_S02_final_weights_bitwise_equal=all(torch.equal(a['global'][k],b['global'][k]) for k in a['global']),
                  S00_S02_training_rng_equal=rng_equal(a['rng'],b['rng']),
                  S00_S02_final_parameter_difference_l2=sum(float((a['global'][k].double()-b['global'][k].double()).square().sum()) for k in a['global'])**.5)
    result={'pairing':paired,'mechanisms':records,'causal_contrasts':[['S00','S02'],['S03','S06'],['S05','S06'],['S06','S07'],['S07','S08'],['S07','S09']]}
    (Path(output)/'mechanism_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Mechanism diagnostics','',json.dumps(paired,indent=2),'',
          'Inspect S00/S02 weight and RNG equality directly. Nonzero weight difference with equal RNG requires numerical/backend analysis; measurement neutrality is not inferred from matching Accuracy alone.','',
          'S07/S09 use the same evidence rule, Gate and ungated dispatch anchor; their algebraic difference at fixed inputs is only the denominator. Their trained trajectories can subsequently differ.','',
          'S08 preserves the per-region valid score multiset and validity metadata; a shuffled zero/constant score stays unchanged. Count actual changed scores before interpreting the negative control.','',
          'Detailed aggregate counts, timing and payload proxies are in mechanism_summary.json; per-region contributors, raw risks, Gate/confidence and update norms remain in each method_diagnostics.csv.']
    text += ['', '|ID|Direction conflict|Semantic valid|Gate region/slot|Gate target params|Actually changed params|Neutral regions|Measurement seconds|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def show(v):return 'N/A' if v is None else f'{v:.6f}'
    for r in records:
        fields=['mean_direction_conflict_fraction','mean_semantic_valid_fraction','mean_gate_region_slot_fraction',
                'mean_gate_target_parameter_fraction','mean_actual_changed_parameter_fraction','mean_neutral_or_fallback_region_fraction','measurement_seconds']
        text.append('|'+r['id']+'|'+ '|'.join(show(r[k]) for k in fields)+'|')
    (Path(output)/'mechanism_report.md').write_text('\n'.join(text)+'\n')
