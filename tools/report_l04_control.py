"""Offline R0/R1/R2 summary; no training or checkpoint selection by validation."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_tonight10 import validate


def factorial_contrasts(rows):
    values={r['id']:r for r in rows}
    return {key:{
        'initial_native_without_later':values['R3'][key]-values['R0'][key],
        'initial_native_with_later':values['R1'][key]-values['R2'][key],
        'later_native_without_initial':values['R2'][key]-values['R0'][key],
        'later_native_with_initial':values['R1'][key]-values['R3'][key],
        'interaction':values['R1'][key]-values['R2'][key]-values['R3'][key]+values['R0'][key],
        'm2_minus_native':values['R0'][key]-values['B00-D'][key],
        } for key in ('test_oracle_peak','final','last20','last50','final_ce')}


def report(root,extra_root=None):
    rows=[];curves={};ce_curves={};diagnostics={};initial=None;schedule=None
    shared_identity=None;shared_torch=None;full_logs={}
    plans=[(root,'R0','m2'),(root,'R1','l04'),(root,'R2','l04')]
    if extra_root is not None:plans.extend([(extra_root,'B00-D','baseline'),(extra_root,'R3','l04')])
    for base,name,method in plans:
        p=base/method/(name+'_full_only_400_seed1_deterministic')
        validate(p)
        m=json.loads((p/'manifest.json').read_text())
        policy=m['identity']['execution_policy']
        if not policy['deterministic_algorithms'] or policy['warn_only']:raise ValueError('Strict execution missing')
        identity={k:v for k,v in m['identity'].items() if k not in ('source_sha256','method','options')}
        if shared_identity is None:shared_identity=identity;shared_torch=m.get('torch')
        elif identity!=shared_identity or m.get('torch')!=shared_torch:raise ValueError('Unpaired training or execution settings')
        with (p/'accuracy_each_round.csv').open() as f:data=list(csv.DictReader(f))
        acc=[float(r['full_accuracy']) for r in data];ce=[float(r['full_ce']) for r in data]
        peak=max(range(400),key=lambda i:acc[i])
        rows.append(dict(id=name,test_oracle_peak=acc[peak],best_round=peak+1,final=acc[-1],
            last20=statistics.mean(acc[-20:]),last50=statistics.mean(acc[-50:]),final_ce=ce[-1],peak_ce=ce[peak]))
        curves[name]=acc;ce_curves[name]=ce
        with (p/'method_diagnostics.csv').open() as f:logs=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        slots=[r['slots'] for r in logs]
        if initial is None:initial=m['initial_state_sha256'];schedule=slots
        elif initial!=m['initial_state_sha256'] or schedule!=slots:raise ValueError('Unpaired initialization or schedules')
        servers=[r['server'] for r in logs]
        full_logs[name]=servers
        keys=['difference_from_e03_l2','changed_from_e03_fraction','update_l2_ratio_e03','actual_update_l2','e03_update_l2']
        def mean_present(key):
            values=[s[key] for s in servers[1:] if s.get(key) is not None]
            return statistics.mean(values) if values else None
        diagnostics[name]=dict(first_round=servers[0],later={k:mean_present(k) for k in keys},
            changed_rounds=sum(s['difference_from_e03_l2']>0 for s in servers) if all('difference_from_e03_l2' in s for s in servers) else None,
            classifier_active_rounds=sum(s.get('classifier_protection_active',False) for s in servers))
    for row in rows:
        for key in ('test_oracle_peak','final','last20','last50'):row[key+'_minus_R0']=row[key]-rows[0][key]
    contrasts=None;checks={}
    if extra_root is not None:
        contrasts=factorial_contrasts(rows)
        checks=dict(R3_later_same_input_exact_M2=all(s['difference_from_e03_l2']==0 for s in full_logs['R3'][1:]),
            R3_only_first_classifier_active=[i+1 for i,s in enumerate(full_logs['R3']) if s.get('classifier_protection_active')]==[1],
            R0_R2_first_evaluation_equal=curves['R0'][0]==curves['R2'][0] and ce_curves['R0'][0]==ce_curves['R2'][0],
            R1_R3_first_evaluation_equal=curves['R1'][0]==curves['R3'][0] and ce_curves['R1'][0]==ce_curves['R3'][0])
        if not all(checks.values()):raise ValueError('R3 or first-round paired controls differ: '+str(checks))
    out=(extra_root if extra_root is not None else root)/'report';out.mkdir(exist_ok=False)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(comparison=rows,diagnostics=diagnostics,factorial_contrasts=contrasts,checks=checks,
        interpretation='All peaks official test-oracle; paired deterministic seed1 controls, not independent validation; historical E03 is contextual only'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,values in [('accuracy',curves),('ce',ce_curves)]:
        with (out/(name+'_all_rounds.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['round',*values]);w.writerows([i+1,*[v[i] for v in values.values()]] for i in range(400))
        fig,ax=plt.subplots(figsize=(10,5))
        for label,y in values.items():ax.plot(range(1,401),y,label=label,linewidth=1)
        ax.set(xlabel='Communication round',ylabel='Full test Accuracy (%)' if name=='accuracy' else 'Full test CE')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160);plt.close(fig)
    lines=['# L04 controlled experiment','', '| ID | test-oracle peak | round | Final | Last20 | Last50 | CE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['id']}|{r['test_oracle_peak']:.4f}|{r['best_round']}|{r['final']:.4f}|{r['last20']:.4f}|{r['last50']:.4f}|{r['final_ce']:.6f}|")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--extra-root',type=Path)
    a=p.parse_args();report(a.root,a.extra_root)
