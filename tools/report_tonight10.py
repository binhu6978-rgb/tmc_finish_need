"""Offline results only: completion validation, curves and fixed comparisons."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.report_next10_m2_m3 import describe


def validate(path):
    import torch
    done=json.loads((path/'completed.json').read_text())
    item,_=describe(path.name,path.parent.name,path/'accuracy_each_round.csv',2)
    state=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
    if not done['success'] or done['rounds_completed']!=400 or state['next_round']!=400 or len(state['rows'])!=400:
        raise ValueError('Not400 actual completed rounds: '+str(path))
    best=torch.load(path/'best_full.pt',map_location='cpu',weights_only=False)
    if best['round']!=item['best_round'] or best['full_accuracy']!=item['test_oracle_peak']:
        raise ValueError('Best snapshot does not match first test-oracle peak')
    return item


def diagnostic_summary(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    metrics=[json.loads(r['metrics_json']) for r in rows]
    servers=[m['server'] for m in metrics]
    def avg(key,subset=servers):
        a=[x[key] for x in subset if x.get(key) is not None]
        return statistics.mean(a) if a else None
    counts=[s.get('conflict_slot_count',0) for s in servers]
    return {'rounds':len(rows),'conflict_slots':sum(counts),'conflict_rounds':sum(n>0 for n in counts),
        'slot_comparisons':len(rows)*10,
        'same_input_e03_difference_l2_mean':avg('difference_from_e03_l2'),
        'same_input_e03_changed_fraction_mean':avg('changed_from_e03_fraction'),
        'later_l2_ratio_e03_mean':avg('update_l2_ratio_e03',servers[1:]),
        'later_effective_step_e03_mean':avg('effective_step_along_e03',servers[1:]),
        'later_update_cosine_e03_mean':avg('update_cosine_e03',servers[1:]),
        'first_round':servers[0],
        'protected_components':sum(len(s.get('protected_components',[])) for s in servers),
        'weight_mean':statistics.mean(v for s in servers for v in s['conflict_weights']),
        'weight_min':min(v for s in servers for v in s['conflict_weights']),
        'phases':{f'{a}-{b}':{'conflict_slots':sum(counts[a-1:b]),
            'same_input_e03_difference_l2_mean':avg('difference_from_e03_l2',servers[a-1:b]),
            'update_l2_ratio_e03_mean':avg('update_l2_ratio_e03',servers[a-1:b])}
            for a,b in ((2,100),(101,200),(201,300),(301,400))}},[m['slots'] for m in metrics]


def report(root,references,c1):
    config=json.loads((ROOT/'configs/e03_tonight10_400.json').read_text())
    records=[];curves={};ce_curves={};diagnostics={};failed={};initial=None;schedule=None
    for method in config['order']:
        name='C2_full_only_400_seed1_after_update' if method=='c2' else method.upper()+'_full_only_400_seed1'
        p=root/method/name
        try:
            validate(p)
        except (OSError,ValueError,KeyError) as error:
            failed[method]=str(error);continue
        identifier=method.upper()
        item,curve=describe(identifier,method,p/'accuracy_each_round.csv',2);records.append(item);curves[identifier]=curve
        with (p/'accuracy_each_round.csv').open() as f:ce_curves[identifier]=[float(r['full_ce']) for r in csv.DictReader(f)]
        diagnostics[identifier],own_schedule=diagnostic_summary(p/'method_diagnostics.csv')
        own_initial=json.loads((p/'manifest.json').read_text())['initial_state_sha256']
        if initial is None:initial,schedule=own_initial,own_schedule
        elif initial!=own_initial or schedule!=own_schedule:raise ValueError('Completed new runs do not share initialization/schedule')
    for name,p in [('B00',references/'B00.csv'),('E03',references/'E03.csv'),('N08',references/'N08.csv'),('C1',c1/'accuracy_each_round.csv')]:
        if name=='C1':validate(c1)
        item,curve=describe(name,'historical',p,2);records.append(item);curves[name]=curve
        with p.open() as f:ce_curves[name]=[float(r['full_ce']) for r in csv.DictReader(f)]
    e03=next(r for r in records if r['id']=='E03')
    for r in records:
        r['peak_minus_e03_pp']=r['test_oracle_peak']-e03['test_oracle_peak']
        r['last20_minus_e03_pp']=r['last20_mean']-e03['last20_mean']
        r['last50_minus_e03_pp']=r['last50_mean']-e03['last50_mean']
    output=root/'report';output.mkdir(exist_ok=False)
    def write_csv(path,rows,fields):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_csv(output/'comparison.csv',records,list(records[0]))
    (output/'summary.json').write_text(json.dumps({'comparison':records,'mechanisms':diagnostics,'failed_or_incomplete':failed,
        'peak_label':'test-oracle peak','source':'same-input E03 diagnostics are counterfactual computations, not rerun historical E03',
        'pairing':'all completed new runs share actual initialization and400 client/profile schedules'},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for data,label,filename in [(curves,'Full test Accuracy (%)','accuracy'),(ce_curves,'Full test CE','ce')]:
        write_csv(output/(filename+'_all_rounds.csv'),[{'round':i+1,**{n:v[i] for n,v in data.items()}} for i in range(400)],['round',*data])
        fig,ax=plt.subplots(figsize=(12,6))
        for name,values in data.items():ax.plot(range(1,401),values,label=name,linewidth=1,linestyle='--' if name in ('B00','E03','N08','C1') else '-')
        ax.set(xlabel='Communication round',ylabel=label);ax.legend(ncol=5);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/(filename+'.png'),dpi=160);plt.close(fig)
    text=['# E03 fixed mechanism controls:400 rounds','',
        'Single seed, official-test **test-oracle peaks**; exploratory comparisons, not independent validation. No online tuning.','',
        '|Method|test-oracle peak|Round|Final|Last20|Last50|Final CE|Peak minus E03 (pp)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:text.append('|{id}|{test_oracle_peak:.4f}|{best_round}|{final_accuracy:.4f}|{last20_mean:.4f}|{last50_mean:.4f}|{final_ce:.6f}|{peak_minus_e03_pp:+.4f}|'.format(**r))
    by_name={r['id']:r for r in records}
    text+=['','## Fixed contrasts','', '|Comparison (first minus second)|Peak difference (pp)|Last20 difference (pp)|Last50 difference (pp)|',
        '|---|---:|---:|---:|']
    for first,second in [('T01','T02'),('T03','T01'),('T04','E03'),('T05','E03'),('T06','T05'),('T07','T08'),('T09','E03'),('T09','C2'),('T10','T01')]:
        if first in by_name and second in by_name:
            a,b=by_name[first],by_name[second]
            text.append(f"|{first} - {second}|{a['test_oracle_peak']-b['test_oracle_peak']:+.4f}|{a['last20_mean']-b['last20_mean']:+.4f}|{a['last50_mean']-b['last50_mean']:+.4f}|")
    candidates=[r for r in records if r['id'].startswith('T')]
    if candidates:
        peak=max(candidates,key=lambda r:r['test_oracle_peak']);late=max(candidates,key=lambda r:r['last20_mean'])
        text+=['',f"Highest new observed test-oracle peak: {peak['id']} ({peak['test_oracle_peak']:.4f}%). Exceeds historical E03: {peak['test_oracle_peak']>e03['test_oracle_peak']}.",
            f"Highest new Last20: {late['id']} ({late['last20_mean']:.4f}%). Historical E03 Last20: {e03['last20_mean']:.4f}%.",
            'These are descriptive rankings. A peak/late-mean tradeoff, mechanism and total update-scale changes must be considered together before selecting1-2 research candidates; they do not establish causal or generalization superiority.']
    text+=['','Failed/incomplete: '+json.dumps(failed),
        '', 'Mechanism diagnostics compare each run with E03 on its own uploads. They do not isolate a unique causal explanation from total update scale.',
        'T05 changes physical-reference pooling; T06 also changes LOO averaging to a direct-sum pairwise-overlap cosine. Their differences cannot be attributed only to larger support.',
        'T09 preserves E03 fusion and negative-projection coefficient, restoring only underweighted negative-slot residuals. Residuals are not established semantic knowledge.',
        '', '![Full Accuracy](accuracy.png)','', '![Full CE](ce.png)']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'completed_new':len(diagnostics),'failed':failed,'report':str(output)},indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--validate-run',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--references',type=Path);p.add_argument('--c1',type=Path)
    a=p.parse_args()
    if a.validate_run:validate(a.validate_run)
    else:report(a.root,a.references,a.c1)


if __name__=='__main__':main()
