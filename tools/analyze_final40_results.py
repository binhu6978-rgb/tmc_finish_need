"""Read-only final evidence audit and fixed descriptive contrasts, no training."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_final40 import jobs
from tools.report_native_groups10 import validate_run
from tools.report_next10_m2_m3 import describe
from tools.report_final40 import read_logs, write_csv, METRICS


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(root, output, draw=True):
    original = load(root / 'report/summary.json')
    by_id = {r['id']: r for r in original['comparison']}
    if load(root / 'status.json')['state'] != 'completed' or len(by_id) != 40:
        raise ValueError('Exactly forty completed jobs required')
    prepared = load(root / 'prepared/manifest.json')
    curves, schedule, sources, coverage, special, costs = {}, {}, None, {}, {}, []
    for gpu in (0, 1):
        state = load(root / f'gpu{gpu}/status.json')
        if state['failed'] or len(state['completed']) != 20 or state['state'] != 'completed':
            raise ValueError('Each card must actually complete twenty jobs')
        for job in sum(jobs(gpu, Path('.'), Path('.'), root), []):
            name, condition = job['id'], job['condition']
            if state['jobs'][name]['exit_code'] != 0:
                raise ValueError('A child did not exit successfully')
            path = Path(job['output'])
            validate_run(path)
            manifest = load(path / 'manifest.json')
            env = load(path / 'run_environment.json')
            check = load(path / 'final_model_check.json')
            numel = sum(math.prod(v) for v in check['full_shapes'].values())
            classes = prepared[condition]['condition']['classes']
            if (numel != check['full_numel'] or numel != (11378186 if classes == 10 else 11424356)
                    or check['full_shapes']['classifier.weight'][0] != classes
                    or env['cuda_visible_devices'] != str(gpu)
                    or env['train_examples'] != 50000 or env['test_examples'] != 10000):
                raise ValueError('Actual class head, final shapes, data or device differs')
            fingerprint = manifest['identity']['source_sha256']
            if sources is None:
                sources = fingerprint
            if sources != fingerprint:
                raise ValueError('Training source changed between jobs')
            actual, _ = describe(name, job['method'], path / 'accuracy_each_round.csv', 2)
            if any(actual[k] != by_id[name][k] for k in actual):
                raise ValueError('Server summary differs from raw CSV')
            scores, logs = read_logs(path)
            curves[name] = scores
            pairing = (manifest['initial_state_sha256'], [r['slots'] for r in logs],
                       manifest['identity']['partition_sha256'], scores[0]['full_accuracy'], scores[0]['full_ce'])
            if condition not in schedule:
                schedule[condition] = pairing
            if pairing != schedule[condition] or pairing[2] != prepared[condition]['sha256']:
                raise ValueError('Condition pairing differs')
            totals = [0] * 9
            for log in logs:
                server = log['server']
                for region, detail in zip(manifest['structural_regions'], server['regions']):
                    indices = [i for i, (_, p) in enumerate(log['slots']) if p in region['profiles']]
                    physical = {log['slots'][i][0] for i in indices}
                    if indices != detail['slots'] or len(physical) != detail['distinct_physical_clients']:
                        raise ValueError('Real prefix coverage/physical identities differ')
                    totals[region['id']] += len(physical)
                    if not indices and detail['actual_changed_coordinates'] != 0:
                        raise ValueError('Uncovered coordinates changed')
                    if name == 'A05' and len(region['profiles']) != 7 and detail['actual_changed_coordinates']:
                        raise ValueError('A05 outside common scope')
                    if name == 'A06' and not 1 < len(region['profiles']) < 7 and detail['actual_changed_coordinates']:
                        raise ValueError('A06 outside partial-sharing scope')
                    if len(region['profiles']) == 1 and name not in ('A01','A02','A03','A04') and detail['actual_changed_coordinates']:
                        raise ValueError('Single-source Full-only area should be Native')
            if condition in coverage and coverage[condition] != totals:
                raise ValueError('Coverage opportunities differ within condition')
            coverage[condition] = totals
            if name in ('A07', 'A09', 'A11', 'A16'):
                servers = [r['server'] for r in logs[1:]]
                special[name] = dict(actual_to_same_input_p04_norm_ratio_mean=statistics.mean(
                    s['actual_update_l2'] / s['virtual_p04_update_l2'] for s in servers if s['virtual_p04_update_l2']),
                    different_from_p04_rounds=sum(s['difference_from_p04_l2'] > 0 for s in servers))
            if name.endswith('-Native'):
                model_numel = [sum(r['numel'] for r in manifest['structural_regions'] if p in r['profiles']) for p in range(7)]
                costs.append(dict(condition=condition,
                    unique_full_physical_clients=len({cid for log in logs for cid, p in log['slots'] if p == 6}),
                    full_training_slots=sum(p == 6 for log in logs for _, p in log['slots']),
                    physical_slot_counts=original['mechanisms'][name]['physical_slot_counts'],
                    unique_region_physical_coverages=totals,
                    uplink_float32_state_bytes=sum(model_numel[p]*4 for log in logs for _, p in log['slots'])))
    contrasts = {}
    for a, b in [('A07','A09'),('A10','A07'),('A12','A13'),('A15','A14'),('A16','A17')]:
        contrasts[a+'-'+b] = {k: by_id[a][k] - by_id[b][k] for k in METRICS}
    cross = []
    for c in ('S1','S2','S3','S4','S5'):
        for method in ('G03','P04','P05'):
            r = by_id[c+'-'+method]
            cross.append(dict(condition=c, dataset=r['dataset'], method=method,
                **{k+'_minus_native': r[k+'_minus_native'] for k in METRICS}))
    candidates = [name for name, r in by_id.items() if name.startswith('A')
        and r['test_oracle_peak'] > original['references']['P05']['test_oracle_peak']
        and r['last20_mean'] > original['references']['P04']['last20_mean']
        and r['final_ce'] < original['references']['P04']['final_ce']]
    result = dict(verification=dict(actual40x400=True, finite_evaluations_and_final_shapes=True,
        same_frozen_training_sources=True, within_condition_initialization_partition_schedule_and_first_eval_paired=True,
        physical_prefix_coverage_correct=True, first_native_and_uncovered_protection=True,
        actual_class_heads_10_and_100=True, no_pt=True), coverage=coverage,
        same_input_p04_special=special, contrasts=contrasts, cross_condition_delta=cross,
        fixed_three_metric_candidates=candidates, source_fingerprints=sources,
        note='Same-input virtual norms belong to each method own trajectory; rounds and conditions are not independent seeds')
    output.mkdir(parents=True, exist_ok=False)
    (output/'verification_and_contrasts.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_csv(output/'cross_condition_deltas.csv', cross)
    write_csv(output/'participation_and_communication.csv', costs)
    if draw:
        plots(root, output, original, curves)
    return result


def plots(root, output, original, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref_root = ROOT/'docs/evidence/g03_completed_results/p400'
    for name in ('P04','P05'):
        curves[name], _ = read_logs(ref_root/name.lower()/(name+'_full_only_400_seed1_deterministic'))
    for key, ylabel, filename in [('full_accuracy','Full test Accuracy (%)','late_fixed_contrasts'),
                                   ('full_ce','Full test CE','late_ce_contrasts')]:
        fig, axes = plt.subplots(2,3,figsize=(16,9))
        for ax, names in zip(axes.flat, [('A01','A02'),('A03','A04'),('A07','A09','A10'),
                                       ('A11','A12','A13'),('A14','A15','A16','A17'),('A18','A19','A20')]):
            for name in ('P04','P05',*names):
                ax.plot(range(301,401),[float(r[key]) for r in curves[name][300:]],label=name,
                    linestyle='--' if name in ('P04','P05') else '-',linewidth=1)
            ax.set(xlabel='Round',ylabel=ylabel);ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/(filename+'.png'),dpi=160);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,title in zip(axes,['test_oracle_peak','last20_mean'],['test-oracle peak delta','Last20 delta']):
        for offset,method in zip([-.24,0,.24],['G03','P04','P05']):
            values=[next(r for r in original['comparison'] if r['id']==f'S{i}-{method}')[key+'_minus_native'] for i in range(1,6)]
            ax.bar([i+offset for i in range(5)],values,width=.24,label=method)
        ax.axhline(0,color='black',linewidth=.7);ax.set(xticks=range(5),xticklabels=['C100 .2','C100 .5','C100 1.0','C100 2:3:5','C10 .3'],
            ylabel='Within-condition improvement (percentage points)',title=title)
        ax.tick_params(axis='x',labelrotation=20);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(output/'within_condition_deltas.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.root,a.output)
