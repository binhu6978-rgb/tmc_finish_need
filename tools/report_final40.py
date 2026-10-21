"""Offline forty-run tables/curves, within-condition contrasts and diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_final40 import jobs
from tools.report_native_groups10 import validate_run
from tools.report_next10_m2_m3 import describe

REFERENCES = {
    'B00-D': 'runs/e03_l04_final_pair400_seed1/baseline/B00-D_full_only_400_seed1_deterministic',
    'G03': 'runs/native_groups10_full400_seed1/g03/G03_full_only_400_seed1_deterministic',
    'P04': 'runs/g03_refinements10_full400_seed1_gpu1_nopt_20261021/p04/P04_full_only_400_seed1_deterministic',
    'P05': 'runs/g03_refinements10_full400_seed1_gpu1_nopt_20261021/p05/P05_full_only_400_seed1_deterministic'
}
METRICS = ('test_oracle_peak', 'final_accuracy', 'last20_mean', 'last50_mean', 'final_ce')


def load(path):
    return json.loads(path.read_text())


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def read_logs(path):
    with (path / 'accuracy_each_round.csv').open() as f:
        scores = list(csv.DictReader(f))
    with (path / 'method_diagnostics.csv').open() as f:
        logs = [json.loads(r['metrics_json']) for r in csv.DictReader(f)]
    return scores, logs


def summarize_diagnostics(logs):
    servers = [r['server'] for r in logs]
    source, regions = [], []
    for rid in range(len(servers[0]['regions'])):
        for log, server in zip(logs[1:], servers[1:]):
            r = server['regions'][rid]
            if not r['slots']:
                continue
            native, applied = [0.] * 3, [0.] * 3
            fraction = r['weighted_parameter_numel'] / r['numel']
            for i, a, b in zip(r['slots'], r['native_slot_shares'], r['applied_slot_shares']):
                tier = sum(log['slots'][i][0] >= edge for edge in server['boundaries'])
                native[tier] += a
                applied[tier] += a * (1 - fraction) + b * fraction
            source.append(dict(region=rid, native=native, applied=applied))
        rr = [s['regions'][rid] for s in servers]
        regions.append(dict(region=rid, numel=rr[0]['numel'], support=rr[0]['support_profiles'],
            covered_rounds=sum(bool(r['slots']) for r in rr),
            unique_physical_coverages=sum(r['distinct_physical_clients'] for r in rr),
            changed_rounds=sum(r['actual_changed_coordinates'] > 0 for r in rr),
            mean_changed_coordinate_fraction=mean(r['actual_changed_coordinates'] / r['numel'] for r in rr)))
    region_shares = {}
    for rid in range(len(regions)):
        values = [r for r in source if r['region'] == rid]
        region_shares[rid] = {key: [mean(v[key][g] for v in values) for g in range(3)] for key in ('native', 'applied')}
    return dict(regions=regions, source_shares_by_region=region_shares,
        full_training_slots=sum(p == 6 for r in logs for _, p in r['slots']),
        physical_slot_counts=[sum(sum(c >= edge for edge in servers[0]['boundaries']) == g for r in logs for c, _ in r['slots']) for g in range(3)],
        changed_rounds=sum(s['difference_from_native_l2'] > 0 for s in servers),
        update_l2_ratio_native_mean=mean(s['actual_update_l2_ratio_native'] for s in servers[1:] if s['actual_update_l2_ratio_native'] is not None),
        cosine_native_mean=mean(s['actual_update_cosine_native'] for s in servers[1:] if s['actual_update_cosine_native'] is not None),
        cosine_p04_mean=mean(s['actual_update_cosine_p04'] for s in servers[1:] if s['actual_update_cosine_p04'] is not None),
        difference_from_p04_l2_mean=mean(s['difference_from_p04_l2'] for s in servers[1:]),
        matched_norm_relative_error_max=max((s['matched_norm_relative_error'] for s in servers if s['matched_norm_relative_error'] is not None), default=None))


def report(root):
    prepared = load(root / 'prepared/manifest.json')
    references, curves, records, mechanisms, costs, first = {}, {}, [], {}, [], {}
    for name, relative in REFERENCES.items():
        path = ROOT / relative
        references[name], _ = describe(name, name.lower(), path / 'accuracy_each_round.csv', 2)
        curves[name], _ = read_logs(path)
    a_schedule = read_logs(ROOT / REFERENCES['P04'])[1]
    for gpu in (0, 1):
        status = load(root / f'gpu{gpu}/status.json')
        if status['state'] != 'completed' or status['failed'] or len(status['completed']) != 20:
            raise ValueError('Require actual twenty completed jobs on each card')
        for job in sum(jobs(gpu, Path(sys.executable), Path('.'), root), []):
            path = Path(job['output'])
            validate_run(path)
            name, condition = job['id'], job['condition']
            item, _ = describe(name, job['method'], path / 'accuracy_each_round.csv', 2)
            item.update(condition=condition, dataset=prepared[condition]['condition']['dataset'],
                        alpha=prepared[condition]['condition']['alpha'], ratio=prepared[condition]['condition']['ratio'])
            manifest = load(path / 'manifest.json')
            scores, logs = read_logs(path)
            if manifest['identity']['partition_sha256'] != prepared[condition]['sha256']:
                raise ValueError('Condition partition differs')
            if manifest['identity']['num_classes'] != prepared[condition]['condition']['classes']:
                raise ValueError('Output classes differ')
            condition_identity = {k: v for k, v in manifest['identity'].items() if k not in ('method', 'options')}
            paired = (manifest['initial_state_sha256'], [r['slots'] for r in logs], condition_identity,
                      scores[0]['full_accuracy'], scores[0]['full_ce'])
            if condition not in first:
                first[condition] = paired
            if first[condition] != paired or logs[0]['server']['difference_from_native_l2'] != 0:
                raise ValueError('Within-condition initialization/protocol/schedule/Native warmup differs')
            if condition == 'A' and [r['slots'] for r in logs] != [r['slots'] for r in a_schedule]:
                raise ValueError('Original A sampling schedule differs')
            check = load(path / 'final_model_check.json')
            if check['full_shapes']['classifier.weight'][0] != prepared[condition]['condition']['classes']:
                raise ValueError('Actual classifier head differs')
            curves[name] = scores
            records.append(item)
            mechanisms[name] = summarize_diagnostics(logs)
            resources = load(path / 'resources.json')
            costs.append(dict(id=name, wall_seconds=resources['wall_seconds'],
                server_seconds=sum(float(r['server_seconds']) for r in scores),
                evaluation_seconds=sum(float(r['evaluation_seconds']) for r in scores),
                peak_allocated_mib=resources['peak_allocated_bytes'] / 2**20,
                peak_reserved_mib=resources['peak_reserved_bytes'] / 2**20))
    for record in records:
        condition = record['condition']
        native = references['B00-D'] if condition == 'A' else next(r for r in records if r['id'] == condition + '-Native')
        for k in METRICS:
            record[k + '_minus_native'] = record[k] - native[k]
            for name in ('G03', 'P04', 'P05'):
                record[k + '_minus_historical_' + name] = record[k] - references[name][k] if condition == 'A' else None
    cross = []
    for method in ('G03', 'P04', 'P05'):
        selected = [r for r in records if r['id'].endswith('-' + method)]
        cross.append(dict(method=method, conditions=5,
            peak_wins=sum(r['test_oracle_peak_minus_native'] > 0 for r in selected),
            last20_wins=sum(r['last20_mean_minus_native'] > 0 for r in selected),
            last50_wins=sum(r['last50_mean_minus_native'] > 0 for r in selected),
            final_wins=sum(r['final_accuracy_minus_native'] > 0 for r in selected),
            final_ce_lower=sum(r['final_ce_minus_native'] < 0 for r in selected)))
    output = root / 'report'
    output.mkdir(exist_ok=False)
    write_csv(output / 'comparison40.csv', records)
    write_csv(output / 'costs.csv', costs)
    write_csv(output / 'cross_condition_win_counts.csv', cross)
    (output / 'summary.json').write_text(json.dumps(dict(comparison=records, references=references,
        mechanisms=mechanisms, cross_condition_counts=cross, paired=True,
        interpretation='Each condition analyzed relative to its own Native; official test-oracle peaks, one seed; no cross-dataset absolute ranking'), indent=2))
    for key, label in [('full_accuracy', 'Full test Accuracy (%)'), ('full_ce', 'Full test CE')]:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        for ax, pair in zip(axes.flat, [sum(jobs(0, Path('.'), Path('.'), root), [])[i:i+2] for i in range(0, 20, 2)]):
            for name in (*REFERENCES, *[j['id'] for j in pair]):
                ax.plot(range(1, 401), [float(r[key]) for r in curves[name]], label=name,
                        linestyle='--' if name in REFERENCES else '-', linewidth=.9)
            ax.set(xlabel='Round', ylabel=label); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(output / ('gpu0_' + key + '.png'), dpi=140); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, condition in zip(axes.flat, ['S1', 'S2', 'S3', 'S4', 'S5']):
            for method in ('Native', 'G03', 'P04', 'P05'):
                name = condition + '-' + method
                ax.plot(range(1, 401), [float(r[key]) for r in curves[name]], label=method, linewidth=1)
            ax.set(title=condition + ' ' + str(prepared[condition]['condition']), xlabel='Round', ylabel=label)
            ax.legend(); ax.grid(alpha=.2)
        axes.flat[-1].axis('off'); fig.tight_layout(); fig.savefig(output / ('gpu1_' + key + '.png'), dpi=150); plt.close(fig)
        write_csv(output / (key + '_all_rounds.csv'), [dict(round=i+1, **{name: data[i][key] for name, data in curves.items()}) for i in range(400)])
    lines = ['# Fixed forty-run results', '',
        'All peaks are official test-oracle peaks; one seed; compare conditions to their own Native.', '',
        '| ID | Peak | Round | Final | Last20 | Last50 | Final CE |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in [*references.values(), *records]:
        lines.append(f"|{r['id']}|{r['test_oracle_peak']:.4f}|{r['best_round']}|{r['final_accuracy']:.4f}|{r['last20_mean']:.4f}|{r['last50_mean']:.4f}|{r['final_ce']:.6f}|")
    candidates = [r['id'] for r in records if r['condition'] == 'A' and r['test_oracle_peak'] > references['P05']['test_oracle_peak']
        and r['last20_mean'] > references['P04']['last20_mean'] and r['final_ce'] < references['P04']['final_ce']]
    lines += ['', 'A methods jointly exceeding P05 peak, P04 Last20 and P04 final CE: ' + (', '.join(candidates) or 'none'),
        'These are exploration criteria, not independent validation; no automatic new training is scheduled.',
        'Detailed source shares, region coverage and same-input L2/cosines: summary.json. Cross-condition win counts: cross_condition_win_counts.csv.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--root', type=Path, required=True)
    report(p.parse_args().root)
