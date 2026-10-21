"""Read-only completion analysis of the fixed P sweep and G03 1000 rounds.

Reads retained scalar evidence only; never loads weights or starts training.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.report_native_groups10 import validate_run, diagnostic_summary


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path, expected):
    with path.open(encoding='utf-8') as f:
        result = list(csv.DictReader(f))
    if [int(r['round']) for r in result] != list(range(1, expected + 1)):
        raise ValueError('Incomplete actual round budget: ' + str(path))
    if not all(math.isfinite(float(r[k])) for r in result for k in ('full_accuracy', 'full_ce')):
        raise ValueError('Nonfinite evaluation')
    return result


def metrics(name, records):
    acc = [float(r['full_accuracy']) for r in records]
    peak = acc.index(max(acc))
    return dict(id=name, rounds=len(records), test_oracle_peak=acc[peak], best_round=peak + 1,
                final_accuracy=acc[-1], last20_mean=statistics.mean(acc[-20:]),
                last20_std=statistics.pstdev(acc[-20:]), last50_mean=statistics.mean(acc[-50:]),
                final_ce=float(records[-1]['full_ce']), peak_ce=float(records[peak]['full_ce']))


def identity(manifest, exclude_epochs=False):
    omitted = {'method', 'options', 'source_sha256'}
    if exclude_epochs:
        omitted.add('epochs')
    return {k: v for k, v in manifest['identity'].items() if k not in omitted}


def diagnostics(path, expected):
    with path.open(encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    if [int(r['round']) for r in records] != list(range(1, expected + 1)):
        raise ValueError('Incomplete actual aggregation diagnostics')
    return [json.loads(r['metrics_json']) for r in records]


def final_check(path, expected):
    done = load(path / 'completed.json')
    manifest = load(path / 'manifest.json')
    check = load(path / 'final_model_check.json')
    if (done['success'] is not True or done['rounds_completed'] != expected
            or done['requested_epochs'] != expected or done['checkpoint'] is not None
            or done['checkpoint_policy'] != 'disabled_by_user' or manifest['save_pt'] is not False):
        raise ValueError('Completion/no-PT policy differs')
    numel = sum(math.prod(shape) for shape in check['full_shapes'].values())
    if (check['round'] != expected or check['finite_full_parameters'] is not True
            or check['full_numel'] != numel or numel != 11424356):
        raise ValueError('Final in-memory Full check differs')
    if list(path.rglob('*.pt')) or list(path.rglob('*.pending')):
        raise ValueError('Unexpected model/pending file')
    records = rows(path / 'accuracy_each_round.csv', expected)
    stats = metrics(path.name, records)
    summary = load(path / 'summary.json')
    if any(summary[k] != v for k, v in [('rounds_completed', expected),
            ('best_round', stats['best_round']), ('best_full_accuracy', stats['test_oracle_peak'])]):
        raise ValueError('Scalar peak summary differs')
    return records, diagnostics(path / 'method_diagnostics.csv', expected)


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def analyze(root, output, historical=None, draw=True):
    historical = historical or ROOT / 'docs/evidence/native_groups10_results'
    p_root, g_root = root / 'p400', root / 'g1000'
    status = load(p_root / 'status.json')
    if (status['state'] != 'completed' or status['failed']
            or set(status['completed']) != {f'P{i:02}' for i in range(1, 11)}
            or load(g_root / 'status.json')['state'] != 'completed'):
        raise ValueError('Both authorized tasks must be complete')
    baseline = historical / 'baseline'
    old = historical / 'raw/G03'
    old_rows = rows(old / 'accuracy_each_round.csv', 400)
    old_logs = diagnostics(old / 'method_diagnostics.csv', 400)
    old_manifest = load(old / 'manifest.json')
    schedule = [r['slots'] for r in old_logs]
    reference = metrics('G03', old_rows)
    base_rows = rows(baseline / 'accuracy_each_round.csv', 400)
    base = metrics('B00-D', base_rows)
    comparison, mechanisms, costs = [], {}, []
    curves = {'B00-D': base_rows, 'G03': old_rows}
    original = load(p_root / 'report/summary.json')
    coverage = None
    training_sources = None
    source_weights = []
    for i in range(1, 11):
        name = f'P{i:02}'
        p = p_root / name.lower() / (name + '_full_only_400_seed1_deterministic')
        validate_run(p)
        records, logs = final_check(p, 400)
        manifest = load(p / 'manifest.json')
        env = load(p / 'run_environment.json')
        sources = manifest['identity']['source_sha256']
        if training_sources is None:
            training_sources = sources
        if training_sources != sources or manifest['torch'] != old_manifest['torch']:
            raise ValueError('Training source/PyTorch differs within sweep')
        if (identity(manifest) != identity(old_manifest)
                or manifest['initial_state_sha256'] != old_manifest['initial_state_sha256']
                or [r['slots'] for r in logs] != schedule
                or env['cuda_visible_devices'] != '1'
                or env['train_examples'] != 50000 or env['test_examples'] != 10000
                or any(records[0][k] != old_rows[0][k] for k in ('full_accuracy', 'full_ce'))):
            raise ValueError('Unpaired protocol, initialization, schedule, data or first Native evaluation')
        servers = [r['server'] for r in logs]
        total = [0] * 9
        for log, server in zip(logs, servers):
            for region, detail in zip(manifest['structural_regions'], server['regions']):
                indices = [j for j, (_, profile) in enumerate(log['slots']) if profile in region['profiles']]
                unique = {log['slots'][j][0] for j in indices}
                if detail['slots'] != indices or detail['distinct_physical_clients'] != len(unique):
                    raise ValueError('Incorrect physical/structural coverage')
                total[region['id']] += len(unique)
                if not indices and detail['actual_changed_coordinates']:
                    raise ValueError('Uncovered parameter changed')
                if name == 'P08' and region['id'] != 0 and detail['actual_changed_coordinates']:
                    raise ValueError('P08 scope differs')
                if name == 'P09' and region['id'] == 0 and detail['actual_changed_coordinates']:
                    raise ValueError('P09 scope differs')
        if coverage is None:
            coverage = total
        if coverage != total:
            raise ValueError('Coverage opportunities differ')
        item = metrics(name, records)
        prior = next(r for r in original['comparison'] if r['id'] == name)
        if any(item[k] != prior[k] for k in item if k in prior):
            raise ValueError('Server report differs from raw values')
        for key in ('test_oracle_peak', 'final_accuracy', 'last20_mean', 'last50_mean', 'final_ce'):
            item[key + '_minus_G03'] = item[key] - reference[key]
            item[key + '_minus_B00_D'] = item[key] - base[key]
        comparison.append(item)
        curves[name] = records
        m, _ = diagnostic_summary(p / 'method_diagnostics.csv')
        m['full_training_slots'] = sum(profile == 6 for log in logs for _, profile in log['slots'])
        m['duplicate_client_rounds'] = sum(len({c for c, _ in log['slots']}) < 10 for log in logs)
        mechanisms[name] = m
        share = m['source_shares'][0]['applied_slot_shares']
        source_weights.append(dict(id=name, low=share[0], mid=share[1], high=share[2],
            update_l2_ratio_native=m['later_update_l2_ratio_native_mean'],
            cosine_native=m['later_actual_update_cosine_native_mean'],
            cosine_same_input_g03=m['later_actual_update_cosine_g03_mean'],
            difference_same_input_g03_l2=m['later_difference_from_g03_l2_mean']))
        resource = load(p / 'resources.json')
        costs.append(dict(id=name, wall_minutes=resource['wall_seconds'] / 60,
            peak_allocated_mib=resource['peak_allocated_bytes'] / 2**20,
            peak_reserved_mib=resource['peak_reserved_bytes'] / 2**20,
            server_seconds=sum(float(r['server_seconds']) for r in records),
            evaluation_seconds=sum(float(r['evaluation_seconds']) for r in records)))
    g = g_root / 'g03/G03_full_only_1000_seed1_deterministic'
    g_records, g_logs = final_check(g, 1000)
    g_manifest, g_env = load(g / 'manifest.json'), load(g / 'run_environment.json')
    if (identity(g_manifest, True) != identity(old_manifest, True)
            or g_manifest['initial_state_sha256'] != old_manifest['initial_state_sha256']
            or [r['slots'] for r in g_logs[:400]] != schedule or g_env['cuda_visible_devices'] != '0'
            or g_env['train_examples'] != 50000 or g_env['test_examples'] != 10000):
        raise ValueError('G03 1000 protocol differs beyond budget')
    first400_equal = all(all(a[k] == b[k] for k in ('full_accuracy', 'full_ce'))
                         for a, b in zip(g_records[:400], old_rows))
    long = [reference, metrics('G03 new first400', g_records[:400]), metrics('G03 1000', g_records)]
    by_id = {r['id']: r for r in comparison}
    contrasts = {a + '-' + b: {k: by_id[a][k] - by_id[b][k] for k in
        ('test_oracle_peak', 'final_accuracy', 'last20_mean', 'last50_mean', 'final_ce')}
        for a, b in [('P04', 'P05'), ('P02', 'P01'), ('P08', 'P09')]}
    numels = [sum(r['numel'] for r in old_manifest['structural_regions'] if profile in r['profiles']) for profile in range(7)]
    result = dict(comparison=comparison, reference=reference, baseline=base, g03_long=long,
        mechanisms=mechanisms, contrasts=contrasts, coverage_unique_physical_totals=coverage,
        verification=dict(ten_actual_400_and_one_actual_1000=True, all_eval_diagnostics_contiguous_finite=True,
            final_in_memory_finite_full_shapes_valid=True, no_pt_requested_and_observed=True,
            same_init_protocol_400_slot_schedule=True, first_round_native_identical=True,
            p_training_source_fingerprints_identical=True,
            g03_1000_first400_accuracy_and_ce_exactly_equal_historical=first400_equal,
            note='Scalar equality is observed, not a saved-weight bitwise comparison across cards'),
        communication_per_400_run=dict(uplink_float32_state_bytes=sum(numels[p] * 4 for slots in schedule for _, p in slots),
            note='Same downlink payload; no new client payload; excludes transport headers'),
        g03_1000_resources=load(g / 'resources.json'),
        inference='Single-seed official test-oracle exploration; own-input L2 controls, not shared later trajectories')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_csv(output / 'comparison.csv', comparison)
    write_csv(output / 'g03_1000_comparison.csv', long)
    write_csv(output / 'source_weights_and_geometry.csv', source_weights)
    write_csv(output / 'costs.csv', costs)
    if draw:
        plots(output, curves, g_records)
    return result


def plots(output, curves, g_records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key, ylabel, filename in [('full_accuracy', 'Full test Accuracy (%)', 'late_accuracy'),
                                  ('full_ce', 'Full test CE', 'late_ce')]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, names in zip(axes.flat, [('P01', 'P02'), ('P03', 'P04', 'P05'), ('P06',), ('P07',), ('P08', 'P09'), ('P10',)]):
            for name in ('B00-D', 'G03', *names):
                ax.plot(range(301, 401), [float(r[key]) for r in curves[name][300:]], label=name,
                        linestyle='--' if name in ('B00-D', 'G03') else '-', linewidth=1)
            ax.set(xlabel='Round', ylabel=ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / (filename + '.png'), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, ylabel in zip(axes, ['full_accuracy', 'full_ce'], ['Full test Accuracy (%)', 'Full test CE']):
        ax.plot(range(1, 1001), [float(r[key]) for r in g_records], label='G03 1000', linewidth=1)
        ax.plot(range(1, 401), [float(r[key]) for r in curves['G03']], label='G03 historical 400', linestyle='--', linewidth=1)
        ax.axvline(400, color='grey', linestyle=':', linewidth=.8)
        ax.set(xlabel='Round', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'g03_1000_accuracy_ce.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'docs/evidence/g03_completed_results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)
