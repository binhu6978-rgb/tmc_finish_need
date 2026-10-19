"""Summarize activation evidence, without reading official-test accuracies."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def distribution(values):
    return {'n': len(values), 'positive': sum(v > 0 for v in values),
            'positive_fraction': sum(v > 0 for v in values) / len(values) if values else None,
            'mean': statistics.fmean(values) if values else None,
            'max': max(values) if values else None}


def summarize_round(server, slots, supplemental):
    direction = []; semantic = []; joint = []; confidence = []; gates = []
    counts = Counter(); two = Counter(); changes = 0; covered = 0
    changed_clients = set()
    for region in server['regions']:
        if region['slot_count']:
            covered += region['numel']
        changes += region.get('changed_coordinates', 0)
        ds = region.get('direction_risks') or supplemental.get('direction_observation_only', {}).get(str(region['region_id']), {})
        qs = region.get('semantic_risks', {})
        gs = region['gates']
        if region['client_count'] < 2:
            counts['insufficient_client_regions'] += 1
            continue
        counts['comparable_regions'] += 1
        direction.extend(ds.values()); semantic.extend(qs.values())
        confidence.extend(region.get('confidence', {}).values()); gates.extend(gs)
        for slot in region['slots']:
            cid = str(slots[slot][0]); key = str(slot)
            if key in ds and cid in qs:
                joint.append(float(ds[key] > 0 and qs[cid] > 0))
                if ds[key] > 0 and qs[cid] > 0:
                    c = region.get('confidence', {}).get(key, 1.)
                    counts['joint_positive_confidence_zero_slots' if c == 0 else 'joint_positive_confidence_positive_slots'] += 1
                if ds[key] > 0 and qs[cid] == 0:
                    counts['direction_positive_semantic_zero_slots'] += 1
            if key in ds and ds[key] > 0 and cid not in qs:
                counts['direction_positive_semantic_missing_slots'] += 1
        active = any(g < 1 - 1e-12 for g in gs)
        uniform = bool(gs) and max(gs) - min(gs) <= 1e-14
        canceled = active and uniform and region.get('normalization') == 'gated_coverage_mass'
        counts['active_gate_regions'] += active
        counts['normalized_uniform_canceled_regions'] += canceled
        counts['active_but_no_changed_coordinate_regions'] += active and not region.get('changed_coordinates', 0)
        counts['changed_regions'] += region.get('changed_coordinates', 0) > 0
        sem = region.get('semantic', {})
        counts['permuted_region_client_scores'] += sem.get('permuted_client_count', 0)
        counts['changed_region_client_scores'] += sem.get('changed_score_count', 0)
        for cid, val in sem.get('scores', {}).items():
            if val != sem['original_scores'][cid]:
                changed_clients.add(int(cid))
        if region['client_count'] == 2:
            two['regions'] += 1; two['active_gate_regions'] += active
            two['normalized_uniform_canceled_regions'] += canceled
            two['changed_regions'] += region.get('changed_coordinates', 0) > 0
            two['positive_semantic_scores'] += sum(v > 0 for v in qs.values())
            two['semantic_scores'] += len(qs)
    counts['s08_changed_distinct_clients'] = len(changed_clients)
    return {'direction': distribution(direction), 'semantic': distribution(semantic),
            'joint_positive': distribution(joint),
            'confidence': {'n': len(confidence), 'min': min(confidence) if confidence else None,
                           'mean': statistics.fmean(confidence) if confidence else None,
                           'max': max(confidence) if confidence else None},
            'gate_non_neutral_fraction': sum(g < 1 - 1e-12 for g in gates) / len(gates) if gates else 0.,
            'gate_min': min(gates) if gates else 1., 'gate_comparisons': len(gates),
            'gate_mean': statistics.fmean(gates) if gates else 1.,
            'non_neutral_gates': sum(g < 1 - 1e-12 for g in gates),
            'actual_changed_parameter_fraction_full': server['actual_changed_parameter_fraction_full'],
            'changed_coordinates': changes, 'covered_coordinates': covered,
            'difference_from_native_l2': server['difference_from_native_aggregate_l2'],
            'difference_to_native_learning_l2_ratio': server['difference_from_native_aggregate_l2'] / server['comparable_native_learning_update_l2'] if server['comparable_native_learning_update_l2'] else None,
            'counts': dict(counts), 'two_distinct_clients': dict(two),
            'history_total_events': server['history_total_events'],
            'semantic_valid_comparison_fraction': server['semantic_valid_comparison_fraction'],
            'same_input_s07_s09': supplemental.get('same_input_s07_s09')}


def analyze(root):
    result = {'kind': 'full-data short activation preflight; not formal400', 'runs': {}}
    schedules = []; identities = []; initial = []
    for mode in ('s03', 's04', 's06', 's07', 's08', 's09'):
        paths = list((root / mode).glob('activation_*'))
        if len(paths) != 1:
            raise ValueError(f'Exactly one preflight directory required for {mode}')
        path = paths[0]
        provenance = json.loads((path / 'preflight_provenance.json').read_text())
        manifest = json.loads((path / 'manifest.json').read_text())
        resources = json.loads((path / 'preflight_resources.json').read_text())
        expected = manifest['identity']['epochs']
        if resources['status'] != 'completed_short_preflight' or not (path / 'completed.json').is_file():
            raise ValueError(f'Incomplete preflight {mode}')
        if not provenance['full_data'] or provenance['train_examples'] != 50000 or provenance['test_examples'] != 10000:
            raise ValueError('Full original data required')
        extra = {r['round']: r for r in map(json.loads, (path / 'activation_observations.jsonl').read_text().splitlines())}
        with (path / 'method_diagnostics.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != expected or len(extra) != expected:
            raise ValueError(f'Round journals do not match {expected}')
        rounds = []; schedule = []
        for row in rows:
            metrics = json.loads(row['metrics_json']); r = int(row['round'])
            out = summarize_round(metrics['server'], metrics['slots'], extra[r]); out['round'] = r
            rounds.append(out); schedule.append(metrics['slots'])
        total = Counter(); two = Counter()
        for r in rounds:
            total.update(r['counts']); two.update(r['two_distinct_clients'])
        def pooled(name):
            n = sum(r[name]['n'] for r in rounds); p = sum(r[name]['positive'] for r in rounds)
            return {'n': n, 'positive': p, 'positive_fraction': p / n if n else None,
                    'mean': sum((r[name]['mean'] or 0) * r[name]['n'] for r in rounds) / n if n else None,
                    'max': max((r[name]['max'] or 0) for r in rounds)}
        gc = sum(r['gate_comparisons'] for r in rounds)
        relative = [r['difference_to_native_learning_l2_ratio'] for r in rounds if r['difference_to_native_learning_l2_ratio'] is not None]
        result['runs'][mode] = {'path': str(path), 'rounds': rounds, 'resources': resources,
            'direction': pooled('direction'), 'semantic': pooled('semantic'), 'joint_positive': pooled('joint_positive'),
            'gate_non_neutral_fraction': sum(r['non_neutral_gates'] for r in rounds) / gc if gc else 0,
            'mean_changed_parameter_fraction_full': statistics.fmean(r['actual_changed_parameter_fraction_full'] for r in rounds),
            'max_changed_parameter_fraction_full': max(r['actual_changed_parameter_fraction_full'] for r in rounds),
            'minimum_gate': min(r['gate_min'] for r in rounds),
            'mean_gate': sum(r['gate_mean'] * r['gate_comparisons'] for r in rounds) / gc if gc else 1.,
            'mean_relative_aggregate_intervention': statistics.fmean(relative) if relative else None,
            'counts': dict(total), 'two_distinct_clients': dict(two),
            'normalized_cancellation_fraction_of_active_regions': total['normalized_uniform_canceled_regions'] / total['active_gate_regions'] if total['active_gate_regions'] else None,
            'partition_sha256': provenance['partition_sha256'], 'completed_rounds': len(rounds)}
        schedules.append(schedule); identities.append(manifest['identity']); initial.append(manifest['initial_state_sha256'])
    keys = ('partition_sha256', 'source_sha256', 'seed', 'num_users', 'frac', 'local_ep', 'local_bs', 'lr', 'lr_decay', 'momentum', 'weight_decay')
    result['pairing'] = {'schedules_identical': all(s == schedules[0] for s in schedules),
                         'initial_models_identical': all(s == initial[0] for s in initial),
                         'protocol_identical': all(all(s[k] == identities[0][k] for k in keys) for s in identities)}
    return result


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('root', type=Path); opt = p.parse_args()
    result = analyze(opt.root)
    (opt.root / 'activation_summary.json').write_text(json.dumps(result, indent=2))
    fields = ['mode', 'round', 'direction_positive_fraction', 'semantic_positive_fraction', 'joint_positive_fraction',
              'gate_non_neutral_fraction', 'actual_changed_parameter_fraction_full', 'difference_from_native_l2', 'confidence_mean']
    with (opt.root / 'activation_each_round.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for mode, run in result['runs'].items():
            for r in run['rounds']:
                writer.writerow({'mode': mode, 'round': r['round'],
                    **{k + '_positive_fraction': r[k]['positive_fraction'] for k in ('direction', 'semantic')},
                    'joint_positive_fraction': r['joint_positive']['positive_fraction'],
                    **{k: r[k] for k in ('gate_non_neutral_fraction', 'actual_changed_parameter_fraction_full', 'difference_from_native_l2')},
                    'confidence_mean': r['confidence']['mean']})
    print(json.dumps({k: {x: v[x] for x in ('completed_rounds', 'semantic', 'joint_positive', 'gate_non_neutral_fraction', 'mean_changed_parameter_fraction_full', 'counts', 'two_distinct_clients')} for k, v in result['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
