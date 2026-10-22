"""Precommitted train-side utility analysis. Official-test scores are not inputs."""
import argparse
from collections import defaultdict, Counter
import json
import hashlib
import math
from pathlib import Path
import sys
import subprocess
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable'); result = np.empty(len(values), float)
    start = 0
    while start < len(values):
        stop = start+1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start+stop-1)/2+1
        start = stop
    return result


def auc(score, labels):
    y = np.asarray(labels, bool); n = int(y.sum()); m = len(y)-n
    if not n or not m:
        return None
    return float((ranks(score)[y].sum()-n*(n+1)/2)/(n*m))


def correlation(x, y, rank=False):
    if len(x) < 3:
        return None
    a = ranks(x) if rank else np.asarray(x, float)
    b = ranks(y) if rank else np.asarray(y, float)
    a = a-a.mean(); b = b-b.mean(); norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a.dot(b)/norm) if norm else None


def controls(row):
    return [row['direction_cosine'] or 0., float(row['direction_cosine'] is None),
        math.log(max(row['scaled_update_l2'], 1e-12)), row['alpha'], row['before_ce'],
        row['own_gain'], math.log(row['sender_sample_mass']),
        *[float(row['sender_profile']==p) for p in range(7)],
        *[float(row['receiver_profile']==p) for p in range(7)],
        *[float(row['sender_tier']==p) for p in range(3)],
        *[float(row['receiver_tier']==p) for p in range(3)],
        *[float(row.get('region')==p) for p in range(9)]]


def history_features(row):
    values = []
    for key in ('client', 'client_profile'):
        h = row['reliability'][key]
        values.extend([math.log1p(h['support']), float(h['ready']), (h['age'] or 0)/100,
            h['mean'] or 0., h['std'] or 0.])
    return values


def features(row, score=None, current=False, history=False):
    result = controls(row)
    if current:
        result += [row['evidence']['current_embedding_change'], row['evidence']['current_task_change']]
    if score:
        result.append(row['evidence'][score])
    if history:
        result += history_features(row)
    return result


def predict(train, test, score=None, current=False, history=False):
    x = np.array([features(r, score, current, history) for r in train], float)
    v = np.array([features(r, score, current, history) for r in test], float)
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-12] = 1.
    x = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    v = np.column_stack([np.ones(len(v)), (v-mean)/scale])
    target = torch.tensor([r['delta_ce'] for r in train], dtype=torch.float64)
    beta = torch.linalg.lstsq(torch.from_numpy(x), target, driver='gelsd').solution
    return (torch.from_numpy(v) @ beta).numpy()


def grouped_cv(rows, score):
    errors = []; count = 0
    for fold in range(5):
        train = [r for r in rows if r['sender'] % 5 != fold]
        test = [r for r in rows if r['sender'] % 5 == fold]
        if len(train) < 30 or len(test) < 5:
            continue
        prediction = predict(train, test, score)
        errors.extend((prediction-np.array([r['delta_ce'] for r in test]))**2); count += len(test)
    return dict(mse=float(np.mean(errors)) if errors else None, evaluated_pairs=count)


def simple_control(train, test, kind):
    def xx(rows):
        if kind=='direction':
            return np.array([[1., r['direction_cosine'] or 0., float(r['direction_cosine'] is None)] for r in rows])
        return np.array([[1., math.log(max(r['scaled_update_l2'], 1e-12))] for r in rows])
    x, v = xx(train), xx(test)
    beta = torch.linalg.lstsq(torch.from_numpy(x), torch.tensor([r['delta_ce'] for r in train],dtype=torch.float64),driver='gelsd').solution
    prediction = (torch.from_numpy(v)@beta).numpy()
    valid = [r for r in test if kind!='direction' or r['direction_cosine'] is not None]
    score = [-r['direction_cosine'] if kind=='direction' else math.log(max(r['scaled_update_l2'],1e-12)) for r in valid]
    return dict(mse=float(np.mean((prediction-np.array([r['delta_ce'] for r in test]))**2)),
        auc=auc(score,[r['delta_ce']>0 for r in valid]), pairs=len(valid),
        spearman=correlation(score,[r['delta_ce'] for r in valid], True),
        definition='negative cosine; missing has a separate model indicator' if kind=='direction' else 'log scaled regional update L2')


def interval(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return [float(x) for x in np.quantile(values, [.025, .975])] if values else None


def cluster_indices(rows, key, rng):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[key]].append(i)
    ids = sorted(groups)
    return [i for cid in rng.choice(ids, len(ids), replace=True).tolist() for i in groups[cid]]


def bootstrap(rows, cfg, score, base_errors=None, changed_errors=None):
    result = {}
    for key in ('sender', 'receiver'):
        rng = np.random.default_rng(cfg['analysis_seed'] + (0 if key=='sender' else 1))
        aa = []; gains = []; correlations = []
        for _ in range(cfg['bootstrap_replicates']):
            ids = cluster_indices(rows, key, rng)
            sub = [rows[i] for i in ids]
            aa.append(auc([r['evidence'][score] for r in sub], [r['delta_ce']>0 for r in sub]))
            correlations.append(correlation([r['evidence'][score] for r in sub], [r['delta_ce'] for r in sub], True))
            if base_errors is not None:
                b = float(np.mean(base_errors[ids])); c = float(np.mean(changed_errors[ids]))
                gains.append((b-c)/b if b>0 else None)
        result[key] = dict(auc_ci95=interval(aa), spearman_ci95=interval(correlations), relative_mse_gain_ci95=interval(gains))
    return result


def description(rows, score=None):
    result = dict(pairs=len(rows), senders=len({r['sender'] for r in rows}), receivers=len({r['receiver'] for r in rows}),
        harmful=sum(r['delta_ce']>0 for r in rows), beneficial=sum(r['delta_ce']<0 for r in rows),
        zero=sum(r['delta_ce']==0 for r in rows), negative_loo=sum(r['negative_loo'] for r in rows),
        stages=dict(Counter(r['stage'] for r in rows)), profiles=dict(Counter(r['sender_profile'] for r in rows)))
    if rows and score:
        xx = [r['evidence'][score] for r in rows]; yy = [r['delta_ce'] for r in rows]
        result.update(auc=auc(xx, [y>0 for y in yy]), pearson=correlation(xx, yy), spearman=correlation(xx, yy, True))
    return result


def enough(rows, cfg):
    d = description(rows)
    return (d['pairs'] >= cfg['minimum_pairs_per_split'] and d['senders'] >= cfg['minimum_senders_per_split']
        and d['receivers'] >= cfg['minimum_receivers_per_split'] and min(d['harmful'], d['beneficial']) >= cfg['minimum_sign_labels_per_split'])


def shuffled(rows, score, rng):
    result = [dict(r, evidence=dict(r['evidence'])) for r in rows]
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['stage'], r['sender_profile'], r['sender_tier'],
            len(r['evidence']['valid_embedding_classes']), r['evidence']['valid_task_classes']].append(i)
    changed = 0
    for ids in groups.values():
        donors = rng.permutation(ids)
        for i, donor in zip(ids, donors):
            result[i]['evidence'][score] = rows[int(donor)]['evidence'][score]
            changed += result[i]['evidence'][score] != rows[i]['evidence'][score]
    return result, changed


def score_result(rows, cfg, score):
    valid = [r for r in rows if r['evidence'][score] is not None]
    dev = [r for r in valid if r['split']=='development']
    confirm = [r for r in valid if r['split']=='confirmation']
    result = dict(development=description(dev, score), confirmation=description(confirm, score),
                  missing=len(rows)-len(valid), development_cv=grouped_cv(dev, score))
    if not dev or not confirm or len(dev) < 30:
        result.update(go=False, reason='insufficient_disjoint_development_or_confirmation')
        return result
    yy = np.array([r['delta_ce'] for r in confirm])
    base = predict(dev, confirm); actual = predict(dev, confirm, score)
    current = predict(dev, confirm, current=True)
    with_memory = predict(dev, confirm, score, current=True)
    with_history = predict(dev, confirm, score, current=True, history=True)
    b = (base-yy)**2; c = (actual-yy)**2
    def gain(a, b):
        den = float(np.mean(a))
        return (den-float(np.mean(b)))/den if den>0 else None
    result.update(control_mse=float(b.mean()), augmented_mse=float(c.mean()), relative_mse_gain=gain(b,c),
        direction_only=simple_control(dev, confirm, 'direction'), magnitude_only=simple_control(dev, confirm, 'magnitude'),
        memory_increment_gain=gain((current-yy)**2, (with_memory-yy)**2),
        reliability_increment_gain=gain((with_memory-yy)**2, (with_history-yy)**2),
        bootstrap=bootstrap(confirm, cfg, score, b, c),
        memory_bootstrap=bootstrap(confirm, cfg, score, (current-yy)**2, (with_memory-yy)**2),
        reliability_bootstrap=bootstrap(confirm, cfg, score, (with_memory-yy)**2, (with_history-yy)**2),
        stages={s: description([r for r in confirm if r['stage']==s], score) for s in ('middle','late')})
    rng = np.random.default_rng(cfg['analysis_seed']); permutations = []; changed = []
    # Fixed 100 descriptive permutations; no rule or threshold is selected by them.
    for _ in range(100):
        dd, a = shuffled(dev, score, rng); cc, bcount = shuffled(confirm, score, rng)
        pred = predict(dd, cc, score)
        permutations.append(gain(b, (pred-yy)**2)); changed.append(a+bcount)
    result['permutation'] = dict(replicates=100, mean_changed_scores=float(np.mean(changed)),
        relative_mse_gain_ci95=interval(permutations), true_gain_greater_fraction=float(np.mean([
            v < result['relative_mse_gain'] for v in permutations if v is not None])))
    def positive_ci(boot, field):
        return all(boot[k][field] is not None and boot[k][field][0]>0 for k in ('sender','receiver'))
    checks = dict(sufficient_disjoint_sample=enough(dev, cfg) and enough(confirm,cfg),
        auc=result['confirmation']['auc'] is not None and result['confirmation']['auc']>=cfg['minimum_auc'],
        auc_intervals=all(result['bootstrap'][k]['auc_ci95'] is not None and result['bootstrap'][k]['auc_ci95'][0]>.5 for k in ('sender','receiver')),
        incremental_error=result['relative_mse_gain'] is not None and result['relative_mse_gain']>=cfg['minimum_incremental_relative_mse_gain'] and positive_ci(result['bootstrap'],'relative_mse_gain_ci95'),
        stages_positive=all(v['spearman'] is not None and v['spearman']>0 for v in result['stages'].values()),
        permutation=result['permutation']['mean_changed_scores']>0 and result['permutation']['true_gain_greater_fraction']>=.95,
        semantic_memory=positive_ci(result['memory_bootstrap'],'relative_mse_gain_ci95'),
        reliability_memory=positive_ci(result['reliability_bootstrap'],'relative_mse_gain_ci95'))
    result['checks'] = checks; result['go'] = all(checks.values())
    return result


def analyze(path, output):
    manifest = json.loads((path/'observation_manifest.json').read_text()); cfg = manifest['config']
    complete = json.loads((path/'completed.json').read_text())
    rows = [json.loads(s) for s in (path/'utility_pairs.jsonl').read_text().splitlines()]
    observation = [json.loads(s) for s in (path/'observation_rounds.jsonl').read_text().splitlines()]
    by_head = {seed: [r for r in rows if r['head_seed']==seed] for seed in cfg['head_seeds']}
    primary = by_head[cfg['head_seeds'][0]]
    results = {score: score_result(primary, cfg, score) for score in ('S0','S1','S2')}
    eligible = [(v['development_cv']['mse'], i, k) for i,(k,v) in enumerate(results.items()) if v['development_cv']['mse'] is not None]
    selected = min(eligible)[2] if eligible else None
    sensitivity = {str(seed): score_result(r, cfg, selected) for seed, r in by_head.items()} if selected else {}
    comparability = {}
    for seed in cfg['head_seeds']:
        values = [r['comparability'][str(seed)] for r in observation]
        gaps = [v['same_class_cosine']-v['different_class_cosine'] for v in values if v['same_class_cosine'] is not None and v['different_class_cosine'] is not None]
        accuracies = [v['prototype_classification_accuracy'] for v in values if v['prototype_classification_accuracy'] is not None]
        comparability[str(seed)] = dict(mean_same_minus_different_cosine=float(np.mean(gaps)) if gaps else None,
            mean_classification_accuracy=float(np.mean(accuracies)) if accuracies else None)
    sensitive_go = all(v['go'] for v in sensitivity.values()) if selected in ('S0','S1') else True
    space_valid = all(v['mean_same_minus_different_cosine'] is not None and v['mean_same_minus_different_cosine']>0
        and v['mean_classification_accuracy']>1/100 for v in comparability.values())
    full_budget = complete['success'] and complete['rounds_completed']==400
    go = bool(full_budget and selected and results[selected]['go'] and sensitive_go and space_valid)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='Go' if go else 'No-Go' if full_budget else 'Pilot only: no scientific Go',
        rounds=complete['rounds_completed'], source_commit=manifest['commit'], config_sha256=cfg['config_sha256'],
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), analysis_command=sys.argv,
        scoring_input='training-side paired CE only; official-test CSV is not read by this tool',
        selected_by_development=selected, actual_pairs=description(primary), scores=results,
        head_sensitivity=sensitivity, comparability= comparability,
        adapter_space_minimal_checks=space_valid, memory_roles='Increment tests above; support is not measured correctness',
        pair_dependence='Each pair once per head; sender and receiver cluster bootstrap separately; repeated classes/regions are dependent',
        overhead=dict(backbone_forward_examples=sum(c['backbone_forward_examples'] for r in observation for c in r['clients'] if c),
            adapter_backward_examples=sum(c['adapter_backward_examples'] for r in observation for c in r['clients'] if c),
            logical_upload_bytes=sum(c['logical_upload_bytes'] for r in observation for c in r['clients'] if c),
            measurement_seconds=sum(c['pre_measurement_seconds']+c['post_measurement_seconds'] for r in observation for c in r['clients'] if c),
            final_adapter_storage_bytes=observation[-1]['adapter_storage_bytes']),
        formal_gate_authorized=go, formal_training_started=False)
    (output/'mechanism_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    text = ['# A07 observational utility evidence', '', f'Status: **{result["status"]}**. Actual rounds: {result["rounds"]}.',
        f'Source: `{manifest["commit"]}`. Train-side physical pairs: {len(primary)}. Negative LOO: {sum(r["negative_loo"] for r in primary)}.',
        '', '|Score|Dev pairs|Held-out pairs|Held-out AUC|Incremental relative MSE gain|Scientific checks pass|', '|---|---:|---:|---:|---:|---|']
    for name,v in results.items():
        text.append(f'|{name}|{v["development"]["pairs"]}|{v["confirmation"]["pairs"]}|{v["confirmation"].get("auc")}|{v.get("relative_mse_gain")}|{v["go"]}|')
    text += ['', f'Development-selected formula: {selected}. No measurement head is selected by test performance.',
        'Full cluster intervals, stage results, head sensitivity, memory increments and permutation activity are in mechanism_summary.json.',
        'A No-Go retains frozen A07. A pilot cannot authorize a formal Gate. No new formula or threshold is generated by this analysis.']
    (output/'report.md').write_text('\n'.join(text)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('run', type=Path); p.add_argument('--output', type=Path, required=True)
    opt = p.parse_args(); result=analyze(opt.run, opt.output)
    print(json.dumps(dict(status=result['status'], selected=result['selected_by_development'], pairs=result['actual_pairs']), indent=2))
