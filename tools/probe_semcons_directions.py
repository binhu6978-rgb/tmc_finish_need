"""Three full-data rounds, unchanged S07 decisions; observe tensor directions only."""
import copy
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapterfl_methods import isolated_evaluation_rng
from semcons_scope import ScopeHooks
import tools.preflight_semcons_activation as preflight


def signed_cos(dot, aa, bb):
    return max(-1., min(1., dot / (aa * bb) ** .5)) if min(aa, bb) > 1e-24 else None


def references(hook, deltas, lens):
    groups = {}
    for i in deltas:
        groups.setdefault(hook.slots[i][0], []).append(i)
    pooled = {c: sum(deltas[i] * lens[i] for i in ids) / sum(lens[i] for i in ids) for c, ids in groups.items()}
    weights = {c: lens[ids[0]] for c, ids in groups.items()}
    return {i: sum(pooled[c] * weights[c] for c in groups if c != hook.slots[i][0]) /
            sum(weights[c] for c in groups if c != hook.slots[i][0]) for i in deltas}


@torch.no_grad()
def tensor_observation(hook, uploads, lens, dispatched):
    records = []
    for region in hook.regions:
        members = [i for i, (_, p) in enumerate(hook.slots) if p in region['profiles']]
        if len({hook.slots[i][0] for i in members}) < 2:
            continue
        values = hook.region_deltas(region, uploads, dispatched, members)
        refs = references(hook, values, lens)
        for i in members:
            a = values[i]; b = refs[i]
            dot = float(torch.dot(a, b)); aa = float(a.square().sum()); bb = float(b.square().sum())
            cells = []; offset = 0
            for cell in region['cells']:
                n = cell['numel']; x = a[offset:offset+n]; y = b[offset:offset+n]; offset += n
                ab = float(torch.dot(x, y)); xx = float(x.square().sum()); yy = float(y.square().sum())
                key = cell['key']; shape = uploads[i][key].shape
                family = 'classifier' if key.startswith('classifier.') else 'conv' if len(shape) == 4 else 'bn_weight' if key.endswith('weight') else 'bn_bias'
                cells.append({'key': key, 'bounds': cell['bounds'], 'numel': n, 'family': family,
                              'stage': '.'.join(key.split('.')[:2]) if key.startswith('features.') else 'classifier',
                              'dot': ab, 'delta_square': xx, 'reference_square': yy, 'cosine': signed_cos(ab, xx, yy)})
            if abs(sum(c['dot'] for c in cells)-dot) > 1e-10 * max(1., abs(dot)):
                raise AssertionError('Tensor dot products do not reconstruct region dot product')
            records.append({'region': region['id'], 'slot': i, 'client': hook.slots[i][0],
                            'profile': hook.slots[i][1], 'numel': region['numel'], 'cosine': signed_cos(dot, aa, bb),
                            'dot': dot, 'delta_square': aa, 'reference_square': bb, 'cells': cells})
    return records


def observed(original, output):
    def aggregate(hook, uploads, lens, previous, dispatched=None):
        # The existing S07 diagnostic also calls S09 on cloned buffers; observe S07 only.
        begin = time.perf_counter()
        records = None
        if hook.method == 's07':
            with isolated_evaluation_rng():
                records = tensor_observation(hook, uploads, lens, dispatched)
        result = original(hook, uploads, lens, previous, dispatched)
        if records is not None:
            with output.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'round': hook.round_index+1, 'slots': hook.slots, 'records': records,
                                    'diagnostic_and_aggregate_seconds': time.perf_counter()-begin})+'\n')
        return result
    return aggregate


def main():
    import argparse
    p = argparse.ArgumentParser(__doc__); p.add_argument('--output-root', required=True); opt = p.parse_args()
    root = Path(opt.output_root); root.mkdir(parents=True, exist_ok=False)
    original = ScopeHooks.aggregate
    ScopeHooks.aggregate = observed(original, root / 'tensor_directions.jsonl')
    saved = sys.argv
    try:
        sys.argv = ['preflight_semcons_activation.py', '--mode', 's07', '--rounds', '3', '--gpu', '0', '--output-root', str(root)]
        preflight.main()
    finally:
        ScopeHooks.aggregate = original; sys.argv = saved
    (root/'probe_source.json').write_text(json.dumps({'kind':'read-only tensor directions; not new aggregation',
        'wrapper_source_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
