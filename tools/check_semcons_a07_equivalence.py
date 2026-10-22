"""Compare real original-loop outputs without saving model tensors."""
import argparse
import json
from pathlib import Path


def compare(reference, observed):
    manifests = [json.loads((p/'observation_manifest.json').read_text()) for p in (reference, observed)]
    if manifests[0]['mode'] != 'reference' or manifests[1]['mode'] != 'observe':
        raise ValueError('Need original reference versus observational run')
    for key in ('config', 'partition_sha256', 'rounds', 'deterministic_execution', 'source_sha256', 'commit', 'torch'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'Paired run setup differs: {key}')
    streams = [[json.loads(s) for s in (p/'regression.jsonl').read_text().splitlines()] for p in (reference, observed)]
    rounds = manifests[0]['rounds']
    success = all(json.loads((p/'completed.json').read_text())['success'] for p in (reference, observed))
    mismatches = []
    for stream in streams:
        if [r['round'] for r in stream] != list(range(rounds+1)):
            success = False
    for a, b in zip(*streams):
        for key in set(a) | set(b):
            if a.get(key) != b.get(key):
                mismatches.append(dict(round=a['round'], field=key))
    return dict(exact_match=success and not mismatches, rounds=rounds,
        config_sha256=manifests[0]['config']['config_sha256'], commit=manifests[0]['commit'],
        source_sha256=manifests[0]['source_sha256'], partition_sha256=manifests[0]['partition_sha256'],
        frozen_source_sha256=json.loads((reference/'manifest.json').read_text())['identity']['source_sha256'],
        reference=str(reference), observed=str(observed), round2_exact=not any(r['round']<=2 for r in mismatches) and success,
        mismatches=mismatches, checks='all seven numerical state digests, per-slot upload and RNG, round RNG, schedule, Full CE/accuracy')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('reference', type=Path); p.add_argument('observed', type=Path)
    p.add_argument('--output', type=Path, required=True)
    opt = p.parse_args(); result = compare(opt.reference, opt.observed)
    opt.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['exact_match']:
        raise SystemExit(1)
