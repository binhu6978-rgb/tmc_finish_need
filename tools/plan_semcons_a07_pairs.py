"""Outcome-blind opportunity counts from archived identities and real geometry."""
import argparse
from collections import defaultdict, Counter
import csv
import hashlib
import json
from pathlib import Path


def seed(*items):
    return int.from_bytes(hashlib.sha256(repr(items).encode()).digest()[:8], 'little') % (2**32)


def opportunities(root, cfg):
    regions = json.loads((root/'manifest.json').read_text())['structural_regions']
    groups = defaultdict(list)
    with (root/'method_diagnostics.csv').open() as f:
        for record in csv.DictReader(f):
            r = int(record['round'])
            stage = next((s for s, v in cfg['diagnostic_rounds'].items() if r in v), None)
            if not stage:
                continue
            # No Accuracy CSV, CE, directions, weights or semantic outcomes are consulted.
            slots = json.loads(record['metrics_json'])['slots']
            for region in regions:
                clients = sorted({cid for cid, p in slots if p in region['profiles']})
                if len(clients) < 2:
                    continue
                senders = sorted(clients, key=lambda c: seed('sender', r, region['id'], c))[:cfg['senders_per_region']]
                for cid in senders:
                    receivers = sorted([c for c in clients if c != cid], key=lambda c: seed('receiver', r, region['id'], cid, c))[:cfg['receivers_per_sender_region']]
                    for receiver in receivers:
                        split = ('development' if stage=='early' and cid<70 and receiver<70 else
                            'confirmation' if stage in ('middle','late') and cid>=70 and receiver>=70 else 'descriptive')
                        groups[split].append((cid, receiver, r, region['id']))
    return {s:dict(pairs=len(v), senders=len({x[0] for x in v}), receivers=len({x[1] for x in v}),
                   per_round=dict(Counter(x[2] for x in v))) for s, v in groups.items()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('archive', type=Path); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    opt = p.parse_args(); result=opportunities(opt.archive, json.loads(opt.config.read_text()))
    opt.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
