"""Short-run checks, no performance selection or training launch."""
import argparse
import json
import math
from pathlib import Path


def rows(path):return [json.loads(v) for v in path.read_text().splitlines()]


def check(root,rounds):
    paths={m:root/'a07'/f'{m}_{rounds}_seed1' for m in ('reference','neutral','active')}
    streams={m:rows(p/'regression.jsonl') for m,p in paths.items()}
    manifests={m:json.loads((p/'v2_manifest.json').read_text()) for m,p in paths.items()}
    for field in ('commit','source_sha256','config_sha256','partition_sha256','configuration','deterministic_execution'):
        assert all(v[field]==manifests['reference'][field] for v in manifests.values()),field
    for mode,path in paths.items():
        completed=json.loads((path/'completed.json').read_text())
        assert completed['success'] and completed['rounds_completed']==rounds
        assert [r['round'] for r in streams[mode]]==list(range(rounds+1))
        assert not list(path.glob('*.pt'))
    exact=streams['reference']==streams['neutral']
    if not exact:raise AssertionError('Measured lambda=0 trajectory differs from original A07')
    active=streams['active'];reference=streams['reference']
    assert active[:2]==reference[:2]  # initialization and first round, including per-slot SGD
    assert all((a['slots'],a['rng'])==(b['slots'],b['rng']) for a,b in zip(active[1:],reference[1:]))
    changes=rows(paths['active']/'semantic_v2_rounds.jsonl')
    neutral=rows(paths['neutral']/'semantic_v2_rounds.jsonl')
    assert all(r['fusion']['changed_parameters']==0 and r['fusion']['difference_from_a07_l2']==0 for r in neutral)
    assert all(math.isfinite(r['fusion']['difference_from_a07_l2']) for r in changes)
    for r in changes:
        for d in r['fusion']['regions']:
            if d['reason']=='semantic_source_blend':
                assert all(v>=0 and math.isfinite(v) for v in d['new_slot_shares'])
                assert math.isclose(sum(d['new_slot_shares']),1.,abs_tol=1e-12)
    return dict(rounds=rounds,source_commit=manifests['reference']['commit'],
        config_sha256=manifests['reference']['config_sha256'],partition_sha256=manifests['reference']['partition_sha256'],
        neutral_exact_all_states_rng_evaluations=True,active_first_round_exact=True,
        active_schedule_rng_equal=True,nonneutral_rounds=[r['round'] for r in changes if r['fusion']['changed_parameters']>0],
        valid_evidence_rounds=[r['round'] for r in changes if r['decision']['valid']],
        maximum_difference_from_same_input_a07_l2=max(r['fusion']['difference_from_a07_l2'] for r in changes),
        adapter_in_decision=False,short_correctness_only=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--rounds',type=int,choices=(2,20),required=True);p.add_argument('--output',type=Path,required=True)
    opt=p.parse_args();result=check(opt.root,opt.rounds);opt.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
