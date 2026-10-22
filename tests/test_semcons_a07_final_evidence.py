"""Real completed mechanism evidence integrity and independently checkable results."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pytest
from tools.analyze_semcons_a07 import auc

ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'docs/evidence/semcons_a07/final'


@pytest.fixture(scope='module')
def data():
    with gzip.open(FINAL/'utility_pairs.jsonl.gz','rt',encoding='utf-8') as f:
        return [json.loads(v) for v in f]


def test_lossless_scalar_archive_index():
    for item in json.loads((FINAL/'raw_scalar_index.json').read_text()):
        compressed=(FINAL/item['file']).read_bytes();raw=gzip.decompress(compressed)
        assert hashlib.sha256(raw).hexdigest()==item['raw_sha256']
        assert hashlib.sha256(compressed).hexdigest()==item['gzip_sha256']
        assert len(raw)==item['raw_bytes']


def test_real_pairs_not_heads_as_independent_replicates(data):
    assert len(data)==11988 and len({r['pair_id'] for r in data})==3996
    primary=[r for r in data if r['head_seed']==17]
    assert all(r['sender']!=r['receiver'] for r in primary)
    assert sum(r['negative_loo'] for r in primary)==226
    assert sum(r['delta_ce']>0 for r in primary)==1441
    assert sum(r['delta_ce']<0 for r in primary)==2555


def test_recompute_heldout_auc_and_disjoint_physical_ids(data):
    primary=[r for r in data if r['head_seed']==17]
    dev=[r for r in primary if r['split']=='development' and r['evidence']['S0'] is not None]
    confirm=[r for r in primary if r['split']=='confirmation']
    assert len(dev)==140 and len(confirm)==652
    assert not ({r['sender'] for r in dev}|{r['receiver'] for r in dev}) & ({r['sender'] for r in confirm}|{r['receiver'] for r in confirm})
    summary=json.loads((FINAL/'server_analysis/mechanism_summary.json').read_text())
    for name in ('S0','S1','S2'):
        actual=auc([r['evidence'][name] for r in confirm],[r['delta_ce']>0 for r in confirm])
        assert actual==pytest.approx(summary['scores'][name]['confirmation']['auc'],abs=1e-14)
    assert summary['selected_by_development']=='S0' and summary['status']=='No-Go'


def test_full400_trace_matches_frozen_a07_and_gate_is_locked():
    historical=ROOT/'docs/evidence/final40_results/gpu0/a07/A07_full_only_400_seed1/accuracy_each_round.csv'
    with historical.open() as f:old=list(csv.DictReader(f))
    with (FINAL/'accuracy_each_round.csv').open() as f:current=list(csv.DictReader(f))
    assert len(current)==400
    assert [int(r['round']) for r in current]==list(range(1,401))
    assert all(float(a[k])==float(b[k]) for a,b in zip(old,current) for k in ('full_accuracy','full_ce'))
    audit=json.loads((FINAL/'audit_summary.json').read_text())
    assert audit['nonneutral_gates']==0 and audit['no_pt'] and audit['all400_schedule_identical']
    assert audit['final_anchor_classes']=={'task':100,'17':100,'29':100,'43':100}


def test_unchanged_preregistration_and_full_local_recomputation():
    manifest=json.loads((FINAL/'observation_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'configs/semcons_a07_observation_v2.json').read_bytes()).hexdigest()==manifest['config']['config_sha256']
    assert manifest['config']['minimum_auc']==.6 and manifest['config']['minimum_incremental_relative_mse_gain']==.02
    check=json.loads((FINAL/'reproduction_check.json').read_text())
    assert check['decision_identical'] and check['selected_identical']
    assert check['numeric_fields_compared']==779 and check['max_absolute_difference']<1e-10
