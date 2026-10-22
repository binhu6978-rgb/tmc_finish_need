"""Checks on archived *real* complete GPU regressions, not synthetic efficacy."""
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from tools.check_semcons_a07_equivalence import compare

EVIDENCE = Path(__file__).resolve().parents[1]/'docs/evidence/semcons_a07/regression_v1/semcons_a07_regression_v1_20261022'


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('rounds', [2, 20])
def test_archived_real_gpu_a07_observation_is_exact(rounds, version):
    evidence = EVIDENCE if version==1 else EVIDENCE.parents[1]/'regression_v2/semcons_a07_regression_v2_20261022'
    reference = evidence/'a07'/f'reference_{rounds}_seed1'
    observed = evidence/'a07'/f'observe_{rounds}_seed1'
    proof = compare(reference, observed)
    assert proof['exact_match'] and proof['round2_exact'] and proof['rounds']==rounds
    rows = [json.loads(s) for s in (observed/'regression.jsonl').read_text().splitlines()]
    assert all(len(r['models'])==7 for r in rows)
    assert all(len(r['slots'])==10 and len(r['local_checks'])==10 for r in rows[1:])
    assert not list(observed.rglob('*.pt'))


def test_untouched_frozen_a07_files_match_actual_training_proof():
    proof = json.loads((EVIDENCE/'equivalence_20.json').read_text())
    for name in ('capability_final40.py','native_group_aggregation.py','models/Update.py','models/Fed.py',
                 'models/resnet.py','Algorithm/Training_AdaptiveFL.py'):
        frozen = subprocess.check_output(['git','show','24e79dbb26ce718b533e7bc2c440a91c628caa25:'+name])
        assert hashlib.sha256(frozen).hexdigest()==proof['frozen_source_sha256'][name]
        assert Path(name).read_bytes()==frozen
