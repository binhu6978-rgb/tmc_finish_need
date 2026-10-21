"""Regressions against completed scalar evidence, including negative findings."""
import json
from pathlib import Path
import shutil

import pytest

from tools.analyze_g03_completed import analyze, final_check


EVIDENCE = Path(__file__).resolve().parents[1] / 'docs/evidence/g03_completed_results'


def test_completed_fixed_comparisons_and_scale_controls(tmp_path):
    result = analyze(EVIDENCE, tmp_path / 'analysis', draw=False)
    by_id = {r['id']: r for r in result['comparison']}
    assert len(by_id) == 10
    assert by_id['P05']['test_oracle_peak'] == pytest.approx(38.64, abs=1e-5)
    assert by_id['P04']['last20_mean'] == pytest.approx(37.948, abs=1e-5)
    for name in ('P04', 'P05'):
        assert all(by_id[name][k + '_minus_G03'] > 0 for k in
                   ('test_oracle_peak', 'last20_mean', 'last50_mean'))
    assert by_id['P05']['final_accuracy_minus_G03'] < 0
    for name in ('P01', 'P02', 'P03', 'P06', 'P07', 'P08', 'P09', 'P10'):
        assert by_id[name]['test_oracle_peak_minus_G03'] < 0
        assert by_id[name]['last20_mean_minus_G03'] < 0
    m = result['mechanisms']
    assert m['P01']['later_actual_update_cosine_native_mean'] > .99999999
    assert m['P02']['later_actual_update_cosine_g03_mean'] > .99999999
    for name in ('P01', 'P02'):
        assert m[name]['matched_norm_relative_error_max'] < 1e-7
    assert m['P02']['later_update_l2_ratio_native_mean'] == pytest.approx(1, abs=1e-7)
    assert all(m[n]['regions'][8]['actually_changed_rounds'] == 0 for n in by_id if n not in ('P01', 'P02'))
    assert result['verification']['g03_1000_first400_accuracy_and_ce_exactly_equal_historical']
    long = result['g03_long'][-1]
    assert long['rounds'] == 1000 and long['best_round'] == 774
    assert long['test_oracle_peak'] == pytest.approx(38.58, abs=1e-5)
    assert long['final_ce'] > result['reference']['final_ce']


@pytest.mark.parametrize('defect', ['nonfinite_final', 'missing_round'])
def test_completion_rejects_invalid_final_or_truncated_evidence(tmp_path, defect):
    source = EVIDENCE / 'g1000/g03/G03_full_only_1000_seed1_deterministic'
    for name in ('completed.json', 'manifest.json', 'final_model_check.json', 'accuracy_each_round.csv'):
        shutil.copyfile(source / name, tmp_path / name)
    if defect == 'nonfinite_final':
        path = tmp_path / 'final_model_check.json'
        value = json.loads(path.read_text())
        value['finite_full_parameters'] = False
        path.write_text(json.dumps(value))
        message = 'Final in-memory'
    else:
        path = tmp_path / 'accuracy_each_round.csv'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        message = 'Incomplete actual round budget'
    with pytest.raises(ValueError, match=message):
        final_check(tmp_path, 1000)
