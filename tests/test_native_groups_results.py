"""Regression checks for the completed, immutable G01-G10 evidence report."""
from pathlib import Path

import pytest

from tools import analyze_native_groups_results as analysis


EVIDENCE=Path(__file__).resolve().parents[1]/'docs/evidence/native_groups10_results'


def test_actual_results_preserve_mixed_and_negative_findings(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'plots',lambda *args:None)
    result=analysis.analyze(EVIDENCE,tmp_path/'analysis')
    rows={r['id']:r for r in result['comparison']}
    assert len(rows)==10 and all(r['rounds']==400 for r in rows.values())
    assert rows['G03']['test_oracle_peak']==pytest.approx(38.45,abs=1e-5)
    assert rows['G03']['last20_mean_minus_B00_D']==pytest.approx(1.3395,abs=1e-5)
    for name in ('G03','G05'):
        assert all(rows[name][k+'_minus_B00_D']>0 for k in
            ('test_oracle_peak','final_accuracy','last20_mean','last50_mean'))
    assert rows['G02']['test_oracle_peak_minus_B00_D']>0
    assert rows['G02']['last20_mean_minus_B00_D']<0
    assert result['contrasts']['G09-G10']['last20_mean']<-4.9
    m=result['mechanisms']
    assert m['G03']['full_only_directly_changed_rounds']==0
    assert m['G05']['full_only_directly_changed_rounds']==0
    assert m['G02']['changed_rounds']==164
    assert m['G07']['coverage_unique_physical_totals']==m['G08']['coverage_unique_physical_totals']
    assert m['G03']['shared_tier_applied_shares'][0]>m['G03']['shared_tier_native_shares'][0]
    assert m['G03']['later_geometry']['perpendicular_fraction']>.9
    assert m['G08']['later_geometry']['perpendicular_fraction']<1e-6
    assert m['G10']['worst_scale_match']['round']==127
    assert m['G10']['worst_scale_match']['scale_norm_absolute_error']<1.2e-8
    assert result['communication']['each_run_uplink_float32_state_bytes']==88387695816


@pytest.mark.parametrize('actual,native,difference,expected',[
    (2.,1.,1.,0.),       # Only changes magnitude: (1,0) -> (2,0).
    (2**.5,1.,1.,1.),   # Orthogonal change: (1,0) -> (1,1).
])
def test_displacement_energy_is_not_confused_with_total_energy(actual,native,difference,expected):
    result=analysis.geometry_change([dict(actual_update_l2=actual,native_update_l2=native,
        difference_from_native_l2=difference)])
    assert result['perpendicular_fraction']==pytest.approx(expected,abs=1e-14)
