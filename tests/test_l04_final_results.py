"""Check the final scalar report against the actual completed evidence."""
import json
from pathlib import Path

import pytest

from tools.analyze_l04_results import analyze


def test_real_five_run_evidence_and_classifier_controls(tmp_path):
    root=Path(__file__).resolve().parents[1]/'docs/evidence/e03_l04_final_results'
    result=analyze(root,tmp_path/'analysis')
    rows={r['id']:r for r in result['comparison']}
    assert rows['B00-D']['test_oracle_peak']==pytest.approx(36.68,abs=1e-5)
    assert rows['B00-D']['last20']>max(rows[k]['last20'] for k in ('R0','R1','R2','R3'))
    assert rows['R2']['final']>rows['B00-D']['final']  # Preserve the mixed result.
    assert rows['R3']['final_ce']<rows['B00-D']['final_ce']
    mechanisms=result['mechanisms']
    assert mechanisms['B00-D']['negative_slots'] is None  # No invented measurements.
    assert mechanisms['R3']['actual_changed_from_same_input_m2_rounds']==1
    assert mechanisms['R3']['later_difference_from_same_input_m2_l2']==0
    assert [mechanisms[k]['classifier_active_rounds'] for k in ('R0','R1','R2','R3')]==[0,400,399,1]
    assert mechanisms['R1']['first_difference_from_same_input_m2_l2']==mechanisms['R3']['first_difference_from_same_input_m2_l2']
    written=json.loads((tmp_path/'analysis/analysis.json').read_text())
    assert all(written['verification'].values())
