"""Actual fixed-budget evidence and mixed cross-dataset outcome regressions."""
import json
from pathlib import Path

import pytest

from tools.analyze_final40_results import analyze
from tools import report_final40


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/evidence/final40_results'


def test_actual_forty_completed_results_preserve_scope_and_negative_findings(tmp_path):
    result = analyze(EVIDENCE, tmp_path/'audit', draw=False)
    assert all(result['verification'].values())
    summary=json.loads((EVIDENCE/'report/summary.json').read_text(encoding='utf-8'))
    by_id={r['id']:r for r in summary['comparison']}
    assert len(by_id)==40 and all(r['rounds']==400 for r in by_id.values())
    assert by_id['A07']['test_oracle_peak']==pytest.approx(39.94,abs=1e-5)
    assert by_id['A07']['last20_mean']==pytest.approx(38.6235,abs=1e-5)
    assert by_id['A07']['final_ce']<summary['references']['P04']['final_ce']
    assert by_id['A10']['final_ce']>summary['references']['P04']['final_ce']
    assert by_id['S4-P04']['last20_mean_minus_native']==pytest.approx(2.5435,abs=1e-5)
    assert by_id['S5-P04']['test_oracle_peak_minus_native']>0
    assert by_id['S5-P04']['last20_mean_minus_native']<0
    assert by_id['S5-P04']['final_ce_minus_native']>0
    assert by_id['S5-P05']['last20_mean_minus_native']>0
    assert by_id['S5-P05']['final_accuracy_minus_native']<0
    assert by_id['S5-P05']['final_ce_minus_native']>0
    for name in ('A01','A02','A03','A04'):
        assert summary['mechanisms'][name]['matched_norm_relative_error_max']<1e-6
    assert result['same_input_p04_special']['A09']['actual_to_same_input_p04_norm_ratio_mean']==pytest.approx(1,abs=.0001)
    assert set(result['fixed_three_metric_candidates'])=={'A07','A09','A12','A15','A16','A17','A19','A20'}


def test_portable_offline_report_reproduces_server_tables(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    monkeypatch.setattr(Figure,'savefig',lambda *a,**kw:None)
    references={
        'B00-D':ROOT/'docs/evidence/native_groups10_results/baseline',
        'G03':ROOT/'docs/evidence/native_groups10_results/raw/G03',
        **{n:ROOT/'docs/evidence/g03_completed_results/p400'/n.lower()/(n+'_full_only_400_seed1_deterministic') for n in ('P04','P05')}
    }
    rows=report_final40.report(EVIDENCE,tmp_path/'recomputed',references)
    original=json.loads((EVIDENCE/'report/summary.json').read_text(encoding='utf-8'))
    assert rows==original['comparison']
