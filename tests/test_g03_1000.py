import csv
from pathlib import Path

import pytest

from tools.run_g03_1000 import fixed_args,read_rows,summarize


def test_single_fixed_original_g03_protocol_no_model_saving(tmp_path):
    args=fixed_args(tmp_path/'new')
    assert (args.method,args.epochs,args.seed,args.gpu,args.generate_data,args.eval_profiles)==('g03',1000,1,0,0,'full')
    assert args.no_save_pt and args.deterministic_execution and not args.method_resume
    assert args.num_users==100 and args.local_ep==5 and args.local_bs==50
    assert args.lr==.01 and args.lr_decay==.998 and args.momentum==.5 and args.weight_decay==.0001
    assert not (tmp_path/'new').exists()


def test_actual_historical_g03_summary_and_budget_check(tmp_path):
    source=Path(__file__).resolve().parents[1]/'docs/evidence/native_groups10_results/raw/G03/accuracy_each_round.csv'
    rows=read_rows(source,400);r=summarize('old',rows)
    assert r['test_oracle_peak']==pytest.approx(38.45,abs=1e-5) and r['best_round']==339
    assert r['last20_mean']==pytest.approx(37.6585,abs=1e-5)
    with pytest.raises(ValueError,match='Incomplete round budget'):read_rows(source,1000)
    bad=tmp_path/'bad.csv'
    with bad.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();rows[0]['full_ce']='nan';writer.writerows(rows)
    with pytest.raises(ValueError,match='Nonfinite'):read_rows(bad,400)
