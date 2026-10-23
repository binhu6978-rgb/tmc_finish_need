"""Report repairs only: no local SGD or frozen semantic algorithm changes."""
import csv
import json
from pathlib import Path
from tools.report_semantic_pair500 import shares
from tools.analyze_semantic_a07_iv_e import incremental


def test_large_csv_fields_and_sparse_actual_round_mapping(tmp_path):
    rows=[]
    for rd in (1,250,500):
        (tmp_path/'regression.jsonl').open('a').write(json.dumps(dict(round=rd,slots=[[1,0],[40,1],[70,6]]))+'\n')
        region=dict(region_id=0,slots=[0,1,2],applied_slot_shares=[rd/1000,.2,.8-rd/1000])
        rows.append(dict(round=rd,metrics_json=json.dumps(dict(server=dict(a07_baseline=dict(regions=[region])),
            long_realistic_diagnostics='x'*150000))))
    with (tmp_path/'method_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','metrics_json']);w.writeheader();w.writerows(rows)
    records=[dict(round=r,regions={'regions':[]},counterfactual_fusion={'permuted':{'regions':[]}}) for r in (1,500)]
    result=shares(records,tmp_path)
    assert len(result)==12
    assert next(r for r in result if r['round']==500 and r['tier']==0)['pi_a07']==.5
    assert all(r['delta_pi']==0 for r in result)


def test_explicit_fold_assignment_keeps_existing_default_results():
    rows=[dict(round=rd,source_tier=i%3,receiver=i,source_delta_l2=1+i/10,direction_cosine=.1,
        receiver_tier=i%3,receiver_profile=i%7,source_mass=10,region_numel=20,source_receiver_label_js=.1,
        source_profile_mean=2,source_own_gain=.1,before_ce=1.,common_utility=.01*i,
        D=.02*i,M=.01*i,U=.005*i,permuted_U=.005*(12-i),region=0) for i,rd in enumerate([1]*4+[2]*4+[3]*4)]
    default=incremental(rows,'round');explicit=incremental(rows,'round',lambda r:r['round'])
    assert default==explicit
    stage=incremental(rows,'stage',lambda r:r['round'])
    assert stage['models']==default['models'] and stage['rounds']==3
