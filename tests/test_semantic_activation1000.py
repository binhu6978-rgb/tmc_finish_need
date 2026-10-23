import copy
import json
from pathlib import Path
import pytest
from tools.report_semantic_activation import extract
from tools.run_semantic_activation1000 import verify_smoke
from semcons_a07_v2.pair500 import sampled_rounds


def test_fixed_single1000_budget_and_unchanged_lambda():
    p=json.loads(Path('configs/semantic_a07_activation1000.json').read_text())
    assert p['rounds']==1000 and p['mode']=='active' and p['semantic_lambda']==.2
    assert len(sampled_rounds(p))==53 and sampled_rounds(p)[-1]==1000
    assert p['stages']=={'early':[8,24],'middle':[492,508],'late':[984,1000]}
    assert not p['score_selection'] and not p['new_score_candidate']


def records():
    rows=[]
    for rd in range(1,5):
        valid=rd>1;detail=dict(tier=0,category=1,q_pre=.2,q_post=.5,reference_q=.4,reference_round=rd-1,
            d=.3,a=.1,rho=.5,u=.05,own_reliability={'age':1},reference_reliability={'age':1})
        decision=dict(valid=valid,classes=[1] if valid else [],scores={0:.05,1:0.},details=[detail] if valid else [])
        # Both tiers require their own category observation; add real matching score0.
        if valid:decision['details'].append(dict(detail,tier=1,q_pre=.5,q_post=.5,reference_q=.6,d=0.,a=0.,u=0.))
        pi={0:.5,1:.5};denom=1+.025;new={g:.8*.5+.2*.5*(1+decision['scores'][g])/denom for g in pi}
        region=dict(region=0,pi_a07=pi,pi_new=new,feature_numel=10,reason='semantic_source_blend')
        rows.append(dict(round=rd,decision=decision,fusion=dict(reason='source_blend' if valid else 'first_round_native',
            changed_parameters=10 if valid else 0,difference_from_a07_l2=.001 if valid else 0,regions=[region] if valid else []),
            tier_anchor_count=2, tier_history_records=4))
    return rows


def test_activation_extraction_uses_all_rounds_and_never_outcomes():
    raw=records();saved=copy.deepcopy(raw)
    p=dict(stages={'early':[2,4]},semantic_lambda=.2,illustrative_case_rule='first active')
    a=extract(raw,[dict(id=0,profiles=list(range(7)),numel=10)],p)
    assert raw==saved and len(a[0])==4 and a[-1]['active_rounds']==3
    assert {r['round'] for r in a[4]}=={2}
    for r in raw:r['accuracy']=999;r['utility']=-999
    assert extract(raw,[dict(id=0,profiles=list(range(7)),numel=10)],p)==a


def test_missing_qualified_example_is_missing_not_fabricated_zero():
    p=dict(stages={'initial':[1,1]},semantic_lambda=.2,illustrative_case_rule='first active')
    data=extract(records(),[dict(id=0,profiles=list(range(7)),numel=10)],p)
    assert data[4][0]['round'] is None and not data[0][0]['active']


def test_real_smoke_comparison_rejects_any_changed_trace(tmp_path):
    a=tmp_path/'new';b=tmp_path/'old'
    rows=[{'round':r,'models':['actual_hash'],'rng':'same'} for r in range(3)]
    a.write_text('\n'.join(map(json.dumps,rows)));b.write_text(a.read_text())
    assert verify_smoke(a,b)['success']
    rows[2]['rng']='changed';b.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(AssertionError):verify_smoke(a,b)
