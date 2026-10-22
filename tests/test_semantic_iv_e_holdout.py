import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from adapterfl_methods import rng_state
from semcons_a07.measurement import probe_indices
from semcons_a07_v2.iv_e_holdout import split_users,cohort,HeldoutProbes,HeldoutDiagnostic
from semcons_a07_v2.aggregation import fuse
from test_semantic_a07_iv_e import TrainData
from test_semantic_a07_v2 import decision,config
from test_final40 import setup
from test_adapterfl_methods import assert_states,assert_rng
from models.Fed import split_model


def test_global_split_disjoint_reproducible_and_score_probes_training_only():
    data=TrainData();original={c:list(range(c*12,(c+1)*12)) for c in range(100)};saved=copy.deepcopy(original)
    training,evaluation=split_users(data,original)
    assert original==saved and (training,evaluation)==split_users(data,original)
    a={i for values in training.values() for i in values};b={i for values in evaluation.values() for i in values}
    assert not a&b and a|b==set(range(1200))
    for c in original:
        assert not set(probe_indices(data,training[c],c,2))&set(evaluation[c])
        assert len(training[c])==8 and len(evaluation[c])==4


def test_rare_class_kept_training_without_fabricated_validation():
    class Rare(TrainData):targets=[0,0,0,1,1,1,1]
    train,held=split_users(Rare(),{0:list(range(7))})
    assert {0,1,2}<=set(train[0]) and len(held[0])==2


def test_heldout_probe_has_no_training_indices_and_rng_is_unchanged():
    data=TrainData();original={c:list(range(c*12,(c+1)*12)) for c in range(100)};a,b=split_users(data,original)
    before=rng_state();x=HeldoutProbes(data,a,b).batches(1);y=HeldoutProbes(data,a,b).batches(1)
    assert all(torch.equal(p,q) and torch.equal(l,m) for (p,l),(q,m) in zip(x,y));assert_rng(before,rng_state())
    assert all(cohort(c)==cohort(c) for c in original) and {cohort(c) for c in original}=={'development','confirmation'}


def test_real_interface_counterfactuals_preserve_models_and_separate_own_gain(tmp_path):
    torch.set_num_threads(2);h,u,n,w,models=setup('a07')
    h.aggregate(u,n,w);base=h.aggregate(u,n,w);d=decision();actual,report=fuse(base,w,u,n,h,d,.2,False)
    original={c:list(range(c*12,(c+1)*12)) for c in range(100)};train,held=split_users(TrainData(),original)
    diag=HeldoutDiagnostic(TrainData(),train,held,tmp_path);diag.models=models
    sent=[split_model(w,models[p].state_dict()) for c,p in h.slots]
    diag.observer=SimpleNamespace(dispatches=sent,round_number=2,cfg=config(),fusion=report,
        observations=[dict(own_gain=.1) for _ in u])
    saved=copy.deepcopy(actual);rng=rng_state();diag.run(base,actual,w,u,n,h,d,False)
    assert_states(actual,saved);assert_rng(rng,rng_state())
    record=json.loads((tmp_path/'iv_e_rounds.jsonl').read_text())
    assert len(record['figure8'])==36
    assert {r['cohort'] for r in record['figure8']}=={'development','confirmation'}
    assert all(r['source_own_holdout_gain'] is not None and math.isclose(r['source_own_training_gain'],.1) for r in record['figure7'])


def test_fixed_budget_stage_windows_and_no_score_candidate():
    cfg=json.loads(Path('configs/semantic_a07_iv_e_expanded.json').read_text())
    rounds={r for a,b in cfg['stages'].values() for r in range(a,b+1)}
    assert len(rounds)==51 and min(rounds)==8 and max(rounds)==120
    assert cfg['budget_rounds']==120 and not cfg['formal_400_allowed'] and not cfg['new_score_candidate']
