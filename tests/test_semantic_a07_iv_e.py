import copy
import json
from types import SimpleNamespace
import numpy as np
import torch
from adapterfl_methods import rng_state
from test_adapterfl_methods import assert_states,assert_rng
from test_final40 import setup
from test_semantic_a07_v2 import decision,config
from models.Fed import split_model
from semcons_a07.measurement import probe_indices
from semcons_a07_v2.aggregation import fuse
from semcons_a07_v2.iv_e_diagnostics import independent_indices,Probes,score_controls,stage_scores,ce_summary,l2_control,MechanismDiagnostic


class TrainData:
    targets=[y for cid in range(100) for y in [0,1]*6]
    def __getitem__(self,i):
        return torch.arange(48,dtype=torch.float32).reshape(3,4,4)/50+i/1000+torch.rand(3,4,4)*.01,self.targets[i]


def test_independent_disjoint_reproducible_probes_and_rng():
    data=TrainData();users={c:list(range(c*12,(c+1)*12)) for c in range(100)}
    selected=independent_indices(data,users[1],1)
    assert len(selected)==4 and not set(selected)&set(probe_indices(data,users[1],1,2))
    state=rng_state();a=Probes(data,users);b=Probes(data,users)
    assert all(torch.equal(x,y) and torch.equal(l,m) for (x,l),(y,m) in zip(a.batches(1),b.batches(1)))
    assert_rng(state,rng_state())


def test_insufficient_disjoint_support_is_missing():
    class Small(TrainData):targets=[0,0,1,1]
    assert independent_indices(Small(),range(4),1)==[]


def test_ablation_terms_and_permutation_do_not_change_validity_or_formula():
    d=decision();original=copy.deepcopy(d);v=score_controls(d)
    assert d==original
    assert sorted(v['permuted']['scores'].values())==sorted(d['scores'].values())
    for key in v:assert v[key]['classes']==d['classes'] and v[key]['valid']==d['valid']
    for g in d['scores']:
        detail=[x for x in d['details'] if x['tier']==g]
        assert v['no_sm']['scores'][g]==sum(x['rho']*x['d'] for x in detail)/len(detail)
        assert v['no_rm']['scores'][g]==sum(min(x['d'],x['a']) for x in detail)/len(detail)
        assert stage_scores(d,g)['U']==d['scores'][g]
    missing=dict(valid=False,reason='missing',scores={},classes=[],details=[])
    assert all(not x['valid'] for x in score_controls(missing).values())


def test_signed_ce_macro_common_and_missing_not_zero():
    x=ce_summary([2.,1.,3.],[1.,3.,2.],[0,1,0],{0})
    assert x['utility']==0 and x['common_utility']==1 and x['common_class_count']==1
    assert ce_summary([2.],[1.],[0],{1})['common_utility'] is None


def test_scale_control_uses_same_input_norm_without_input_mutation():
    h,u,n,w,_=setup('a07');h.aggregate(u,n,w);base=h.aggregate(u,n,w)
    out,_=fuse(base,w,u,n,h,decision(),.2,False);saved=copy.deepcopy(base)
    scaled,scale=l2_control(base,out,w,h)
    from native_group_aggregation import update_norm
    assert abs(update_norm(scaled,w)-update_norm(out,w))<1e-5
    assert_states(base,saved);assert scale>=0


def test_actual_diagnostic_preserves_upload_dispatch_model_rng_and_pair_identities(tmp_path):
    torch.set_num_threads(2)
    h,u,n,w,models=setup('a07',cids=(1,1,40,70),lens=(2,2,10,20))
    h.aggregate(u,n,w);base=h.aggregate(u,n,w);d=decision();actual,report=fuse(base,w,u,n,h,d,.2,False)
    sent=[split_model(w,models[p].state_dict()) for c,p in h.slots]
    obs=SimpleNamespace(dispatches=sent,round_number=2,cfg=config(),fusion=report,
        observations=[dict(own_gain=.1) for _ in u])
    users={c:list(range(c*12,(c+1)*12)) for c in range(100)}
    probe=MechanismDiagnostic(TrainData(),users,tmp_path);probe.models=models;probe.observer=obs
    before=rng_state();saved=copy.deepcopy((u,sent,w,[m.state_dict() for m in models],actual))
    probe.run(base,actual,w,u,n,h,d,False)
    assert_rng(before,rng_state())
    for x,y in zip(u,saved[0]):assert_states(x,y)
    for x,y in zip(sent,saved[1]):assert_states(x,y)
    assert_states(w,saved[2]);assert_states(actual,saved[4])
    for m,s in zip(models,saved[3]):assert_states(m.state_dict(),s);assert m.training
    row=json.loads((tmp_path/'iv_e_rounds.jsonl').read_text())
    assert row['figure7'] and len(row['figure8'])==18
    assert all(r['receiver'] not in r['source_clients'] and r['receiver_tier']!=r['source_tier'] for r in row['figure7'])
    assert any(len(r['source_slots'])>len(r['source_clients']) for r in row['figure7'])
    assert all(r['receiver'] not in {c for c,p in h.slots} for r in row['figure8'])
    assert all(r['utility_vs_a07']==0 for r in row['figure8'] if r['method']=='A07')
    assert not list(tmp_path.rglob('*.pt'))
