"""Small numerical controls and the original loop; no performance tuning."""
import copy
import csv
import json
import math

import pytest
import torch

from adapterfl_methods import MethodHooks
from models.Fed import Aggregation_AdaptiveFL
from tonight_aggregation import ORDER, TonightHooks
from test_adapterfl_methods import args_for, assert_states, tiny_models
from test_evidence_aggregation import fixture


def setup(method='t01',**changes):
    _,models,u,n,w,s=fixture(**changes)
    h=TonightHooks(args_for(method,conflict_lambda=4.),models)
    h.set_slots([(1,1),(1,3),(2,6)])
    e=MethodHooks(args_for('m2',conflict_lambda=4.),models)
    expected=e.aggregate(u,n,w,s)
    return h,models,u,n,w,s,e,expected


@pytest.mark.parametrize('method',['t01','t02','t03','t04','t09','t10'])
def test_original_scores_and_no_mutation(method):
    h,_,u,n,w,s,e,expected=setup(method)
    before=copy.deepcopy((u,w,s));h.aggregate(u,n,w,s)
    for k in ('cosine','conflict_weights','unnormalized_log_weights'):
        assert h.round_diagnostics[k]==e.round_diagnostics[k]
    for a,b in zip(u,before[0]):assert_states(a,b)
    assert_states(w,before[1])
    for a,b in zip(s,before[2]):assert_states(a,b)


def test_t01_exact_original_m3_geometry_and_t02_global():
    h,models,u,n,w,s,_,e03=setup('t01')
    m3=MethodHooks(args_for('m3',partition_alpha_shared=1.,partition_alpha_remainder=.98),models)
    assert_states(h.aggregate(u,n,w,s),m3.aggregate([e03],[1],w))
    h,_,u,n,w,s,_,e03=setup('t02')
    result=h.aggregate(u,n,w,s)
    assert_states(result,{k:w[k]+.98*(e03[k]-w[k]) for k in w})


def test_t03_neutral_round_not_scaled_and_t04_round2_exact_and_restore():
    values=[[1.,0.],[2.,0.,1.,1.],[3.,0.,2.,2.,2.,2.,2.]]
    h,_,u,n,w,s,_,e03=setup('t03',values=values)
    assert_states(h.aggregate(u,n,w,s),e03)
    assert h.round_diagnostics['scale_applied'] is False
    h,models,u,n,w,s,_,e03=setup('t04')
    assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
    restored=TonightHooks(args_for('t04',conflict_lambda=4.),models)
    restored.load_state_dict(h.state_dict(),w)
    assert_states(restored.aggregate(u,n,w,s),e03)
    assert_states(h.aggregate(u,n,w,s),e03)


@pytest.mark.parametrize('method',['t05','t06'])
def test_physical_identity_exclusion_and_identical_peer_duplication(method):
    h,_,u,n,w,s,_,_=setup(method)
    # Only client2 is a valid reference for both client1 slots.
    h.aggregate(u,n,w,s);cos=h.round_diagnostics['cosine']
    assert cos[0]==pytest.approx(-1/math.sqrt(2))
    h.set_slots([(1,1),(1,3),(2,6),(2,6)])
    h.aggregate(u+[u[2]],n+[n[2]],w,s+[s[2]])
    assert h.round_diagnostics['cosine'][:2]==pytest.approx(cos[:2])
    # Same-client-only uploads cannot provide independent evidence.
    h.set_slots([(1,1),(1,3),(1,6)])
    result=h.aggregate(u,n,w,s)
    assert h.round_diagnostics['cosine']==[None]*3
    assert_states(result,Aggregation_AdaptiveFL(u,n,w))


@pytest.mark.parametrize('method',['t07','t08'])
def test_classifier_backbone_reuse_native_scoring(method):
    models=tiny_models();w=models[-1].state_dict()
    sent=[{k:torch.zeros_like(v) for k,v in m.state_dict().items()} for m in models[:3]]
    u=[{k:torch.full_like(v,float(i+1) if k.startswith('classifier.') else (1. if i<2 else -2.)) for k,v in d.items()} for i,d in enumerate(sent)]
    h=TonightHooks(args_for(method,conflict_lambda=4.),models)
    result=h.aggregate(u,[1,1,1],w,sent)
    keys=h.round_diagnostics['scoring_keys'];sub=lambda d:{k:d[k] for k in keys}
    native=MethodHooks(args_for('m2',conflict_lambda=4.),models)
    native.conflict([sub(d) for d in u],[1,1,1],sub(w),[sub(d) for d in sent])
    assert h.round_diagnostics['cosine']==native.round_diagnostics['cosine']
    assert_states(result,Aggregation_AdaptiveFL(u,native.round_diagnostics['conflict_weights'],w))


def test_t09_preserves_e03_projection_dispatch_and_outside_evidence():
    h,_,u,n,w,s,e,e03=setup('t09',sent_values=(10,20,30))
    result=h.aggregate(u,n,w,s)
    weights=e.round_diagnostics['conflict_weights'];beta=weights[2]/sum(weights)
    # Only slot2 conflicts: d=(-1,1), ref=(2,0), p=(-1,0), z=(0,1).
    assert result['w'][0]==e03['w'][0]
    assert result['w'][1]==pytest.approx(float(e03['w'][1])+(1/3-beta),abs=3e-6)
    assert torch.equal(result['w'][2:],e03['w'][2:])
    assert h.round_diagnostics['protected_components'][0]['restored_l2']==pytest.approx(1/3-beta)
    assert all(torch.isfinite(v).all() for v in result.values())


def test_t10_preserves_dispatch_fusion_and_intersection():
    h,_,u,n,w,s,e,e03=setup('t10',sent_values=(10,20,30))
    result=h.aggregate(u,n,w,s);weights=e.round_diagnostics['conflict_weights']
    d=[{k:a[k]-b[k] for k in w} for a,b in zip(u,s)]
    learning=Aggregation_AdaptiveFL(d,[a*b for a,b in zip(n,weights)],{k:torch.zeros_like(v) for k,v in w.items()})
    assert torch.equal(result['w'][:2],e03['w'][:2])
    assert torch.equal(result['w'][2:],(e03['w']-.02*learning['w'])[2:])


def test_all_ten_native_loop_two_rounds_pairing_and_checkpoints(tmp_path,monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    monkeypatch.setattr(training,'get_model_list',lambda a:(tiny_models(),[(.4,2,1.)]*7))
    data=torch.utils.data.TensorDataset(torch.randn(200,3,4,4),torch.arange(200)%2)
    users={i:[i*2,i*2+1] for i in range(100)};schedules=[]
    for method in ORDER:
        args=args_for(method,tmp_path,epochs=2,seed=1,num_users=100,frac=.1,local_ep=1,
            local_bs=2,bs=50,eval_profiles='full',conflict_lambda=4.,method_run_name='short_check')
        set_random_seed(1);training.AdaptiveFL(args,data,data,users)
        p=tmp_path/method/'short_check'
        state=torch.load(p/'latest.pt',weights_only=False)
        assert state['next_round']==2 and state['method_state']['rounds_seen']==2
        assert all(torch.isfinite(v).all() for v in state['global'].values())
        with (p/'method_diagnostics.csv').open() as f:records=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        assert json.loads((p/'completed.json').read_text())['success']
        schedules.append([r['slots'] for r in records])
    assert all(s==schedules[0] for s in schedules)
