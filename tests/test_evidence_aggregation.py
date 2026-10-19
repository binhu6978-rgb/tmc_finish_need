import copy
import csv
import json
import math

import pytest
import torch
from torch import nn

from adapterfl_methods import MethodHooks
from evidence_aggregation import EvidenceHooks
from models.Fed import Aggregation_AdaptiveFL
from test_adapterfl_methods import args_for, assert_states, tiny_models


class Vector(nn.Module):
    def __init__(self,n):
        super().__init__();self.w=nn.Parameter(torch.zeros(n))


def fixture(method='c2',values=None,sent_values=None,profiles=(1,3,6),lens=(1,1,1)):
    models=[Vector(n) for n in range(1,8)]
    if values is None:values=[[2.,0.],[2.,0.,5.,7.],[-1.,1.,8.,9.,10.,11.,12.]]
    sent=[{'w':torch.full((p+1,),float(sent_values[i]) if sent_values else 0.)} for i,p in enumerate(profiles)]
    uploads=[{'w':s['w']+torch.tensor(v)} for s,v in zip(sent,values)]
    args=args_for(method,conflict_lambda=4.)
    return EvidenceHooks(args,models),models,uploads,list(lens),{'w':torch.zeros(7)},sent


@pytest.mark.parametrize('method',['c1','c2'])
def test_exact_original_e03_scoring_and_scope(method):
    h,models,u,n,w,s=fixture(method)
    old=MethodHooks(args_for('m2',conflict_lambda=4.),models);e03=old.aggregate(u,n,w,s)
    base=Aggregation_AdaptiveFL(u,n,w);copies=copy.deepcopy([u,w,s])
    result=h.aggregate(u,n,w,s)
    for key in ('cosine','conflict_weights','unnormalized_log_weights'):
        assert h.round_diagnostics[key]==old.round_diagnostics[key]
    assert torch.equal(result['w'][2:],base['w'][2:])
    assert not torch.equal(e03['w'][2:4],base['w'][2:4])
    if method=='c1':assert torch.equal(result['w'][:2],e03['w'][:2])
    for a,b in zip(u,copies[0]):assert_states(a,b)
    assert_states(w,copies[1])
    for a,b in zip(s,copies[2]):assert_states(a,b)


def test_c2_preserves_orthogonal_and_nonconflicting_components():
    h,_,u,n,w,s=fixture('c2')
    result=h.aggregate(u,n,w,s)
    g=math.exp(-4/math.sqrt(2))
    assert result['w'][0]==pytest.approx((4-g)/3,abs=1e-6)
    assert result['w'][1]==pytest.approx(1/3,abs=1e-6)
    assert h.round_diagnostics['adjusted_projection_slot_count']==1
    rows=h.round_diagnostics['projection_slots']
    assert rows[0]['negative_projection_l2']==0 and rows[1]['negative_projection_l2']==0
    assert rows[2]['negative_projection_l2']==pytest.approx(1.)
    assert rows[2]['remaining_component_l2']==pytest.approx(1.)


def test_first_independent_dispatch_fusion_matches_c1_not_initialization_delta():
    h,_,u,n,w,s=fixture('c2',sent_values=(10,20,30))
    result=h.aggregate(u,n,w,s);rows=h.round_diagnostics['projection_slots']
    beta=[r['e03_normalized_fraction'] for r in rows]
    fusion=sum(a*b for a,b in zip(beta,(10,20,30)))
    g=math.exp(-4/math.sqrt(2))
    expected=torch.tensor([fusion+(4-g)/3,fusion+1/3])
    assert torch.allclose(result['w'][:2],expected,atol=3e-6)
    assert rows[2]['delta_l2']==pytest.approx(math.sqrt(2))
    assert h.round_diagnostics['initial_m2_dispatch_fusion_shift_from_native_l2']>0


def test_uniform_m2_weight_cancellation_does_not_cancel_projection_suppression():
    kw=dict(profiles=(1,1),values=[[1.,0.],[-2.,0.]],sent_values=(10,30),lens=(1,1))
    c1,_,u,n,w,s=fixture('c1',**kw);c2,*_=fixture('c2',**kw)
    base=Aggregation_AdaptiveFL(u,n,w)
    assert_states(c1.aggregate(u,n,w,s),base)
    result=c2.aggregate(u,n,w,s)
    assert result['w'][0]==pytest.approx(20-.5*math.exp(-4),abs=3e-6)
    assert c2.round_diagnostics['adjusted_projection_slot_count']==2
    assert c2.round_diagnostics['conflict_weights']==[1.,1.]
    assert not torch.equal(result['w'][:2],base['w'][:2])


@pytest.mark.parametrize('method',['c1','c2'])
def test_no_conflict_and_zero_norm_exact_native(method):
    for values in ([[1.,0.],[2.,0.,1.,1.],[3.,0.,2.,2.,2.,2.,2.]],
                   [[0.,0.],[0.]*4,[0.]*7]):
        h,_,u,n,w,s=fixture(method,values=values,sent_values=(10,20,30))
        assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
        assert h.round_diagnostics['conflict_slot_count']==0


def test_duplicate_slots_match_e03_and_state_roundtrip():
    h,models,u,n,w,s=fixture('c2',lens=(3,3,5))
    # Both equal-weight slots remain independent, exactly as the E03 implementation.
    old=MethodHooks(args_for('m2',conflict_lambda=4.),models);old.aggregate(u,n,w,s)
    h.aggregate(u,n,w,s)
    assert h.round_diagnostics['cosine']==old.round_diagnostics['cosine']
    restored=EvidenceHooks(args_for('c2',conflict_lambda=4.),models)
    restored.load_state_dict(h.state_dict(),w)
    assert_states(h.aggregate(u,n,w,s),restored.aggregate(u,n,w,s))
    assert h.rounds_seen==restored.rounds_seen==2


def test_native_three_round_training_path_full_only_and_pairing(tmp_path,monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    torch.set_num_threads(2)
    def built(args):return tiny_models(),[(.4,2,1.)]*7
    monkeypatch.setattr(training,'get_model_list',built)
    data=torch.utils.data.TensorDataset(torch.randn(200,3,4,4),torch.arange(200)%2)
    users={i:[2*i,2*i+1] for i in range(100)}
    runs=[]
    for method in ('c1','c2'):
        args=args_for(method,tmp_path,epochs=3,seed=1,num_users=100,frac=.1,local_ep=1,local_bs=2,
                      bs=50,eval_profiles='full',conflict_lambda=4.,method_run_name='integration')
        set_random_seed(1);training.AdaptiveFL(args,data,data,users)
        root=tmp_path/method/'integration'
        checkpoint=torch.load(root/'latest.pt',weights_only=False)
        assert checkpoint['next_round']==3 and checkpoint['method_state']['rounds_seen']==3
        assert all(torch.isfinite(v).all() for v in checkpoint['global'].values())
        with (root/'method_diagnostics.csv').open() as f:diag=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        assert json.loads((root/'completed.json').read_text())['success']
        assert set(checkpoint['rows'][0]).isdisjoint({'P0_accuracy','P1_accuracy','P5_accuracy'})
        runs.append([r['slots'] for r in diag])
    assert runs[0]==runs[1]
