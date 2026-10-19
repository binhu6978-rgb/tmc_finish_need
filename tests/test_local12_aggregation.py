"""Mechanism scope tests and a short original-loop check for all local controls."""
import copy
import csv
import json
import math

import pytest
import torch

from adapterfl_methods import MethodHooks
from local12_aggregation import Local12Hooks,LocalReferenceHooks
from models.Fed import Aggregation_AdaptiveFL
from test_adapterfl_methods import args_for,assert_states,tiny_models


def setup(method='l01',values=(-1.,1.,1.),lens=(1,3,3),aligned=False):
    models=tiny_models();w={k:torch.zeros_like(v) for k,v in models[-1].state_dict().items()}
    sent=[{k:torch.full_like(v,0. if aligned else float(10*(i+1))) for k,v in models[p].state_dict().items()} for i,p in enumerate((0,3,6))]
    u=[{k:v+values[i] for k,v in d.items()} for i,d in enumerate(sent)]
    h=Local12Hooks(args_for(method,conflict_lambda=4.),models);h.set_slots([(1,0),(40,3),(70,6)])
    old=MethodHooks(args_for('m2',conflict_lambda=4.),models);e03=old.aggregate(u,list(lens),w,sent)
    return h,models,u,list(lens),w,sent,old,e03


@pytest.mark.parametrize('method',[f'l{i:02d}' for i in range(1,8)])
def test_original_direction_scores_and_input_protection(method):
    h,_,u,n,w,s,old,_=setup(method)
    snapshot=copy.deepcopy((u,w,s));h.aggregate(u,n,w,s)
    assert h.round_diagnostics['original_e03_cosine']==old.round_diagnostics['cosine']
    assert h.round_diagnostics['cosine']==old.round_diagnostics['cosine']
    if method not in ('l06','l07'):
        assert h.round_diagnostics['conflict_weights']==old.round_diagnostics['conflict_weights']
    for a,b in zip(u,snapshot[0]):assert_states(a,b)
    assert_states(w,snapshot[1])
    for a,b in zip(s,snapshot[2]):assert_states(a,b)


def test_l01_native_dispatch_m2_learning_and_l02_first_round_exact():
    h,_,u,n,w,s,old,e03=setup('l01')
    result=h.aggregate(u,n,w,s);g=old.round_diagnostics['conflict_weights']
    native_fusion=Aggregation_AdaptiveFL(s,n,w)
    weighted_fusion=Aggregation_AdaptiveFL(s,[a*b for a,b in zip(n,g)],w)
    assert_states(result,{k:e03[k]+(native_fusion[k]-weighted_fusion[k]) for k in w})
    h,_,u,n,w,s,_,e03=setup('l02')
    assert_states(h.aggregate(u,n,w,s),e03)


@pytest.mark.parametrize('method',['l01','l02'])
def test_synced_dispatch_is_exact_e03_not_roundoff_mechanism(method):
    h,models,u,n,w,s,_,e03=setup(method,aligned=True)
    assert_states(h.aggregate(u,n,w,s),e03)
    restored=Local12Hooks(args_for(method,conflict_lambda=4.),models)
    restored.load_state_dict(h.state_dict(),w)
    assert_states(restored.aggregate(u,n,w,s),e03)
    assert restored.round_diagnostics['dispatch_matches_full_anchor'] is True


@pytest.mark.parametrize('method',['l03','l04','l05'])
def test_parameter_role_protection_is_exact_native_only_on_requested_keys(method):
    h,_,u,n,w,s,_,e03=setup(method)
    result=h.aggregate(u,n,w,s);native=Aggregation_AdaptiveFL(u,n,w)
    protected=h.bn if method=='l03' else h.classifier if method=='l04' else h.bn|h.classifier
    assert h.features.isdisjoint(h.bn|h.classifier)
    for k in w:assert torch.equal(result[k],native[k] if k in protected else e03[k])


@pytest.mark.parametrize('method,values,lens,slot',[('l06',(-1.,1.,1.),(1,3,3),0),('l07',(1.,1.,-1.),(3,3,1),2)])
def test_capacity_protection_uses_intrinsic_gate_before_stabilization(method,values,lens,slot):
    h,_,u,n,w,s,old,_=setup(method,values,lens)
    h.aggregate(u,n,w,s)
    expected=list(old.round_diagnostics['unnormalized_log_weights']);expected[slot]=0.
    assert h.round_diagnostics['unnormalized_log_weights']==expected
    assert h.round_diagnostics['conflict_weights'][slot]==1.
    assert slot in h.round_diagnostics['protected_slot_indices']


def test_l08_exact_native_scoring_without_bn_and_l09_equal_cosine_mean():
    for method in ('l08','l09'):
        h,_,u,n,w,s,_,_=setup(method)
        result=h.aggregate(u,n,w,s)
        if method=='l08':
            expected=h.subset_scores(set(w)-h.bn,u,n,w,s)
        else:
            a=h.round_diagnostics['feature_cosine'];b=h.round_diagnostics['classifier_cosine']
            expected=[(x+y)/2 if x is not None and y is not None else x if y is None else y for x,y in zip(a,b)]
            assert h.round_diagnostics['valid_evidence_parts']==[2,2,2]
        assert h.round_diagnostics['cosine']==expected
        assert_states(result,Aggregation_AdaptiveFL(u,[a*b for a,b in zip(n,h.round_diagnostics['conflict_weights'])],w))


def test_l10_median_per_distinct_client_excludes_own_duplicates():
    h,_,u,n,w,s,_,_=setup('l10')
    h.set_slots([(1,0),(1,3),(2,6)]);h.aggregate(u,n,w,s)
    assert h.round_diagnostics['cosine'][0]==pytest.approx(-1.)
    assert h.round_diagnostics['valid_distinct_physical_peers']==[1,1,1]
    h.set_slots([(1,0),(1,3),(2,6),(2,6)])
    h.aggregate(u+[u[2]],n+[n[2]],w,s+[s[2]])
    assert h.round_diagnostics['cosine'][0]==pytest.approx(-1.)
    h.set_slots([(1,0),(1,3),(1,6)])
    assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))


def test_l11_bounded_risk_uses_relative_amplitude_and_original_reference():
    h,_,u,n,w,s,old,_=setup('l11',values=(-10.,1.,1.),lens=(1,1,1))
    h.aggregate(u,n,w,s)
    assert old.round_diagnostics['cosine'][0]==pytest.approx(-1.)
    assert h.round_diagnostics['amplitude_risk'][0]==pytest.approx(20/101)
    assert all(0<=r<=1 for r in h.round_diagnostics['amplitude_risk'])
    assert h.round_diagnostics['conflict_weights'][0]>math.exp(-4.)


@pytest.mark.parametrize('method',['l08','l09','l10','l11'])
def test_zero_reference_neutral(method):
    h,_,u,n,w,s,_,_=setup(method,values=(0.,0.,0.))
    assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
    assert h.round_diagnostics['conflict_weights']==[1.,1.,1.]


def test_original_loop_ref_and_eleven_controls_two_rounds(tmp_path,monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    monkeypatch.setattr(training,'get_model_list',lambda a:(tiny_models(),[(.4,2,1.)]*7))
    data=torch.utils.data.TensorDataset(torch.randn(200,3,4,4),torch.arange(200)%2)
    users={i:[i*2,i*2+1] for i in range(100)};schedules=[];ref_state=None
    for method in ('m2',*[f'l{i:02d}' for i in range(1,12)]):
        a=args_for(method,tmp_path,epochs=2,seed=1,num_users=100,frac=.1,local_ep=1,local_bs=2,
            bs=50,eval_profiles='full',conflict_lambda=4.,method_run_name='short_check')
        a._local12_reference=method=='m2'
        set_random_seed(1);training.AdaptiveFL(a,data,data,users)
        p=tmp_path/method/'short_check';state=torch.load(p/'latest.pt',weights_only=False)
        assert state['next_round']==2 and all(torch.isfinite(v).all() for v in state['global'].values())
        with (p/'method_diagnostics.csv').open() as f:rows=[json.loads(r['metrics_json']) for r in csv.DictReader(f)]
        schedules.append([r['slots'] for r in rows])
        assert json.loads((p/'completed.json').read_text())['success']
        if method=='m2':ref_state=state
        if method=='l02':assert_states(state['global'],ref_state['global'])
    assert all(s==schedules[0] for s in schedules)


def test_reference_readonly_diagnostics_return_exact_original_m2():
    _,models,u,n,w,s,old,expected=setup()
    h=LocalReferenceHooks(args_for('m2',conflict_lambda=4.),models)
    assert_states(h.aggregate(u,n,w,s),expected)
    assert h.round_diagnostics['cosine']==old.round_diagnostics['cosine']
    assert h.round_diagnostics['difference_from_e03_l2']==0.
