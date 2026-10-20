"""Small algebra/mapping tests, not performance experiments."""
import copy
import csv
import io
import contextlib

import pytest
import torch
from torch import nn

from native_group_aggregation import NativeGroupHooks,MODES,reliability,update_norm
from models.Fed import Aggregation_AdaptiveFL
from test_adapterfl_methods import args_for,assert_states,assert_rng,tiny_models


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


class One(nn.Module):
    def __init__(self,n):super().__init__();self.weight=nn.Parameter(torch.zeros(n))


def case(method,profiles=(0,0,2,6),cids=(10,11,45,75),lens=(1,3,7,9),values=(.1,.2,.3,.4)):
    models=[One(i) for i in range(1,8)]
    h=NativeGroupHooks(args_for(method),models);h.set_slots(zip(cids,profiles))
    w={'weight':torch.arange(7).float()/100}
    sent=[{'weight':w['weight'][:p+1].clone()} for p in profiles]
    uploads=[{'weight':s['weight']+v} for s,v in zip(sent,values)]
    return h,uploads,list(lens),w,sent


def active(h,u,n,w,s):
    first=h.aggregate(u,n,w,s)
    assert_states(first,Aggregation_AdaptiveFL(u,n,w))
    return h.aggregate(u,n,w,s)


@pytest.mark.parametrize('method',MODES)
def test_all_first_round_exact_native_inputs_rng_and_snapshot_state(method):
    from adapterfl_methods import rng_state
    h,u,n,w,s=case(method);snapshot=copy.deepcopy((u,w,s));before=rng_state()
    actual=h.aggregate(u,n,w,s)
    assert_states(actual,Aggregation_AdaptiveFL(u,n,w));assert_rng(before,rng_state())
    for a,b in zip(u,snapshot[0]):assert_states(a,b)
    assert_states(w,snapshot[1])
    for a,b in zip(s,snapshot[2]):assert_states(a,b)
    restored,*_=case(method);restored.load_state_dict(h.state_dict(),w)
    assert_states(restored.aggregate(u,n,w,s),h.aggregate(u,n,w,s))


def test_profile_balancing_and_static_shared_complement_relation():
    output={}
    for method in ('g01','g04','g05'):
        h,u,n,w,s=case(method);output[method]=active(h,u,n,w,s)['weight']
    native=Aggregation_AdaptiveFL(u,n,w)['weight']
    assert float(output['g01'][0])==pytest.approx((.175+.3+.4)/3)
    assert torch.equal(output['g04'][:1],output['g01'][:1])
    assert torch.equal(output['g04'][1:],native[1:])
    assert torch.equal(output['g05'][:1],native[:1])
    assert torch.equal(output['g05'][1:],output['g01'][1:])


def test_physical_dedup_is_per_region_and_does_not_merge_sgd_uploads():
    h,u,n,w,s=case('g02',profiles=(0,2,6),cids=(10,10,75),lens=(2,2,6),values=(1.,3.,5.))
    result=active(h,u,n,w,s)
    assert float(result['weight'][0])==pytest.approx((2.*2+5.*6)/8)
    # Outside profile0 the repeated client has only one covering slot: Native.
    assert torch.equal(result['weight'][1:],Aggregation_AdaptiveFL(u,n,w)['weight'][1:])
    assert len(u)==3


def test_capability_groups_differ_from_profile_groups_and_equal_slots():
    out={}
    for method in ('g01','g03','g06'):
        h,u,n,w,s=case(method,profiles=(0,2,6),cids=(10,11,50),lens=(1,3,6),values=(1.,3.,5.))
        out[method]=active(h,u,n,w,s)['weight']
    assert float(out['g01'][0])==pytest.approx(3.)
    assert float(out['g03'][0])==pytest.approx(((1.+9.)/4+5.)/2)
    expected=Aggregation_AdaptiveFL(u,[1]*len(u),w)['weight']
    assert torch.allclose(out['g06'],expected,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('regional,scale',[('g07','g08'),('g09','g10')])
def test_scale_control_matches_virtual_region_norm_and_preserves_uncovered(regional,scale):
    out={};hooks={}
    for method in (regional,scale):
        h,u,n,w,s=case(method,profiles=(0,2,2),cids=(1,2,3),lens=(2,2,2),values=(1.,-.2,.8))
        out[method]=active(h,u,n,w,s);hooks[method]=h
        assert torch.equal(out[method]['weight'][3:],w['weight'][3:])
    target=update_norm(out[regional],w)
    assert update_norm(out[scale],w)==pytest.approx(target,rel=2e-6,abs=1e-7)
    assert hooks[scale].round_diagnostics['virtual_regional_update_l2']==pytest.approx(target)
    if regional=='g07':
        counts=hooks[regional].coverage
        assert any(c==0 for c in counts) and all(c in (0,4,6) for c in counts)
        assert hooks[scale].coverage==counts


def test_reliability_independent_clients_consistent_noisy_zero_and_duplicate_fallback():
    slots=[(1,0),(2,0)];n=[1,1]
    gain,d=reliability({0:torch.ones(3).double(),1:torch.ones(3).double()},[0,1],slots,n)
    assert gain==1 and d['estimated_mean_noise_energy']==0
    gain,d=reliability({0:torch.ones(3).double(),1:torch.full((3,),-.5).double()},[0,1],slots,n)
    assert gain==0 and d['estimated_mean_noise_energy']>d['mean_energy']
    gain,d=reliability({0:torch.zeros(3).double(),1:torch.zeros(3).double()},[0,1],slots,n)
    assert gain==1
    gain,d=reliability({0:torch.ones(3).double(),1:-torch.ones(3).double()},[0,1],[(1,0),(1,0)],n)
    assert gain==1 and d['physical_clients']==1
    # Three slots, two physical blocks: native mass retained, not 3 independent samples.
    gain,d=reliability({0:torch.ones(2).double(),1:torch.ones(2).double(),2:-torch.ones(2).double()},
        [0,1,2],[(1,0),(1,0),(2,0)],[1,1,1])
    assert d['physical_clients']==2 and d['physical_weight_square_sum']==pytest.approx(5/9)


@pytest.mark.parametrize('method',MODES)
def test_one_source_neutral_and_zero_update(method):
    h,u,n,w,s=case(method,profiles=(6,6),cids=(1,1),lens=(2,2),values=(0.,0.))
    assert_states(active(h,u,n,w,s),Aggregation_AdaptiveFL(u,n,w))


def test_two_round_original_loop_integration_all_ten(tmp_path,monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    data=torch.utils.data.TensorDataset(torch.arange(4*3*8*8).reshape(4,3,8,8).float()/1000,torch.tensor([0,1,0,1]))
    users={i:[0,1,2,3] for i in range(100)}
    monkeypatch.setattr(training,'get_model_list',lambda args:(tiny_models(),[(.4,2,i+1.) for i in range(7)]))
    monkeypatch.setattr(training,'tqdm',lambda x:x)
    for method in MODES:
        args=args_for(method,tmp_path/method,epochs=2,eval_profiles='full',frac=.1)
        set_random_seed(1)
        with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,data,data,users)
        folders=list((tmp_path/method/method).iterdir());assert len(folders)==1
        with (folders[0]/'accuracy_each_round.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==2 and set(k for k in rows[0] if k.startswith('P'))=={'P6_accuracy','P6_ce'}
        saved=torch.load(folders[0]/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['next_round']==2 and saved['method_state']['rounds_seen']==2
