import copy
import math

import pytest
import torch

from adapterfl_methods import rng_state
from models.Fed import Aggregation_AdaptiveFL
from test_adapterfl_methods import args_for, assert_states, assert_rng
from test_semcons_scope import setup, One
from test_receiver_ce_probe import Tiny
from tools.structure_counterfactual import whole_client_control, aggregates, geometry, rescale, evaluate


def test_whole_client_and_region_share_evidence_but_have_different_scope():
    h,u,n,w,s=setup('s03')
    a4=h.aggregate(u,n,w,s); base=Aggregation_AdaptiveFL(u,n,w)
    a3,diag=whole_client_control(h,u,n,s,base,h.round_diagnostics['regions'])
    assert torch.equal(a4['w'][1:],base['w'][1:])
    assert not torch.equal(a3['w'][1:3],base['w'][1:3])
    assert torch.equal(a3['w'][3:],base['w'][3:])
    assert diag['physical_client_risks'][2]==pytest.approx(1/3)
    assert any(row['extrapolated_slots'] for row in diag['regions'])


def test_one_physical_client_falls_back_exactly():
    h,u,n,w,s=setup('s03',cids=(7,7,7))
    base=Aggregation_AdaptiveFL(u,n,w); a4=h.aggregate(u,n,w,s)
    a3,diag=whole_client_control(h,u,n,s,base,h.round_diagnostics['regions'])
    assert_states(a3,base); assert_states(a4,base)
    assert not diag['physical_client_risks']


def test_common_non_neutral_gates_cancel_without_changing_native_scale():
    h,u,n,w,s=setup('s03',profiles=(6,6),cids=(1,2),deltas=[[1.]*7,[-2.]*7],response_values=(1.,-1.))
    base=Aggregation_AdaptiveFL(u,n,w); a4=h.aggregate(u,n,w,s)
    a3,diag=whole_client_control(h,u,n,s,base,h.round_diagnostics['regions'])
    assert_states(a3,base); assert_states(a4,base)
    assert any(row['common_gate_cancellation'] for row in diag['regions'])


def test_duplicate_uploads_keep_native_slot_mass_and_pool_only_global_risk():
    h,u,n,w,s=setup('s03',profiles=(6,6,6),cids=(1,1,2),deltas=[[1.]*7,[-2.]*7,[3.]*7])
    base=Aggregation_AdaptiveFL(u,n,w); h.aggregate(u,n,w,s)
    result,diag=whole_client_control(h,u,n,s,base,h.round_diagnostics['regions'])
    assert diag['physical_client_risks']=={1:.5,2:1.}
    a,b=math.exp(-2),math.exp(-4)
    expected=(a*(1-2)+b*3)/(2*a+b)
    assert torch.allclose(result['w'],torch.full((7,),expected),atol=1e-6)
    assert all(len(r['slots'])==3 for r in diag['regions'])


def test_native_controls_m3_uncovered_protection_and_all_inputs_immutable():
    h,u,n,w,s=setup('s03'); models=[One(i) for i in range(1,8)]
    snap={'uploads':u,'lens':n,'dispatched':s,'slots':h.slots,'regions':h.regions}
    saved=copy.deepcopy([w,u,s]); args=args_for(num_classes=2)
    states,_,_=aggregates(models,args,w,snap)
    base=Aggregation_AdaptiveFL(u,n,w)
    assert_states(states['A0'],base)
    assert_states(states['A4'],h.aggregate(u,n,w,s))
    assert states['A2']['w'][0]==base['w'][0]
    assert torch.allclose(states['A2']['w'][1:],.98*base['w'][1:])
    assert states['A1']['w'][0]==pytest.approx(states['A4']['w'][0],abs=1e-6)
    assert_states(w,saved[0])
    for a,b in zip(u,saved[1]):assert_states(a,b)
    for a,b in zip(s,saved[2]):assert_states(a,b)


def test_single_scale_matching_preserves_direction_without_ce_based_choice():
    anchor={'w':torch.tensor([2.,3.,4.])}; base={'w':torch.tensor([3.,3.,4.])}
    method={'w':torch.tensor([2.,5.,4.])}
    g=geometry(method,anchor,base)
    matched=rescale(method,anchor,1/g['update_norm_ratio_A0'])
    assert geometry(matched,anchor,base)['update_l2']==pytest.approx(1.)
    assert matched['w'].tolist()==[2.,4.,4.]


def test_full_probe_evaluation_uses_fixed_batches_and_preserves_rng_and_state():
    net=Tiny(3); state=copy.deepcopy(net.state_dict())
    probes={c:{'x':torch.arange(28).float().reshape(4,7)+c,'y':torch.tensor([0,1,0,1])} for c in range(2)}
    before=rng_state(); a=evaluate(net,state,probes); b=evaluate(net,state,probes)
    assert a==b and set(a)=={0,1}; assert_rng(before,rng_state()); assert_states(state,net.state_dict())
