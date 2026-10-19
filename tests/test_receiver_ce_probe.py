import copy
import json
import random

import numpy as np
import torch
from torch import nn

from adapterfl_methods import rng_state
from semcons_scope import ScopeHooks
from test_adapterfl_methods import args_for,assert_states,assert_rng
from tools.receiver_ce_probe import fixed_plan,materialize,ce_vector,inject,select_pairs,ReceiverProbe,observed,PROTOCOL
from tools.analyze_receiver_ce import auc,ce_sign,summarize,incremental,analyze


class TrainData:
    targets=[0,1]*12
    def __getitem__(self,i):
        return torch.arange(7).float()+i+torch.rand(7)+random.random()+np.random.rand(),self.targets[i]


class Tiny(nn.Module):
    def __init__(self,width):
        super().__init__();self.bn=nn.BatchNorm1d(width,track_running_stats=False)
        self.weight=nn.Parameter(torch.randn(2,width))
    def forward(self,x):return {'output':self.bn(x[:,:self.weight.shape[1]])@self.weight.T}


def test_fixed_training_plan_views_and_rng():
    data=TrainData();indices=list(range(12));p,classes=fixed_plan(data,indices,1)
    assert len(p)==4 and set(p)<=set(indices) and classes==[0,1]
    before=rng_state();a=materialize(data,p,1);b=materialize(data,p,1)
    assert a['sha256']==b['sha256'] and torch.equal(a['x'],b['x'])
    assert_rng(before,rng_state())


def test_injection_support_scale_and_input_protection():
    net=Tiny(7);source=Tiny(3).state_dict();sent=copy.deepcopy(source)
    source['weight'][0,0]+=2
    original=copy.deepcopy(net.state_dict());source_copy=copy.deepcopy(source)
    region={'cells':[{'key':'weight','bounds':((0,1),(0,1)),'numel':1}]}
    inject(net,region,source,sent,.25)
    assert abs(float(net.weight[0,0]-original['weight'][0,0])-.5)<1e-6
    assert torch.equal(net.weight[1],original['weight'][1])
    assert torch.equal(net.weight[0,1:],original['weight'][0,1:])
    assert_states(source,source_copy);assert_states(net.bn.state_dict(),{'weight':original['bn.weight'],'bias':original['bn.bias']})


def test_same_materialized_bn_batch_zero_control():
    data=TrainData();plan,_=fixed_plan(data,range(12),2);probe=materialize(data,plan,2)
    net=Tiny(3).eval();a=ce_vector(net,probe);before=copy.deepcopy(net.state_dict())
    inject(net,{'cells':[{'key':'weight','bounds':((0,2),(0,3)),'numel':6}]},before,before,.2)
    b=ce_vector(net,probe)
    assert torch.equal(a,b) and float((a-b).abs().max())==0


def test_fixed_sign_strata_and_physical_client_exclusion():
    class Hook:round_index=0;slots=[(1,0),(1,2),(2,6),(3,6)]
    region={'id':0};cos={0:-.2,1:-.1,2:.2,3:.3}
    a,counts=select_pairs(Hook(),region,list(range(4)),cos)
    b,_=select_pairs(Hook(),region,list(range(4)),cos)
    assert a==b and {v[0] for v in a}=={'conflict','nonconflict'}
    assert counts['conflict']['candidate_slots']==2 and counts['conflict']['selected_physical_senders']==1
    assert all(Hook.slots[i][0]!=Hook.slots[j][0] for _,i,j in a)


def test_observer_preserves_actual_aggregation_history_rng_and_models(tmp_path):
    torch.set_num_threads(2)
    models=[Tiny(i) for i in range(1,8)];h=ScopeHooks(args_for('s07',num_classes=2,conflict_lambda=4.),models)
    h.start_round(0);h.slots=[(1,0),(1,2),(2,6)]
    h.observations=[{'delta':torch.tensor([[1.,-1.],[1.,-1.]]),'counts':torch.tensor([2,2])} for _ in h.slots]
    s=[copy.deepcopy(models[p].state_dict()) for c,p in h.slots];u=copy.deepcopy(s)
    for i,state in enumerate(u):state['weight']+=(-.02 if i<2 else .03)
    w=copy.deepcopy(models[-1].state_dict());lens=[12,12,12]
    obs=ReceiverProbe(TrainData(),{1:list(range(12)),2:list(range(12,24))},tmp_path);obs.templates=models
    reference=copy.deepcopy(h);expected=reference.aggregate(u,lens,w,s)
    originals=[copy.deepcopy(m.state_dict()) for m in models];saved_s=copy.deepcopy(s);saved_u=copy.deepcopy(u)
    before=rng_state();actual=observed(ScopeHooks.aggregate,obs)(h,u,lens,w,s)
    assert_states(actual,expected);assert_rng(before,rng_state())
    assert h.history==reference.history and h.round_diagnostics==reference.round_diagnostics
    for m,state in zip(models,originals):assert_states(m.state_dict(),state);assert m.training
    for a,b in zip(s,saved_s):assert_states(a,b)
    for a,b in zip(u,saved_u):assert_states(a,b)
    row=json.loads((tmp_path/'receiver_ce.jsonl').read_text())
    assert row['records'] and all(r['sender']!=r['receiver'] for r in row['records'])
    for r in row['records']:
        assert r['alpha']==r['sender_weight']/r['native_mass']
        assert r['native_mass']==r['covering_slot_count']*12
    assert (tmp_path/'raw_round_01.pt').is_file()


def test_no_class_with_two_samples_is_missing_observation(tmp_path):
    class Empty(TrainData):targets=[0,1]
    probe=ReceiverProbe(Empty(),{1:[0,1]},tmp_path)
    assert probe.plans[1]==[] and probe.probes[1] is None


def test_protocol_is_bounded_and_preserves_original_regions():
    assert PROTOCOL['rounds']==5 and PROTOCOL['max_classes']*PROTOCOL['samples_per_class']==32
    assert PROTOCOL['ce_tolerance']==1e-6 and PROTOCOL['feedback_to_training'] is False


def test_auc_ties_and_missing_sign_are_explicit():
    assert auc([0,0],[0,1])==.5 and auc([0,1],[0,1])==1
    assert auc([1,0],[0,1])==0 and auc([1,2],[1,1]) is None
    assert ce_sign(1e-8,1e-6)=='neutral'


def test_receiver_specific_opposite_signs_not_called_global_harm():
    cell={'delta_ce':.01,'sender_has_training_class':True,'in_sender_q_classes':True}
    row={'valid':True,'round':2,'sender':1,'receiver':2,'sender_slot':0,'region':0,'sender_profile':0,'receiver_profile':2,
         'q':.1,'direction_cosine':.2,'direction_risk':0.,'pair_direction_cosine':.3,'sender_own_gain':.1,'delta_ce':.01,'classes':[cell]}
    other=dict(row,receiver=3,delta_ce=-.02,classes=[dict(cell,delta_ce=-.02)])
    result=summarize([row,other],PROTOCOL)
    assert result['mixed_receiver_sign_contexts']==1
    assert result['outcomes']=={'worse':1,'better':1} and result['incremental_status']=='无法判断'


def test_fixed_leave_round_out_increment_without_gate_training():
    rows=[]
    for r in range(2,6):
        for i in range(20):
            q=(i%7)/7+r*.01
            rows.append({'round':r,'q':q,'direction_cosine':(i%3)*.2,'delta_ce':.1*q})
    result=incremental(rows)
    assert len(result['folds'])==4 and result['direction_plus_q_mse']<1e-20
    assert result['direction_mse']>1e-6


def test_incomplete_diagnostic_rejected(tmp_path):
    import pytest
    (tmp_path/'protocol.json').write_text(json.dumps(PROTOCOL))
    (tmp_path/'receiver_ce.jsonl').write_text('')
    with pytest.raises(ValueError,match='Exactly five'):analyze(tmp_path)
