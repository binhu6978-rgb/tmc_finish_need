"""HGKR numerical invariants; original algorithms are the golden references."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from experiments.baseline import content_hash, state_hash
from hgkr.core import (Config, KernelHistory, anchored_aggregate, anchored_delta, choose_kernels,
                       clone_state, reset_copy, shared_specs)
from hgkr.protocol import allocation_cost, capacity, initialize_models, make_schedule, model_costs
from experiments.run_hgkr import run_round
from models.Fed import Aggregation_AdaptiveFL, split_model
from semconsfl.aggregation import coverage
from semconsfl.measurement import capture_rng, restore_rng

torch.set_num_threads(1)


def args(device='cpu'):
    return SimpleNamespace(width_ration=[.4,.66,1.],depth_saved=[2],model='resnet',num_channels=3,
        num_classes=100,device=torch.device(device),seed=1,local_ep=5,local_bs=50,optimizer='sgd',
        lr=.01,lr_decay=.998,momentum=.5,weight_decay=1e-4,dataset='cifar100')


def equal(a,b):
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_real_profiles_coverage_shapes_and_macs():
    models,profiles=initialize_models(args())
    assert [p[:2] for p in profiles]==[(.4,2),(.66,2),(1.,2)]
    assert [sum(p.numel() for p in m.parameters()) for m in models]==[2120985,5159621,11424356]
    masks,stats=coverage(models)
    assert stats==dict(all_profiles=2120985,partial_profiles=3038636,full_only=6264735,total_parameters=11424356)
    specs=shared_specs(models)
    assert len(specs)==21
    for k,shape in specs.items():
        assert torch.all(masks[k][:shape[0],:shape[1]]==3)
    before=capture_rng()
    hashes=[state_hash(m.state_dict()) for m in models]
    costs=model_costs(models)
    assert costs[0]['forward_mac']<costs[1]['forward_mac']<costs[2]['forward_mac']
    assert hashes==[state_hash(m.state_dict()) for m in models]
    assert torch.equal(before['torch'],capture_rng()['torch'])


@pytest.mark.parametrize('profiles',[(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)])
def test_anchored_all_coverage_and_golden(profiles):
    full={'x.weight':torch.arange(12,dtype=torch.float32).reshape(3,4)}
    sizes=[(1,2),(2,3),(3,4)]
    starts=[{'x.weight':full['x.weight'][:sizes[p][0],:sizes[p][1]].clone()} for p in profiles]
    uploads=[{k:v+(i+1) for k,v in s.items()} for i,s in enumerate(starts)]
    counts=list(range(1,len(starts)+1))
    expected=Aggregation_AdaptiveFL(uploads,counts,full)
    equal(anchored_aggregate(full,uploads,starts,counts,2,False),expected)
    torch.testing.assert_close(anchored_delta(full,uploads,starts,counts)['x.weight'],expected['x.weight'],rtol=1e-6,atol=1e-6)
    # Reset impulse without any training must cancel exactly, including uncovered coords.
    reset_starts=[{k:v+1234 for k,v in s.items()} for s in starts]
    equal(anchored_aggregate(full,reset_starts,reset_starts,counts,2,True),full)
    trained=[{k:v+2 for k,v in s.items()} for s in reset_starts]
    result=anchored_aggregate(full,trained,reset_starts,counts,2,True)['x.weight']
    largest=max(profiles)
    want=full['x.weight'].clone(); want[:sizes[largest][0],:sizes[largest][1]]+=2
    assert torch.equal(result,want)


def test_round0_independent_golden_and_no_reset_enforced():
    old={'x.weight':torch.zeros(3)}
    starts=[{'x.weight':torch.ones(2)*9}]
    uploads=[{'x.weight':torch.ones(2)*10}]
    equal(anchored_aggregate(old,uploads,starts,[2],0,False),Aggregation_AdaptiveFL(uploads,[2],old))
    with pytest.raises(ValueError):
        anchored_aggregate(old,uploads,starts,[2],0,True)
    with pytest.raises(ValueError):
        anchored_aggregate(old,uploads,starts,[2],1,False)


def mature_history():
    specs={'conv.weight':(4,2,3,3)}
    cfg=replace(Config(),reset_fraction=.25)
    history=KernelHistory(specs,cfg)
    state={'conv.weight':torch.ones(6,5,3,3),'bn.weight':torch.ones(6)}
    for r in range(4):
        new=clone_state(state)
        new['conv.weight'][0]+=1 if r%2 else -1
        new['conv.weight'][1:]+=1
        history.update(r,state,new); state=new
    return cfg,history,state


def test_history_past_only_read_idempotent_reload_and_direction():
    cfg,h,state=mature_history()
    info=h.scores(4)['conv.weight']
    assert info['score'][0]>info['score'][1]
    assert info['valid'].all() and info['covered_rounds']==3
    assert torch.equal(info['score'],h.scores(4)['conv.weight']['score'])
    saved=h.state_dict(); restored=KernelHistory(h.specs,cfg); restored.load_state_dict(saved)
    assert torch.equal(restored.scores(4)['conv.weight']['score'],info['score'])
    with pytest.raises(ValueError): h.update(3,state,state)
    with pytest.raises(ValueError): h.scores(5)
    with pytest.raises(ValueError): restored.load_state_dict(dict(saved,config={}))


def test_zero_direction_and_history_warmup():
    h=KernelHistory({'conv.weight':(4,2,3,3)},replace(Config(),reset_fraction=.25))
    state={'conv.weight':torch.zeros(4,2,3,3)}
    for r in range(4):
        s=h.scores(r)
        a,_=choose_kernels(s,h.config,'B2',1,r,0); b,_=choose_kernels(s,h.config,'B3',1,r,0)
        assert a==b and not a['conv.weight']
        h.update(r,state,state)
    s=h.scores(4)
    a,_=choose_kernels(s,h.config,'B2',1,4,0); b,diag=choose_kernels(s,h.config,'B3',1,4,0)
    assert a==b and len(a['conv.weight'])==1 and diag['conv.weight']['fallback_count']==1


def test_missing_coverage_and_expired_history_are_not_evidence():
    cfg,h,state=mature_history()
    h.update(4,state,state,covered={'conv.weight':False})
    assert len(h.records['conv.weight'])==3
    for r in range(5,25):
        h.update(r,state,state,covered={'conv.weight':False})
    assert not h.scores(25)['conv.weight']['ready']


def test_reset_mapping_budget_nontarget_server_and_rng_protected():
    cfg,h,old=mature_history()
    before=clone_state(old)
    rng=capture_rng()
    scores=h.scores(4)
    selected2,d2=choose_kernels(scores,cfg,'B2',1,4,0)
    selected3,d3=choose_kernels(scores,cfg,'B3',1,4,0)
    assert selected3['conv.weight']==[0]
    assert d2['conv.weight']['count']==d3['conv.weight']['count']==1
    for mode,selection in [('B2',selected2),('B3',selected3)]:
        local=clone_state(old)
        reset_copy(local,old,h.specs,selection,1,4,0)
        mask=torch.zeros_like(local['conv.weight'],dtype=torch.bool)
        mask[selection['conv.weight'],:2]=True
        assert torch.equal(local['conv.weight'][~mask],old['conv.weight'][~mask])
        assert not torch.equal(local['conv.weight'][mask],old['conv.weight'][mask])
        assert torch.equal(local['bn.weight'],old['bn.weight'])
        equal(old,before)
    after=capture_rng()
    assert torch.equal(rng['torch'],after['torch']) and rng['python']==after['python']
    assert np.array_equal(rng['numpy'][1],after['numpy'][1])
    local=clone_state(old)
    empty,_=choose_kernels(scores,cfg,'B1',1,4,0)
    assert reset_copy(local,old,h.specs,empty,1,4,0)==0
    equal(local,old)


def test_schedule_capabilities_pairing_reproducibility_and_budget():
    users={i:list(range(250+5*i)) for i in range(100)}
    costs=[{'forward_mac':3,'parameters':2},{'forward_mac':5,'parameters':5},{'forward_mac':10,'parameters':11}]
    cfg=Config()
    a=make_schedule(1,20,users,costs,cfg)
    b=make_schedule(1,20,users,costs,cfg)
    assert content_hash(a)==content_hash(b)
    for r in a:
        assert len(set(r['clients']))==10
        assert r['capacity']==[capacity(c) for c in r['clients']]
        for arm in ('B0','priority'):
            assert all(p<=capacity(c) for c,p in zip(r['clients'],r[arm]))
            cost=allocation_cost(r['clients'],r[arm],users,costs)
            assert all(x<=y+1 for x,y in zip(cost,r['ceiling']))
        assert all(abs(x-y)<=.02*y+1 for x,y in zip(r['B0_cost'],r['priority_cost']))


def test_dataloader_order_and_cpu_training_identical_before_reset_warmup():
    a=args()
    models,_=initialize_models(a)
    old=clone_state(models[-1].state_dict())
    for m in models: m.load_state_dict(split_model(old,m.state_dict()))
    data=TensorDataset(torch.randn(50,3,32,32),torch.arange(50)%100)
    rng=capture_rng()
    outputs=[]
    for mode in ('B1','B2','B3'):
        restore_rng(rng)
        pair=copy.deepcopy(models)
        h=KernelHistory(shared_specs(pair),Config()); h.last_round=0
        mask,_=coverage(pair)
        row=run_round(pair,data,{0:list(range(50))},dict(round=1,clients=[0],B0=[0],priority=[0]),a,Config(),h,mode,mask)
        assert row['reset_total']==0
        outputs.append((clone_state(pair[-1].state_dict()),capture_rng()))
    for state,r in outputs[1:]:
        equal(state,outputs[0][0]); assert torch.equal(r['torch'],outputs[0][1]['torch'])


@pytest.mark.parametrize('mode',['B0','B1','B2','B3'])
def test_real_resnet_mature_history_end_to_end(mode):
    # Real local SGD with synthetic data, explicit synthetic OLD history, not a CIFAR result.
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    a=args(device)
    models,_=initialize_models(a)
    old=clone_state(models[-1].state_dict())
    for m in models: m.load_state_dict(split_model(old,m.state_dict()))
    history=KernelHistory(shared_specs(models),Config())
    state=clone_state(old)
    for r in range(4):
        new={k:v+(1e-3 if r%2 else -1e-3) for k,v in state.items()}
        history.update(r,state,new); state=new
    # Only history is synthetic. The actual dispatch anchor is the unchanged real Full prefix.
    data=TensorDataset(torch.randn(50,3,32,32),torch.arange(50)%100)
    masks,_=coverage(models)
    row=run_round(models,data,{0:list(range(50))},dict(round=4,clients=[0],B0=[0],priority=[0]),a,Config(),history,mode,masks)
    assert (row['reset_total']>0)==(mode in ('B2','B3'))
    assert row['slots'][0]['local_learning_delta_l2']>0
    assert all(torch.isfinite(v).all() for v in models[-1].state_dict().values())
    assert history.last_round==4
