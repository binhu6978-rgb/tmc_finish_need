import copy
import json

import pytest
import torch

from adapterfl_methods import rng_state
from test_adapterfl_methods import args_for, assert_rng, assert_states, tiny_models
from tools.mature_receiver_ce import (PROTOCOL_PATH, CORE, fixed_schedule, restore_profiles,
    sha, state_sha, verify_checkpoint)
from tools.analyze_mature_q import blocked_ols, analyze, describe


def test_frozen_schedule_original_capacity_and_rng_independence():
    p = json.loads(PROTOCOL_PATH.read_text()); args=args_for(seed=1,num_users=100)
    before=rng_state(); a=fixed_schedule(args,p); assert_rng(before,rng_state())
    torch.rand(19)
    assert fixed_schedule(args,p)==a
    assert len(a)==5 and sum(len(x['clients']) for x in a)==50
    for x in a:
        for c, profile in zip(x['clients'],x['profiles']):
            assert (0 if profile<3 else 40 if profile<6 else 70)<=c<100


def test_seven_prefix_restore_copies_mature_anchor_and_resets_local_changes():
    models=tiny_models(); anchor=copy.deepcopy(models[-1].state_dict()); original=copy.deepcopy(anchor)
    hashes=restore_profiles(models,anchor)
    with torch.no_grad(): models[0].features[0].weight.add_(7)
    assert_states(anchor,original)
    assert restore_profiles(models,anchor)==hashes
    assert state_sha(models[-1].state_dict())==state_sha(anchor)
    assert all(m.features[0].weight.data_ptr()!=anchor['features.0.weight'].data_ptr() for m in models)


def test_mature_final_identity_and_source_partition_guards(tmp_path):
    from tools.mature_receiver_ce import ROOT
    p=json.loads(PROTOCOL_PATH.read_text())
    partition=ROOT/'data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
    identity=dict(seed=1,epochs=400,num_users=100,frac=.1,local_ep=5,local_bs=50,lr=.01,lr_decay=.998,
        momentum=.5,weight_decay=.0001,optimizer='sgd',dataset='cifar100',model='resnet',iid=0,data_beta=.3,
        client_hetero_ration='4:3:3',client_chosen_mode='available',method='m2',options={'conflict_lambda':4.},
        partition_sha256=sha(partition),source_sha256={n:sha(ROOT/n) for n in CORE})
    checkpoint={'identity':identity,'next_round':400,'rows':[None]*400,'global':{'weight':torch.ones(2)}}
    verify_checkpoint(checkpoint,p,partition)
    for field,value in [('next_round',399),('rows',[None]*399)]:
        bad=dict(checkpoint,**{field:value})
        with pytest.raises(ValueError,match='final400'): verify_checkpoint(bad,p,partition)
    checkpoint['global']['weight'][0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'): verify_checkpoint(checkpoint,p,partition)


def test_fixed_budget_has_no_initialization_exclusion_or_adaptive_extension():
    p=json.loads(PROTOCOL_PATH.read_text())
    assert p['primary_rounds']==[1,2,3,4,5] and p['sgd_round_index']==400
    assert p['cohorts']==5 and p['local_ep']==5 and p['local_bs']==50
    assert not p['feedback_to_training'] and not p['official_test_access']


def test_dependency_blocked_ols_holds_whole_ids_and_detects_known_increment():
    rows=[]
    for c in range(5):
        for i in range(24):
            q=(i%7)/7 + .03*c
            rows.append(dict(valid=True,round=c+1,sender=i%12,receiver=(i+5)%13,region=i%4,
                direction_cosine=(i%3)*.2,q=q,delta_ce=.1*q+.01,alpha=.1,
                scaled_delta_l2=.01,ce_before=1.,sender_own_gain=.1))
    for key in ('round','sender','receiver','region'):
        out=blocked_ols(rows,key)
        assert out['direction_plus_q_mse']<1e-20 and out['direction_mse']>1e-6
        assert sum(f['test_pairs'] for f in out['folds'])==len(rows)
        for f in out['folds']:
            assert f['test_pairs']==sum(r[key]==f['held_out'] for r in rows)


def test_insufficient_blocks_and_incomplete_diagnostic_not_interpreted(tmp_path):
    rows=[dict(valid=True,q=.1,delta_ce=.1,region=0)]
    assert blocked_ols(rows,'region')['status']=='无法判断'
    (tmp_path/'protocol.json').write_text(json.dumps(json.loads(PROTOCOL_PATH.read_text())))
    (tmp_path/'diagnostic_completed.json').write_text(json.dumps({'success':True,'global_updates':0}))
    (tmp_path/'receiver_ce.jsonl').write_text('')
    with pytest.raises(ValueError,match='Five completed'): analyze(tmp_path)


def test_class_sensitivity_counts_use_recomputed_ce_not_original_outcome():
    p=json.loads(PROTOCOL_PATH.read_text())
    row=dict(valid=True,round=1,sender=1,receiver=2,sender_slot=0,region=0,sender_profile=0,receiver_profile=6,
             q=.1,direction_cosine=-.2,direction_risk=.2,pair_direction_cosine=.1,
             sender_own_gain=.1,delta_ce=-.01,classes=[],outcome='worse')
    result=describe([row],p)
    assert result['outcomes']==result['by_cohort']['1']==result['by_region']['0']=={'better':1}
