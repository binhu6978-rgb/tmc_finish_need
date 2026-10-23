import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from adapterfl_methods import rng_state
from semcons_a07_v2.memory import TierSemanticMemory,TierReliabilityMemory
from semcons_a07_v2.pair500 import sampled_rounds,evaluation_indices,recovery_snapshot,restore_snapshot,atomic_checkpoint,TrainingSideDiagnostic
from test_semantic_a07_iv_e import TrainData
from test_semantic_a07_v2 import config,decision
from test_final40 import setup
from test_adapterfl_methods import assert_states,assert_rng
from semcons_a07_v2.aggregation import fuse
from models.Fed import split_model


def test_frozen500_budget_and_user_removed_crd():
    p=json.loads(Path('configs/semantic_a07_pair500.json').read_text())
    r=sampled_rounds(p)
    assert len(r)==53 and r[:2]==[1,2] and r[-1]==500 and p['rounds']==500
    assert p['semantic_lambda']==.2 and p['checkpoint_interval']==50
    assert p['crd']['status']=='omitted_by_user'


def test_training_side_probe_unchanged_partition_and_no_semantic_overlap():
    from semcons_a07.measurement import probe_indices
    users={c:list(range(c*12,(c+1)*12)) for c in range(100)};saved=copy.deepcopy(users)
    plans=evaluation_indices(TrainData(),users)
    assert users==saved
    for c,v in plans.items():
        assert set(v)<=set(users[c]) and not set(v)&set(probe_indices(TrainData(),users[c],c,2))


def test_complete_recovery_round_trip_memories_adapters_rng_and_next_aggregate(tmp_path):
    import time
    h,u,n,w,models=setup('a07');h.aggregate(u,n,w)
    cfg=config();semantic=TierSemanticMemory(cfg);reliability=TierReliabilityMemory(cfg)
    semantic.tiers[0,1]=dict(q=.3,round=1,support=2,clients=[1,2])
    reliability.tiers[0,1].append(dict(round=1,support=2,change=.1))
    measurement=SimpleNamespace(pending={},bank={(1,0,17):{'weight':torch.ones(2,2)}},head_hashes={17:'frozen'})
    obs=SimpleNamespace(semantic=semantic,reliability=reliability,measurement=measurement,round_number=1)
    log=SimpleNamespace(next_round=1,hooks=h,rows=[dict(round=1)],total_time=1.,elapsed_before=0.,started=time.perf_counter())
    # Snapshots are taken after seven-model synchronization, as in the real loop.
    for m in models:m.load_state_dict(split_model(w,m.state_dict()))
    identity={'fixed':'protocol'};rng=rng_state();state=recovery_snapshot(log,obs,models,identity)
    atomic_checkpoint(tmp_path/'recovery_latest.pt',state)
    state=torch.load(tmp_path/'recovery_latest.pt',map_location='cpu',weights_only=False)
    expected=h.aggregate(u,n,w);h.rounds_seen=0;obs.semantic.tiers={};obs.reliability.tiers={};obs.measurement.bank={}
    restore_snapshot(state,log,obs,models,identity)
    assert_rng(rng,rng_state());assert h.rounds_seen==1 and obs.round_number==1
    assert obs.semantic.tiers[0,1]['q']==.3 and obs.reliability.tiers[0,1][0]['support']==2
    assert torch.equal(obs.measurement.bank[1,0,17]['weight'],torch.ones(2,2))
    obs.reliability.clients[99].append((2,.2))
    obs.reliability.profiles[99,6].append((2,.2))
    obs.reliability.tiers[2,99].append(dict(round=2,support=2,change=.1))
    assert_states(h.aggregate(u,n,w),expected)
    with pytest.raises(ValueError):restore_snapshot(state,log,obs,models,{'changed':True})


def test_nonneutral_diagnostic_and_controls_are_real_readonly_paths(tmp_path):
    torch.set_num_threads(2);h,u,n,w,models=setup('a07');h.aggregate(u,n,w)
    base=h.aggregate(u,n,w);d=decision();actual,report=fuse(base,w,u,n,h,d,.2,False)
    assert report['difference_from_a07_l2']>0
    users={c:list(range(c*12,(c+1)*12)) for c in range(100)}
    diag=TrainingSideDiagnostic(TrainData(),users,evaluation_indices(TrainData(),users),tmp_path);diag.models=models
    diag.observer=SimpleNamespace(dispatches=[split_model(w,models[p].state_dict()) for c,p in h.slots],
        round_number=2,cfg=config(),fusion=report,observations=[dict(own_gain=.1) for _ in u])
    rng=rng_state();saved=copy.deepcopy(actual);diag.run(base,actual,w,u,n,h,d,False)
    assert_rng(rng,rng_state());assert_states(saved,actual)
    row=json.loads((tmp_path/'iv_e_rounds.jsonl').read_text())
    assert len(row['figure8'])==36 and row['norms']['V2']['difference_from_a07_l2']>0
    assert row['figure7'] and all('source_own_evaluation_gain' in r and 'source_own_holdout_gain' not in r for r in row['figure7'])
