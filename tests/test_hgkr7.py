"""Necessary seven-tier mapping, selection, aggregation, and launch checks."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from experiments.baseline import content_hash, state_hash
from experiments.run_hgkr7 import evaluate_full, parse
from experiments.summarize_hgkr7 import load_suite, summarize
from hgkr.core import Config, KernelHistory, anchored_aggregate, clone_state, reset_copy
from hgkr.protocol import allocation_cost, model_costs
from hgkr.seven import (MODES, PROFILES, allowed_profiles, choose, initialize_models, make_schedule,
                        run_round, shared_specs)
from models.Fed import Aggregation_AdaptiveFL, split_model
from semconsfl.aggregation import coverage
from semconsfl.measurement import capture_rng, restore_rng

torch.set_num_threads(1)


def args(device='cpu'):
    return SimpleNamespace(width_ration=[.4,.66,1.],depth_saved=[2,3,4],model='resnet',num_channels=3,
        num_classes=100,device=torch.device(device),seed=1,local_ep=5,local_bs=50,optimizer='sgd',
        lr=.01,lr_decay=.998,momentum=.5,weight_decay=1e-4,dataset='cifar100')


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(1); return initialize_models(args())[0]


def test_actual_seven_profiles_intersection_and_non_monotone_cost(models):
    assert len(models)==7
    assert [sum(p.numel() for p in m.parameters()) for m in models]==[2120985,2646356,4647740,5159621,5528077,6945581,11424356]
    masks,stat=coverage(models)
    assert stat['all_profiles']==2120985 and stat['full_only']==4478775
    specs=shared_specs(models)
    for k,shape in specs.items():
        assert shape==tuple(min(m.state_dict()[k].shape[d] for m in models) for d in range(4))
        assert torch.all(masks[k][:shape[0],:shape[1]]==7)
    costs=model_costs(models)
    assert costs[2]['forward_mac']>costs[3]['forward_mac']
    assert sum(int(.01*s[0]) for s in specs.values())==15


def scores(valid=True):
    return {'conv.weight':dict(score=torch.tensor([.1,.9,.3,.7]),valid=torch.tensor([valid]*4),
                               observations=torch.tensor([3]*4),covered_rounds=3,ready=True)}


def test_six_modes_selection_budget_shared_concentration_ranking_and_rng():
    config=replace(Config(),reset_fraction=.25)
    before=capture_rng(); sc=scores()
    choices={m:[choose(sc,config,m,1,4,i)[0] for i in range(10)] for m in MODES}
    assert all(not x['conv.weight'] for m in ('C0','C1') for x in choices[m])
    assert all(len(x['conv.weight'])==1 for m in ('C2','C3','C4','C5') for x in choices[m])
    assert len({tuple(x['conv.weight']) for x in choices['C2']})>1
    for mode in ('C3','C4','C5'):
        assert all(x==choices[mode][0] for x in choices[mode])
    assert choices['C4'][0]['conv.weight']==[1]
    assert choices['C5'][0]['conv.weight']==[0]
    assert torch.equal(before['torch'],capture_rng()['torch'])
    assert before['python']==capture_rng()['python']
    for mode in ('C4','C5'):
        assert choose(scores(False),config,mode,1,4,9)[0]==choose(scores(False),config,'C3',1,4,0)[0]


@pytest.mark.parametrize('profile',range(7))
def test_reset_maps_only_common_segments_in_every_real_profile(models,profile):
    specs=shared_specs(models); old=clone_state(models[-1].state_dict()); server_hash=state_hash(old)
    local=clone_state(models[profile].state_dict()); before=clone_state(local)
    selected={k:[0] for k in specs}
    reset_copy(local,old,specs,selected,1,4,0)
    for k,t in local.items():
        if k not in specs: assert torch.equal(t,before[k]); continue
        mask=torch.zeros_like(t,dtype=torch.bool); mask[0,:specs[k][1]]=True
        assert torch.equal(t[~mask],before[k][~mask])
    assert state_hash(old)==server_hash
    # The exact dispatch-anchor subtraction must cancel all untrained reset values.
    result=anchored_aggregate(old,[local],[local],[500],4,True)
    assert all(torch.equal(result[k],v) for k,v in old.items())


def test_schedule_all_six_pairing_capabilities_and_budget(models):
    users={c:list(range(262+c*4)) for c in range(100)}
    costs=model_costs(models); config=Config()
    a=make_schedule(1,8,users,costs,config); b=make_schedule(1,8,users,costs,config)
    assert content_hash(a)==content_hash(b)
    assert allowed_profiles(0)==(0,1,2) and allowed_profiles(40)==tuple(range(6)) and allowed_profiles(70)==tuple(range(7))
    for entry in a:
        assert len(set(entry['clients']))==10
        for key in ('random','priority'):
            assert all(p in allowed_profiles(c) for c,p in zip(entry['clients'],entry[key]))
            cost=allocation_cost(entry['clients'],entry[key],users,costs)
            assert all(x<=ceiling+1 for x,ceiling in zip(cost,entry['ceiling']))
        assert all(abs(x-y)<=.02*y+1 for x,y in zip(entry['random_cost'],entry['priority_cost']))


@pytest.mark.parametrize('mode',MODES)
def test_real_short_training_and_mature_reset(models,mode):
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    a=args(device); group=copy.deepcopy(models)
    for m in group: m.to(a.device)
    old=clone_state(group[-1].state_dict())
    for m in group: m.load_state_dict(split_model(old,m.state_dict()))
    h=KernelHistory(shared_specs(group),Config()); state=clone_state(old)
    for r in range(4):
        new={k:v+(.001 if r%2 else -.001) for k,v in state.items()}; h.update(r,state,new); state=new
    dataset=TensorDataset(torch.randn(50,3,32,32),torch.arange(50)%100)
    masks,_=coverage(group)
    row=run_round(group,dataset,{70:list(range(50))},{'round':4,'clients':[70],
                  'random':[6],'priority':[6]},a,Config(),h,mode,masks)
    assert row['reset_total']==(0 if mode in ('C0','C1') else 15)
    assert row['slots'][0]['learning_delta_l2']>0
    assert h.last_round==4


def test_no_reset_round_zero_is_original_golden(models):
    old=clone_state(models[-1].state_dict()); starts=[clone_state(m.state_dict()) for m in models]
    uploads=[{k:v+.01 for k,v in s.items()} for s in starts]
    result=anchored_aggregate(old,uploads,starts,[1]*7,0,False)
    expected=Aggregation_AdaptiveFL(uploads,[1]*7,old)
    assert all(torch.equal(result[k],expected[k]) for k in old)


def test_fixed_formal_interface_and_no_resume():
    for mode in MODES:
        opt,a,c=parse(['--mode',mode,'--gpu','-1','--prepare','--output-dir','unused'])
        assert a.epochs==300 and a.seed==1 and a.local_ep==5 and a.local_bs==50
    with pytest.raises(SystemExit): parse(['--output-dir','unused','--resume'])
    with pytest.raises(SystemExit): parse(['--output-dir','unused','--epochs','500'])


def test_partial_run_never_marked_complete(tmp_path):
    p=tmp_path/'C0'; p.mkdir()
    (p/'manifest.json').write_text(json.dumps({'mode':'C0','formal_300':True}))
    (p/'completed.json').write_text(json.dumps({'rounds_completed':300,'formal_300_complete':True}))
    (p/'rounds.jsonl').write_text('\n'.join(json.dumps({'round':i}) for i in range(299)))
    with pytest.raises(ValueError,match='actually complete'): load_suite(tmp_path)
    assert not (tmp_path/'SUCCESS.json').exists()


def test_summary_peak_and_tail_metrics_without_training(monkeypatch,tmp_path):
    import experiments.summarize_hgkr7 as module
    fake={}
    for i,mode in enumerate(MODES):
        rows=[]
        for r in range(300):
            rows.append({'full_test':dict(profile='P6',accuracy=float(r+i),ce=1.),
                'profile_slots':[0]*6+[10], 'allocation':{'random_cost':[100,100],'priority_cost':[100,100]},
                'slots':[{'client':70,'profile':6,'learning_delta_l2':1.,'kernels':{}}],
                'reset_total':0,'reset_unique_segments':0,'round':r,
                'coverage_update_diagnostics':{'full_only':{'update_l2':.1}},
                'timings':dict(local_train_seconds=1.,selection_reset_seconds=0.,server_seconds=.1)})
        fake[mode]={'rows':rows,'manifest':{'protocol':{}},'completed':{'elapsed_seconds':300}}
    monkeypatch.setattr(module,'load_suite',lambda root:fake)
    obj=summarize(tmp_path)
    assert obj['arms']['C0']['full']==dict(test_oracle_peak=299.,peak_round=300,round300=299.,last20_mean=289.5,last50_mean=274.5)
    assert obj['arms']['C0']['Full_evaluation_count']==300
    assert all('profiles' not in a for a in obj['arms'].values())
    assert not (tmp_path/'profile_accuracy.csv').exists()
    assert len((tmp_path/'full_accuracy_each_round.csv').read_text().splitlines())==1801
    assert 'test-oracle peak' in (tmp_path/'report.md').read_text(encoding='utf-8')


def test_original_Full_test_only_no_grad_and_rng_isolated(models):
    from experiments.baseline import state_hash
    dataset=TensorDataset(torch.randn(4,3,32,32),torch.arange(4))
    a=args(); a.gpu=-1; a.bs=128; a.verbose=False
    modes=[m.training for m in models]; hashes=[state_hash(m.state_dict()) for m in models]
    calls=[]; handles=[]
    for p,m in enumerate(models):
        handles.append(m.register_forward_hook(lambda m,x,y,p=p:calls.append((p,len(x[0]),m.training,torch.is_grad_enabled()))))
    before=capture_rng()
    try: result=evaluate_full(models,dataset,a)
    finally:
        for h in handles: h.remove()
    after=capture_rng()
    assert result['profile']=='P6' and set(result)=={'profile','accuracy','ce'}
    assert calls==[(6,4,False,False)]
    assert [m.training for m in models]==modes
    assert torch.equal(before['torch'],after['torch']) and before['python']==after['python']
    assert all(state_hash(m.state_dict())==h for m,h in zip(models,hashes))


@pytest.mark.parametrize('mode',MODES)
def test_formal_control_loop_exactly_300_Full_tests(monkeypatch,tmp_path,mode):
    # A synthetic control-flow test, not 300 rounds of performance training.
    import experiments.run_hgkr7 as module
    a=args(); a.gpu=-1; a.epochs=300
    opt=SimpleNamespace(mode=mode,prepare=False,plan=None,smoke_rounds=None,
                        output_dir=str(tmp_path/mode),dataset_root='unused')
    group=[torch.nn.Linear(1,1) for _ in range(7)]
    plan={'schedule':[{'round':r} for r in range(300)],
          'evaluation':{'profiles':['P6'],'frequency':'every_round','batch_size':128,'rng':'isolated_from_training'}}
    monkeypatch.setattr(module,'parse',lambda argv:(opt,a,Config()))
    monkeypatch.setattr(module,'load_data',lambda root:(SimpleNamespace(targets=[]),None))
    monkeypatch.setattr(module,'load_partition',lambda *args:({},{}))
    monkeypatch.setattr(module,'initialize_models',lambda args:(group,[]))
    monkeypatch.setattr(module,'prepare',lambda *args:plan)
    monkeypatch.setattr(module,'shared_specs',lambda models:{})
    monkeypatch.setattr(module,'coverage',lambda models:({},{}))
    training_calls=[]; test_calls=[]
    def round_stub(models,train,users,entry,*args):
        assert len(models)==7
        training_calls.append(entry['round'])
        return {'round':entry['round'],'timings':{},'reset_total':0}
    def test_stub(model,data,args):
        assert model is group[6]
        test_calls.append(model)
        return len(test_calls)/10,1.
    monkeypatch.setattr(module,'run_round',round_stub)
    monkeypatch.setattr(module,'test_img',test_stub)
    monkeypatch.setattr(module,'save_weights',lambda *args:None)
    module.main([])
    rows=[json.loads(x) for x in (tmp_path/mode/'rounds.jsonl').read_text().splitlines()]
    assert training_calls==list(range(300)) and len(test_calls)==300
    assert all('official_test' not in r and r['full_test']['profile']=='P6' for r in rows)


def test_shell_stops_on_failed_arm_and_never_runs_next_batch(tmp_path):
    bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '/no-bash')
    if not bash.exists(): pytest.skip('Bash not available')
    shutil.copyfile('run_7tier_300.sh',tmp_path/'run_7tier_300.sh')
    fake=tmp_path/'fakepython'
    fake.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then echo 1; exit 0; fi
case "$2" in
  experiments.hgkr7_preflight) exit 0;;
  experiments.run_hgkr7)
    if [[ " $* " == *" --prepare "* ]]; then exit 0; fi
    if [[ " $* " == *" --mode C0 "* ]]; then echo synthetic-failure >&2; exit 7; fi
    if [[ " $* " == *" --mode C1 "* ]]; then sleep 3; exit 0; fi
    touch unexpected_later_batch; exit 0;;
  *) touch unexpected_summary; exit 0;;
esac
''',encoding='utf-8')
    fake.chmod(0o755)
    env=dict(os.environ,PYTHON='./fakepython',GPUS='0,0')
    env.pop('OUTPUT_ROOT',None)
    run=subprocess.run([str(bash),'run_7tier_300.sh'],cwd=tmp_path,env=env,capture_output=True,timeout=20)
    assert run.returncode!=0
    assert (tmp_path/'runs/hgkr7_300_seed1/FAILED.json').exists()
    assert not (tmp_path/'unexpected_later_batch').exists()
    assert not (tmp_path/'unexpected_summary').exists()
