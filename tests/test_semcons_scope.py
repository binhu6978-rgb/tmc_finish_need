import contextlib
import copy
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import semcons_scope as scope
import adapterfl_methods as methods
import Algorithm.Training_AdaptiveFL as training
from models.Fed import Aggregation_AdaptiveFL,split_model,get_model_list
from test_adapterfl_methods import args_for,tiny_models,assert_states,assert_rng
from utils.set_seed import set_random_seed


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


class One(nn.Module):
    def __init__(self,n):
        super().__init__();self.w=nn.Parameter(torch.zeros(n))


def setup(mode='s06',deltas=None,profiles=(0,2,6),cids=(1,2,3),response_values=(1.,-1.,1.),dispatch_values=None):
    models=[One(n) for n in range(1,8)]
    args=args_for(mode,num_classes=2,conflict_lambda=4.)
    hooks=scope.ScopeHooks(args,models);hooks.start_round(0)
    hooks.slots=list(zip(cids,profiles));hooks.observations=[]
    for v in response_values:
        hooks.observations.append({'delta':torch.tensor([[v,-v],[v,-v]]),'counts':torch.tensor([2,2])})
    if deltas is None:deltas=[[1.],[-1.,2.,2.],[1.,3.,3.,4.,4.,4.,4.]]
    sent=[{'w':torch.full((p+1,),float(dispatch_values[i]) if dispatch_values else 0.)} for i,p in enumerate(profiles)]
    uploads=[{'w':sent[i]['w']+torch.tensor(v)} for i,v in enumerate(deltas)]
    return hooks,uploads,[1]*len(profiles),{'w':torch.zeros(7)},sent


@pytest.mark.parametrize('mode',['s03','s06','s07','s09'])
def test_scoped_evidence_does_not_change_unrelated_regions(mode):
    h,u,n,w,s=setup(mode)
    if mode in ('s07','s09'):
        h.history={f'{cid}:{r["id"]}':1 for r in h.regions for cid in [1,2,3]}
    original=copy.deepcopy(w);base=Aggregation_AdaptiveFL(u,n,w)
    out=h.aggregate(u,n,w,s)
    assert_states(w,original)
    assert torch.equal(out['w'][1:],base['w'][1:])
    assert out['w'][0]!=base['w'][0]


def test_global_gate_control_extends_local_evidence():
    local,u,n,w,s=setup('s06');global_h,*_=setup('s05')
    a=local.aggregate(u,n,w,s);b=global_h.aggregate(u,n,w,s)
    base=Aggregation_AdaptiveFL(u,n,w)
    assert torch.equal(a['w'][1:3],base['w'][1:3])
    assert not torch.equal(b['w'][1:3],base['w'][1:3])
    assert torch.equal(b['w'][3:],base['w'][3:])


@pytest.mark.parametrize('mode',[f's{i:02d}' for i in range(3,10)])
def test_one_physical_client_multiple_slots_is_neutral(mode):
    h,u,n,w,s=setup(mode,cids=(7,7,7))
    out=h.aggregate(u,n,w,s)
    assert_states(out,Aggregation_AdaptiveFL(u,n,w))
    assert not h.history


def test_no_semantic_evidence_and_nonconflicting_semantics_fallback():
    h,u,n,w,s=setup('s06',response_values=(1.,1.,1.))
    assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
    h,u,n,w,s=setup('s06')
    h.observations=[{'delta':torch.zeros(2,2),'counts':torch.zeros(2,dtype=torch.long)} for _ in u]
    assert_states(h.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
    assert h.round_diagnostics['semantic_missing_comparison_fraction']==1.


def test_semantic_only_does_not_use_direction_decisions():
    h,u,n,w,s=setup('s04',deltas=[[1.],[2.,2.,2.],[3.]*7],response_values=(1.,2.,-1.))
    out=h.aggregate(u,n,w,s)
    assert out['w'][0]!=Aggregation_AdaptiveFL(u,n,w)['w'][0]
    assert h.round_diagnostics['direction_valid_comparisons']==0


def test_normalization_only_changes_denominator_and_preserves_dispatch_fusion():
    common=dict(deltas=[[1.]*7,[-2.]*7],profiles=(6,6),cids=(1,2),response_values=(1.,-1.),dispatch_values=(10.,30.))
    h7,u,n,w,s=setup('s07',**common);h9,*_=setup('s09',**common)
    for h in (h7,h9):h.history={f'{cid}:{r["id"]}':1 for r in h.regions for cid in [1,2]}
    a=h7.aggregate(u,n,w,s);b=h9.aggregate(u,n,w,s)
    assert_states(a,Aggregation_AdaptiveFL(u,n,w))
    assert torch.allclose(b['w'],torch.full((7,),20.-.5*np.exp(-2)),atol=2e-6)
    for r7,r9 in zip(h7.round_diagnostics['regions'],h9.round_diagnostics['regions']):
        assert r7['gates']==r9['gates']
        assert r9['applied_learning_update_l2']==pytest.approx(r7['applied_learning_update_l2']*r9['gated_mass_ratio'])


def test_history_deduplicates_physical_client_region_events_and_roundtrip():
    h,u,n,w,s=setup('s07',deltas=[[1.]*7,[2.]*7,[-1.]*7],profiles=(6,6,6),cids=(1,1,2),response_values=(1.,1.,-1.))
    h.aggregate(u,n,w,s)
    assert len(h.history)==2*len(h.regions) and set(h.history.values())=={1}
    state=copy.deepcopy(h.state_dict());restored,*_=setup('s07',deltas=[[1.]*7,[2.]*7,[-1.]*7],profiles=(6,6,6),cids=(1,1,2),response_values=(1.,1.,-1.))
    restored.load_state_dict(state,w)
    for hooks in (h,restored):
        hooks.start_round(1);hooks.slots=[(1,6),(1,6),(2,6)]
        hooks.observations=[{'delta':torch.tensor([[v,-v],[v,-v]]),'counts':torch.tensor([2,2])} for v in [1.,1.,-1.]]
    assert_states(h.aggregate(u,n,w,s),restored.aggregate(u,n,w,s))
    assert set(h.history.values())=={2}


def test_negative_control_preserves_validity_score_distribution_and_rng():
    slots=[(i,6) for i in range(4)]
    observations=[{'delta':torch.tensor([[v,-v],[v,-v]]),'counts':torch.tensor([2,2])} for v in [1.,2.,-1.,-.5]]
    before=methods.rng_state()
    actual,a=scope.semantic_evidence(slots,observations,list(range(4)),2)
    shuffled,b=scope.semantic_evidence(slots,observations,list(range(4)),2,shuffle_seed=7)
    assert a['valid_classes']==b['valid_classes'] and sorted(actual.values())==sorted(shuffled.values())
    assert set(actual)==set(shuffled) and b['changed_score_count']>0
    assert_rng(before,methods.rng_state())


class Augmented(torch.utils.data.TensorDataset):
    def __init__(self):
        super().__init__(torch.arange(4*3*4*4).reshape(4,3,4,4).float()/100,torch.tensor([0,1,0,1]))
        self.targets=[0,1,0,1]
    def __getitem__(self,index):
        x,y=super().__getitem__(index)
        return x+random.random()*.01+float(np.random.rand())*.01+torch.rand_like(x)*.01,y


def test_measurement_replays_views_and_preserves_model_and_rng():
    net=tiny_models()[-1];net.train();data=Augmented()
    weights=copy.deepcopy(net.state_dict());modes=[m.training for m in net.modules()]
    before=methods.rng_state();plan=scope.semantic_plan(data,range(4),17,1,2)
    a,c=scope.response(net,data,plan,123,2,2);b,d=scope.response(net,data,plan,123,2,2)
    assert torch.equal(a,b) and torch.equal(c,d) and c.tolist()==[2,2]
    assert_rng(before,methods.rng_state());assert_states(weights,net.state_dict())
    assert modes==[m.training for m in net.modules()]


def test_s00_s02_and_legacy_baseline_full_training_are_identical(tmp_path,monkeypatch):
    saved=[]
    def build(args):
        models=tiny_models();saved.append(models)
        return models,[(.4,2,.001)]*7
    monkeypatch.setattr(training,'get_model_list',build)
    train=Augmented();test=torch.utils.data.TensorDataset(train.tensors[0],train.tensors[1])
    users={i:list(range(4)) for i in range(100)};results={}
    for mode in ['baseline','s00','s02']:
        args=args_for(mode,tmp_path,epochs=3,num_classes=2,conflict_lambda=4.,eval_profiles='full',method_run_name='paired')
        set_random_seed(11)
        with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,train,test,users)
        results[mode]=([copy.deepcopy(m.state_dict()) for m in saved.pop()],methods.rng_state())
        with (tmp_path/mode/'paired/method_diagnostics.csv').open() as f:
            results[mode]+=([json.loads(r['metrics_json'])['slots'] for r in csv.DictReader(f)],)
    for mode in ['s00','s02']:
        for a,b in zip(results['baseline'][0],results[mode][0]):assert_states(a,b)
        assert_rng(results['baseline'][1],results[mode][1])
        assert results['baseline'][2]==results[mode][2]


def test_s01_delegates_to_unchanged_m2():
    h,u,n,w,s=setup('s01')
    models=[One(i) for i in range(1,8)]
    legacy=methods.MethodHooks(args_for('m2',conflict_lambda=4.,num_classes=2),models)
    assert_states(h.aggregate(u,n,w,s),legacy.aggregate(u,n,w,s))


def test_real_seven_geometry_is_disjoint_and_shape_correct():
    args=args_for('s06',conflict_lambda=4.)
    with contextlib.redirect_stdout(io.StringIO()):models,info=get_model_list(args)
    regions=scope.build_regions(models);full=models[-1].state_dict()
    masks={k:torch.zeros_like(v,dtype=torch.int8) for k,v in full.items()}
    for region in regions:
        for cell in region['cells']:
            k=cell['key'];masks[k][scope.slices(cell)]+=1
            expected=[i for i,m in enumerate(models) if all(stop<=m.state_dict()[k].shape[d] for d,(_,stop) in enumerate(cell['bounds']))]
            assert expected==region['profiles']
    assert all(torch.all(m==1) for m in masks.values())
    assert sum(r['numel'] for r in regions)==11424356


def test_invalid_profile_and_dispatch_shapes_are_rejected():
    h,u,n,w,s=setup();h.slots[0]=(1,6)
    with pytest.raises(ValueError,match='mapping'):h.aggregate(u,n,w,s)
    h,u,n,w,s=setup();s[0]={'w':torch.zeros(2)}
    with pytest.raises(ValueError,match='Dispatch'):h.aggregate(u,n,w,s)


def test_scope_s07_interrupted_resume_matches_uninterrupted(tmp_path,monkeypatch):
    saved=[];journals=[]
    def build(args):
        models=tiny_models();saved.append(models);return models,[(.4,2,.001)]*7
    monkeypatch.setattr(training,'get_model_list',build)
    class Journal(methods.RunLog):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);journals.append(self)
    monkeypatch.setattr(methods,'RunLog',Journal)
    train=Augmented();test=torch.utils.data.TensorDataset(train.tensors[0],train.tensors[1]);users={i:list(range(4)) for i in range(100)}
    def args(name,**kw):return args_for('s07',tmp_path,epochs=3,num_classes=2,eval_profiles='full',conflict_lambda=4.,method_run_name=name,**kw)
    set_random_seed(11)
    with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args('reference'),train,test,users)
    expected=[copy.deepcopy(m.state_dict()) for m in saved.pop()];rng=methods.rng_state()
    history=copy.deepcopy(journals[-1].hooks.history);record=Journal.record
    def stop(self,*a,**kw):
        record(self,*a,**kw)
        if self.next_round==1:raise RuntimeError('saved boundary')
    monkeypatch.setattr(Journal,'record',stop);set_random_seed(11)
    with pytest.raises(RuntimeError,match='saved boundary'),contextlib.redirect_stdout(io.StringIO()):
        training.AdaptiveFL(args('resume'),train,test,users)
    journals[-1].accuracy.close();journals[-1].diagnostics.close();saved.pop()
    monkeypatch.setattr(Journal,'record',record)
    with contextlib.redirect_stdout(io.StringIO()):
        training.AdaptiveFL(args('resume',method_resume=str(tmp_path/'s07/resume/latest.pt')),train,test,users)
    for a,m in zip(expected,saved.pop()):assert_states(a,m.state_dict())
    assert_rng(rng,methods.rng_state());assert history==journals[-1].hooks.history


def test_scope_fixed_plan_and_full_only_parameter_injection(tmp_path,monkeypatch):
    import sys
    from tools.run_full_only_sweep import plan,main,ROOT
    from utils.options import args_parser
    config=ROOT/'configs/semcons_scope10_400.json'
    batches=plan(tmp_path,[0,0],sys.executable,config)
    assert [[j['id'] for j in b] for b in batches]==[['S00','S02'],['S01','S03'],['S04','S05'],['S06','S07'],['S08','S09']]
    for job in [j for b in batches for j in b]:
        monkeypatch.setattr(sys,'argv',['test',*job['argv'][3:]])
        args=args_parser();hooks=scope.ScopeHooks(args,[One(i) for i in range(1,8)])
        assert args.epochs==400 and args.eval_profiles=='full' and args.seed==1 and args.generate_data==0
        assert hooks.config['gate_lambda']==4.
    main(['--config',str(config),'--dry-run','--gpus','0,0','--output-root',str(tmp_path/'unused')])
    assert not (tmp_path/'unused').exists()


def test_global_and_local_gate_controls_use_identical_base_evidence():
    a,u,n,w,s=setup('s05');b,*_=setup('s06')
    a.aggregate(u,n,w,s);b.aggregate(u,n,w,s)
    for ra,rb in zip(a.round_diagnostics['regions'],b.round_diagnostics['regions']):
        assert ra.get('direction_risks')==rb.get('direction_risks')
        assert ra.get('semantic_risks')==rb.get('semantic_risks')


def test_s08_validity_with_missing_classes_and_constant_scores():
    slots=[(i,6) for i in range(4)]
    obs=[{'delta':torch.tensor([[v,-v],[v,-v]]),'counts':torch.tensor([2,2])} for v in [1.,2.,-1.,-.5]]
    obs[0]['counts'][1]=0
    a,ma=scope.semantic_evidence(slots,obs,list(range(4)),2)
    b,mb=scope.semantic_evidence(slots,obs,list(range(4)),2,shuffle_seed=11)
    assert sorted(a.values())==sorted(b.values()) and ma['valid_classes']==mb['valid_classes']
    for o in obs:o['delta'].fill_(1.)
    _,info=scope.semantic_evidence(slots,obs,list(range(4)),2,shuffle_seed=11)
    assert info['changed_score_count']==0 and info['permuted_client_count']>0


def test_scope_report_includes_three_references_and_pairing(tmp_path,monkeypatch):
    from test_next10_m2_m3 import write_fake
    import tools.run_full_only_sweep as sweep
    import tools.report_semcons_scope10 as reports
    config=sweep.ROOT/'configs/semcons_scope10_400.json'
    batches=sweep.plan(tmp_path/'data',[0,0],'python',config)
    refs=[]
    for id,peak in [('B00',36.4),('E03',37.29),('N08',37.18)]:
        f=tmp_path/(id+'.csv');write_fake(f,200,peak);refs.append({'id':id,'rounds_csv':str(f)})
    for job in [j for b in batches for j in b]:
        root=Path(job['output_dir']);write_fake(root/'accuracy_each_round.csv',100,38.)
        (root/'manifest.json').write_text(json.dumps({'initial_state_sha256':['same']*7,'structural_regions':[{'id':0,'profiles':list(range(7)),'numel':2}]}))
        with (root/'method_diagnostics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['round','metrics_json']);writer.writeheader()
            for i in range(400):writer.writerow({'round':i+1,'metrics_json':json.dumps({'server':{},'clients':[],'slots':[[1,6],[2,6]]})})
        torch.save({'global':{'w':torch.ones(2)},'rng':methods.rng_state()},root/'latest.pt')
    out=tmp_path/'report';out.mkdir()
    reports.report_scope(batches,out,{'title':'Synthetic scope reporter test','peak_ce_window_radius':2,'references':refs})
    data=json.loads((out/'mechanism_summary.json').read_text())
    assert data['pairing']['S00_S02_final_weights_bitwise_equal'] and data['pairing']['S00_S02_training_rng_equal']
    assert len(data['mechanisms'])==10 and len(list((out/'accuracy_curves').glob('S*.png')))==10
    assert 'N08 delta pp' in (out/'report.md').read_text()


@pytest.mark.parametrize('mode',['s03','s06','s07','s09'])
def test_uncovered_full_tail_preserves_previous_values(mode):
    h,u,n,w,s=setup(mode,deltas=[[1.],[-1.,2.,2.],[1.,3.,3.]],profiles=(0,2,2))
    w={'w':torch.arange(7,dtype=torch.float32)+100}
    if mode in ('s07','s09'):h.history={f'{cid}:{r["id"]}':1 for r in h.regions for cid in [1,2,3]}
    result=h.aggregate(u,n,w,s)
    assert torch.equal(result['w'][3:],w['w'][3:])


def test_s01_full_loop_matches_original_m2(tmp_path,monkeypatch):
    built=[]
    def build(args):
        models=tiny_models();built.append(models);return models,[(.4,2,.001)]*7
    monkeypatch.setattr(training,'get_model_list',build)
    train=Augmented();test=torch.utils.data.TensorDataset(train.tensors[0],train.tensors[1])
    users={i:list(range(4)) for i in range(100)};states=[]
    for mode in ['m2','s01']:
        args=args_for(mode,tmp_path,epochs=3,num_classes=2,conflict_lambda=4.,eval_profiles='full',method_run_name='legacy_pair')
        set_random_seed(11)
        with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,train,test,users)
        states.append(([copy.deepcopy(m.state_dict()) for m in built.pop()],methods.rng_state()))
    for a,b in zip(states[0][0],states[1][0]):assert_states(a,b)
    assert_rng(states[0][1],states[1][1])
