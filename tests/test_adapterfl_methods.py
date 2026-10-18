"""CPU golden-path and aligned-support tests; no real or long GPU training."""
import contextlib
import copy
import csv
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

import adapterfl_methods as methods
import Algorithm.Training_AdaptiveFL as training
from models.Fed import Aggregation_AdaptiveFL, split_model, get_model_list
from models.Update import LocalUpdate_AdaptiveFL
from utils.options import args_parser
from utils.set_seed import set_random_seed


class Tiny(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,channels,1,bias=False),
                                    nn.BatchNorm2d(channels,track_running_stats=False),nn.ReLU())
        self.classifier=nn.Linear(channels,2)

    def forward(self,x):
        return {'output':self.classifier(self.features(x).mean((2,3)))}


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def args_for(method='baseline',root=None,**changes):
    argv=sys.argv
    try:
        sys.argv=['test']
        args=args_parser()
    finally:
        sys.argv=argv
    args.method=method; args.model='resnet'; args.dataset='cifar100'
    args.device=torch.device('cpu'); args.gpu=-1
    args.epochs=1; args.local_ep=1; args.local_bs=2;args.bs=2;args.frac=.02
    args.seed=11; args.generate_data=0
    if root is not None:
        args.method_output_root=str(root)
    for k,v in changes.items():
        setattr(args,k,v)
    return args


def tiny_models():
    return [Tiny(c) for c in range(1,8)]


def filled(model,value):
    return {k:torch.full_like(v,float(value)) for k,v in model.state_dict().items()}


def assert_states(a,b):
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)


def assert_rng(a,b):
    assert a['python']==b['python']
    assert a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1])
    assert a['numpy'][2:]==b['numpy'][2:]
    assert torch.equal(a['torch'],b['torch'])


def load_original():
    spec=importlib.util.spec_from_file_location('frozen_adapterfl',Path(__file__).parent/'fixtures/adapterfl_original_training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('method',[None,'baseline'])
def test_native_full_loop_bitwise_golden(method,tmp_path,monkeypatch):
    """Actual seven ResNets: initialization, dispatch, SGD, sync, all 7 tests."""
    original=load_original()
    args=args_for(method,tmp_path)
    data=torch.utils.data.TensorDataset(torch.arange(4*3*8*8).reshape(4,3,8,8).float()/1000,
                                        torch.tensor([0,1,0,1]))
    users={i:[0,1,2,3] for i in range(100)}
    saved=[]
    def capture(a):
        models,info=get_model_list(a)
        saved.append(models)
        return models,info
    for module in (original,training):
        monkeypatch.setattr(module,'get_model_list',capture)
        monkeypatch.setattr(module,'save_result',lambda *a:None)
        monkeypatch.setattr(module,'my_save_result',lambda *a:'unused')
        monkeypatch.setattr(module,'get_final_acc',lambda *a:None)
    set_random_seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        original.AdaptiveFL(args,data,data,users)
    rng_original=methods.rng_state()
    golden=[methods.cpu_copy(m.state_dict()) for m in saved.pop()]
    set_random_seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        training.AdaptiveFL(args,data,data,users)
    for expected,model in zip(golden,saved.pop()):
        assert_states(expected,model.state_dict())
    assert_rng(rng_original,methods.rng_state())


def test_native_local_step_and_zero_prox_bitwise():
    original=load_original()
    args=args_for('m4',shared_prox_lambda=0.)
    models=tiny_models();base=copy.deepcopy(models[-1])
    data=torch.utils.data.TensorDataset(torch.randn(4,3,4,4),torch.tensor([0,1,1,0]))
    set_random_seed(27)
    reference=LocalUpdate_AdaptiveFL(args,data,list(range(4)))
    expected=original.original_local_train(reference,0,copy.deepcopy(base))
    set_random_seed(27)
    local=LocalUpdate_AdaptiveFL(args,data,list(range(4)))
    methods.MethodHooks(args,models).configure_local(local)
    actual=local.train(0,copy.deepcopy(base))
    assert_states(expected,actual)
    assert not hasattr(local,'adapterfl_prox')


def test_m1_counts_sample_weights_once_and_scales_only_covered():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m1'),models)
    old=filled(models[-1],7.)
    uploads=[filled(models[0],1.),filled(models[2],3.)]
    base=Aggregation_AdaptiveFL(uploads,[2,6],old)
    actual=hooks.aggregate(uploads,[2,6],old)
    changed=False
    for k,v in old.items():
        counts=torch.zeros_like(v)
        for s in uploads:
            counts[tuple(slice(0,d) for d in s[k].shape)]+=1
        active=counts>0
        eta=(counts[active].mean()/counts[active]).pow(.5).clamp(.8,1.2)
        expected=v.clone();expected[active]+=eta*(base[k][active]-v[active])
        assert torch.allclose(actual[k],expected,atol=1e-6)
        assert torch.equal(actual[k][~active],v[~active])
        changed=changed or not torch.equal(actual[k],base[k])
    assert changed
    assert hooks.round_diagnostics


@pytest.mark.parametrize('method,overrides',[
    ('baseline',{}),('m1',{'coverage_gamma':0.}),('m2',{'conflict_lambda':0.}),
    ('m3',{'partition_alpha_shared':1.,'partition_alpha_remainder':1.}),
    ('m4',{'shared_prox_lambda':0.}),('m5',{'partition_norm_multiplier':1e20}),
    ('m6',{'history_beta':0.})])
def test_disabled_strength_returns_exact_native_aggregation(method,overrides):
    models=tiny_models();hooks=methods.MethodHooks(args_for(method,**overrides),models)
    old=filled(models[-1],7.);uploads=[filled(models[0],1.),filled(models[2],3.)]
    sent=[filled(models[0],0.),filled(models[2],0.)]
    expected=Aggregation_AdaptiveFL(uploads,[2,6],old)
    actual=hooks.aggregate(uploads,[2,6],old,sent)
    assert_states(expected,actual)


def test_m2_actual_dispatch_loo_reference_and_coordinate_renormalization():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m2'),models)
    old=filled(models[-1],5.)
    sent=[filled(models[2],10.) for _ in range(3)]
    uploads=[filled(models[2],v) for v in (11.,12.,9.)]
    actual=hooks.aggregate(uploads,[1,1,1],old,sent)
    expected=Aggregation_AdaptiveFL(uploads,[1.,1.,np.exp(-1.)],old)
    assert_states(actual,expected)
    assert hooks.round_diagnostics['cosine']==[1.,None,-1.]
    assert hooks.round_diagnostics['negative_fraction']==pytest.approx(1/3)
    assert actual['features.0.weight'][-1,0,0,0]==5.  # uncovered Full tail


@pytest.mark.parametrize('case',['single','zero','different_shapes','large_lambda'])
def test_m2_edge_cases(case):
    models=tiny_models();args=args_for('m2',conflict_lambda=10000. if case=='large_lambda' else 1.)
    hooks=methods.MethodHooks(args,models);old=filled(models[-1],5.)
    profiles=[0] if case=='single' else ([0,2,4] if case=='different_shapes' else [2,2,2])
    sent=[filled(models[p],10.) for p in profiles]
    values=[10.]*len(profiles) if case=='zero' else [11.,12.,9.][:len(profiles)]
    uploads=[filled(models[p],v) for p,v in zip(profiles,values)]
    actual=hooks.aggregate(uploads,[1]*len(profiles),old,sent)
    assert all(torch.isfinite(v).all() for v in actual.values())
    if case in ('single','zero'):
        assert_states(actual,Aggregation_AdaptiveFL(uploads,[1]*len(profiles),old))


def test_m3_disjoint_regions_and_history_metrics():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m3',partition_alpha_shared=.8,partition_alpha_remainder=1.2),models)
    old=filled(models[-1],7.);uploads=[filled(models[2],3.)]
    base=Aggregation_AdaptiveFL(uploads,[1],old)
    actual=hooks.aggregate(uploads,[1],old)
    for k,v in old.items():
        shared=hooks.region(k,v,'shared');rest=hooks.region(k,v,'remainder')
        assert not torch.any(shared&rest) and torch.all(shared|rest)
        assert torch.allclose(actual[k][shared],v[shared]+.8*(base[k][shared]-v[shared]))
        assert torch.allclose(actual[k][rest],v[rest]+1.2*(base[k][rest]-v[rest]))
    hooks.aggregate(uploads,[1],old)
    assert hooks.round_diagnostics['shared']['relative_norm_fluctuation']==0.


def test_m4_detached_shared_only_normalized_regularizer_and_training_difference():
    models=tiny_models();args=args_for('m4',shared_prox_lambda=10.)
    hooks=methods.MethodHooks(args,models)
    net=copy.deepcopy(models[-1])
    targets,count,strength=methods.prepare_prox(net,(hooks.shared_shapes,10.))
    assert all(not initial.requires_grad for _,initial in targets.values())
    assert count==sum(np.prod(shape) for shape in hooks.shared_shapes.values())
    with torch.no_grad():
        for p in net.parameters():p.add_(1.)
    regularizer=methods.prox_loss(net,targets,count)
    assert float(regularizer)==pytest.approx(.5)
    regularizer.backward()
    for k,p in net.named_parameters():
        shared=hooks.region(k,p,'shared')
        assert torch.all(p.grad[~shared]==0.)
        assert torch.allclose(p.grad[shared],torch.full_like(p.grad[shared],1./count))
    data=torch.utils.data.TensorDataset(torch.randn(4,3,4,4),torch.tensor([0,1,1,0]))
    set_random_seed(8)
    plain=LocalUpdate_AdaptiveFL(args,data,range(4)).train(0,copy.deepcopy(models[-1]))
    set_random_seed(8)
    local=LocalUpdate_AdaptiveFL(args,data,range(4));hooks.configure_local(local)
    constrained=local.train(0,copy.deepcopy(models[-1]))
    assert any(not torch.equal(v,constrained[k]) for k,v in plain.items())
    assert local.adapterfl_metrics['prox_loss_mean']>0


def test_m5_clips_only_matched_support_and_preserves_normal_updates():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m5'),models)
    old=filled(models[-1],7.)
    sent=[filled(models[p],0.) for p in (0,2,4)]
    uploads=[filled(models[p],v) for p,v in zip((0,2,4),(1.,1.,100.))]
    original=copy.deepcopy(uploads)
    actual=hooks.aggregate(uploads,[1,1,1],old,sent)
    expected_last={k:v.clone() for k,v in uploads[-1].items()}
    for k,v in expected_last.items():
        v[hooks.region(k,v,'shared')]=2.
    expected=Aggregation_AdaptiveFL([uploads[0],uploads[1],expected_last],[1,1,1],old)
    for k in actual:assert torch.allclose(actual[k],expected[k],atol=1e-5)
    for before,after in zip(original,uploads):assert_states(before,after)
    assert any(g['fallback'] for g in hooks.round_diagnostics['groups'] if g['partition']=='remainder')
    assert any(g['clipped_fraction']>0 for g in hooks.round_diagnostics['groups'])


def test_m6_gap_and_roundtrip_state(tmp_path):
    models=tiny_models();hooks=methods.MethodHooks(args_for('m6'),models)
    old=filled(models[-1],0.)
    first=hooks.aggregate([filled(models[-1],1.)],[1],old)
    assert_states(first,filled(models[-1],1.))
    small=filled(models[0],3.)
    second=hooks.aggregate([small],[1],first)
    for k,v in first.items():
        shared=hooks.region(k,v,'shared')
        assert torch.allclose(second[k][shared],torch.full_like(v[shared],2.1))
        assert torch.equal(second[k][~shared],v[~shared])
    torch.save(hooks.state_dict(),tmp_path/'state.pt')
    restored=methods.MethodHooks(args_for('m6'),models)
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),second)
    local={k:v+3. for k,v in second.items()}
    a=hooks.aggregate([local],[1],second);b=restored.aggregate([local],[1],second)
    assert_states(a,b)
    for k,v in second.items():
        shared=hooks.region(k,v,'shared')
        assert torch.allclose(a[k][~shared]-v[~shared],torch.full_like(v[~shared],3.))
        assert torch.allclose(a[k][shared]-v[shared],torch.full_like(v[shared],1.29),atol=1e-6)


@pytest.mark.parametrize('method',['baseline','m1','m2','m3','m4','m5','m6'])
def test_all_methods_do_not_mutate_server_and_keep_uncovered(method):
    models=tiny_models();hooks=methods.MethodHooks(args_for(method),models)
    old=filled(models[-1],7.);before=copy.deepcopy(old)
    profiles=(0,1,2)
    uploads=[filled(models[p],v) for p,v in zip(profiles,(1.,2.,8.))]
    sent=[filled(models[p],0.) for p in profiles]
    result=hooks.aggregate(uploads,[1,2,3],old,sent)
    assert_states(old,before)
    for k,v in result.items():
        covered=torch.zeros_like(v,dtype=torch.bool)
        for s in uploads:covered[tuple(slice(0,d) for d in s[k].shape)]=True
        assert torch.equal(v[~covered],old[k][~covered])
        assert torch.isfinite(v).all() and v.shape==old[k].shape


def test_actual_seven_profile_mapping_bn_shortcut_classifier():
    args=args_for()
    with contextlib.redirect_stdout(io.StringIO()):models,info=get_model_list(args)
    assert [(a,b) for a,b,_ in info]==[(.4,2),(.4,3),(.4,4),(.66,2),(.66,3),(.66,4),(1.,2)]
    hooks=methods.MethodHooks(args,models)
    full=models[-1].state_dict()
    assert any('shortcut' in k for k in full) and 'classifier.weight' in full
    for model in models:
        state=model.state_dict();mapped=split_model(full,state)
        assert mapped.keys()==state.keys()
        for k,v in mapped.items():
            assert v.shape==state[k].shape
            assert torch.equal(v,full[k][methods.prefix(full[k],state[k])])
            shared=hooks.region(k,v,'shared');rest=hooks.region(k,v,'remainder')
            assert not torch.any(shared&rest)
        assert not any('running_' in k or 'num_batches' in k for k in state)
    assert sum(np.prod(v) for v in hooks.shared_shapes.values())==2120985


def test_m6_original_loop_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    contexts=[]
    class Journal(methods.RunLog):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);contexts.append(self)
    monkeypatch.setattr(methods,'RunLog',Journal)
    saved=[]
    def build(args):
        models=tiny_models();saved.append(models)
        return models,[(.4,2,.001)]*7
    monkeypatch.setattr(training,'get_model_list',build)
    data=torch.utils.data.TensorDataset(torch.randn(4,3,4,4),torch.tensor([0,1,1,0]))
    users={i:[0,1,2,3] for i in range(100)}
    args=args_for('m6',tmp_path,epochs=3,method_run_name='reference')
    set_random_seed(11)
    with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,data,data,users)
    expected=[methods.cpu_copy(m.state_dict()) for m in saved.pop()]
    expected_rng=methods.rng_state()
    original_record=Journal.record
    def interrupt(self,*a,**kw):
        original_record(self,*a,**kw)
        if self.next_round==1:raise RuntimeError('deliberate stop after saved boundary')
    monkeypatch.setattr(Journal,'record',interrupt)
    args=args_for('m6',tmp_path,epochs=3,method_run_name='resume')
    set_random_seed(11)
    with pytest.raises(RuntimeError,match='deliberate stop'),contextlib.redirect_stdout(io.StringIO()):
        training.AdaptiveFL(args,data,data,users)
    contexts[-1].accuracy.close();contexts[-1].diagnostics.close()
    saved.pop()
    monkeypatch.setattr(Journal,'record',original_record)
    args=args_for('m6',tmp_path,epochs=3,method_resume=str(tmp_path/'m6/resume/latest.pt'))
    with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,data,data,users)
    for state,model in zip(expected,saved.pop()):assert_states(state,model.state_dict())
    assert_rng(expected_rng,methods.rng_state())
    with (tmp_path/'m6/resume/accuracy_each_round.csv').open() as f:rows=list(csv.DictReader(f))
    assert [int(r['round']) for r in rows]==[1,2,3]


def test_invalid_combinations_and_history_rejected():
    models=tiny_models()
    with pytest.raises(ValueError):methods.MethodHooks(args_for('m1+m6'),models)
    hooks=methods.MethodHooks(args_for('m6'),models)
    hooks.aggregate([filled(models[-1],1.)],[1],filled(models[-1],0.))
    state=hooks.state_dict();state['velocity']['features.0.weight']=torch.zeros(1)
    with pytest.raises(ValueError):hooks.load_state_dict(state,models[-1].state_dict())


@pytest.mark.parametrize('method',['baseline','m1','m2','m3','m4','m5','m6'])
def test_all_method_state_roundtrips(method,tmp_path):
    models=tiny_models();args=args_for(method)
    hooks=methods.MethodHooks(args,models)
    old=filled(models[-1],5.)
    uploads=[filled(models[p],v) for p,v in zip((0,2,4),(1.,2.,8.))]
    sent=[filled(models[p],0.) for p in (0,2,4)]
    current=hooks.aggregate(uploads,[1,2,3],old,sent)
    torch.save(hooks.state_dict(),tmp_path/'method.pt')
    restored=methods.MethodHooks(args,models)
    restored.load_state_dict(torch.load(tmp_path/'method.pt',weights_only=True),current)
    a=hooks.aggregate(uploads,[1,2,3],current,sent)
    b=restored.aggregate(uploads,[1,2,3],current,sent)
    assert_states(a,b)


@pytest.mark.parametrize('method',['m1','m2','m3','m4','m5','m6'])
def test_all_hooks_real_seven_profile_shapes_on_cpu(method):
    args=args_for(method)
    with contextlib.redirect_stdout(io.StringIO()):models,_=get_model_list(args)
    hooks=methods.MethodHooks(args,models)
    old=methods.cpu_copy(models[-1].state_dict())
    sent=[methods.cpu_copy(models[p].state_dict()) for p in (0,3,6)]
    uploads=[{k:v+.001*(i+1) for k,v in state.items()} for i,state in enumerate(sent)]
    output=hooks.aggregate(uploads,[123,234,345],old,sent)
    for model in models:
        mapped=split_model(output,model.state_dict())
        model.load_state_dict(mapped,strict=True)
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())


def test_review_m2_common_fraction_and_exclusive_weight_cancellation():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m2'),models)
    old=filled(models[-1],5.)
    profiles=(0,0,6)
    sent=[filled(models[p],10.) for p in profiles]
    uploads=[filled(models[p],v) for p,v in zip(profiles,(11.,12.,9.))]
    base=Aggregation_AdaptiveFL(uploads,[1,1,1],old)
    actual=hooks.aggregate(uploads,[1,1,1],old,sent)
    diag=hooks.round_diagnostics
    assert diag['common_support_fraction_full']==pytest.approx(sum(v.numel() for v in sent[0].values())/sum(v.numel() for v in old.values()))
    assert diag['exclusive_coordinates_per_slot'][:2]==[0,0]
    assert diag['exclusive_coordinates_per_slot'][2]>0
    assert diag['weight_distribution']['min']<1 and diag['negative_fraction']>0
    assert not torch.equal(actual['features.0.weight'][0],base['features.0.weight'][0])
    # Full-only holder has a negative shared direction, but its normalized
    # exclusive-region coefficient remains exactly one mathematically.
    assert torch.allclose(actual['features.0.weight'][1:],uploads[2]['features.0.weight'][1:],atol=2e-6)


def test_review_m4_gradient_measurements_and_diagnostics_do_not_change_update():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m4'),models)
    net=copy.deepcopy(models[-1]);targets,count,mu=methods.prepare_prox(net,(hooks.shared_shapes,1.))
    with torch.no_grad():
        for p in net.parameters():p.add_(.01)
    ce=nn.functional.cross_entropy(net(torch.randn(2,3,4,4))['output'],torch.tensor([0,1]))
    penalty=methods.prox_loss(net,targets,count)
    sample=methods.prox_gradient_diagnostics(net,ce,penalty,targets,count,mu)
    prox_grad=torch.autograd.grad(penalty,list(net.parameters()),retain_graph=True)
    assert sample['proximal_gradient_l2']==pytest.approx(sum(float(g.double().square().sum()) for g in prox_grad)**.5,rel=1e-6)
    assert sample['weighted_prox_to_task_loss_ratio']==pytest.approx(float(penalty.detach()/ce.detach()))
    assert all(p.grad is None for p in net.parameters())
    data=torch.utils.data.TensorDataset(torch.randn(4,3,4,4),torch.tensor([0,1,1,0]))
    states=[];rngs=[]
    for diagnostic_steps in (0,2):
        set_random_seed(11)
        local=LocalUpdate_AdaptiveFL(args_for('m4'),data,range(4));hooks.configure_local(local)
        local.adapterfl_prox_diagnostic_steps=diagnostic_steps
        states.append(local.train(0,copy.deepcopy(models[-1])));rngs.append(methods.rng_state())
    assert_states(states[0],states[1]);assert_rng(rngs[0],rngs[1])


def test_review_m5_nonactivation_warning_and_fallback_survive_checkpoint(tmp_path):
    models=tiny_models();args=args_for('m5');hooks=methods.MethodHooks(args,models)
    old=filled(models[-1],5.);profiles=(0,2,4)
    sent=[filled(models[p],0.) for p in profiles]
    uploads=[filled(models[p],1.) for p in profiles]
    for _ in range(2):hooks.aggregate(uploads,[1,1,1],old,sent)
    torch.save(hooks.state_dict(),tmp_path/'state.pt')
    restored=methods.MethodHooks(args,models)
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),old)
    actual=restored.aggregate(uploads,[1,1,1],old,sent)
    assert_states(actual,Aggregation_AdaptiveFL(uploads,[1,1,1],old))
    diag=restored.round_diagnostics
    assert diag['partition_totals']['shared']['actual_clip_rate']==0
    assert diag['partition_totals']['remainder']['insufficient_slots_fallback_fraction']==1
    assert 'NO_ACTUAL_CLIPPING_FOR_AT_LEAST_3_ROUNDS' in diag['warning_codes']
    assert diag['rounds_without_actual_clip']==3
    uploads[-1]=filled(models[4],100.)
    restored.aggregate(uploads,[1,1,1],old,sent)
    assert restored.round_diagnostics['actual_changed_upload_slot_fraction']==pytest.approx(1/3)
    assert not restored.round_diagnostics['warning_codes']


def test_review_m6_initial_delta_and_beta_component_magnitude():
    models=tiny_models();hooks=methods.MethodHooks(args_for('m6'),models)
    old=filled(models[-1],0.)
    first=hooks.aggregate([filled(models[-1],10.)],[1],old)
    initial=hooks.round_diagnostics['first_round_initialization_delta_l2']
    assert hooks.round_diagnostics['first_observation']
    assert hooks.round_diagnostics['history_component_l2']==0
    assert hooks.round_diagnostics['difference_from_native_aggregate_l2']==0
    local={k:v+1 for k,v in first.items()}
    hooks.aggregate([local],[1],first)
    diag=hooks.round_diagnostics
    assert not diag['first_observation']
    assert diag['first_round_initialization_delta_l2']==initial
    assert diag['history_to_current_component_ratio']==pytest.approx(90.,rel=1e-6)
    assert diag['applied_to_raw_update_ratio']==pytest.approx(9.1,rel=1e-6)
    assert diag['difference_from_native_aggregate_l2']>0
