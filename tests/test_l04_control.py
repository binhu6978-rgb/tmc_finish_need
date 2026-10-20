"""Narrow controls for delayed classifier protection and strict execution."""
import copy
import os
import subprocess
import sys

import torch

from local12_aggregation import Local12Hooks
from test_adapterfl_methods import args_for,assert_states
from test_local12_aggregation import setup


def test_delayed_l04_first_exact_e03_then_original_l04_and_restore():
    _,models,u,n,w,s,old,e03=setup('l04')
    args=args_for('l04',conflict_lambda=4.,classifier_protection_start_round=2)
    delayed=Local12Hooks(args,models)
    assert_states(delayed.aggregate(u,n,w,s),e03)
    assert delayed.round_diagnostics['classifier_protection_active'] is False
    state=copy.deepcopy(delayed.state_dict())
    restored=Local12Hooks(args,models);restored.load_state_dict(state,w)
    ordinary=Local12Hooks(args_for('l04',conflict_lambda=4.),models)
    expected=ordinary.aggregate(u,n,w,s)
    actual=delayed.aggregate(u,n,w,s)
    assert_states(actual,expected)
    assert_states(restored.aggregate(u,n,w,s),actual)
    assert delayed.round_diagnostics['classifier_protection_active'] is True
    for k in w:
        if not k.startswith('classifier.'):assert torch.equal(actual[k],e03[k])
    assert delayed.round_diagnostics['conflict_weights']==old.round_diagnostics['conflict_weights']


def test_policy_opt_in_and_strict_flags_in_fresh_process():
    code='''
from utils.reproducibility import configure_deterministic_execution,execution_policy
before=execution_policy()
assert configure_deterministic_execution(False) is None
assert execution_policy()==before
p=configure_deterministic_execution(True)
assert p['deterministic_algorithms'] and p['cudnn_deterministic']
assert not p['warn_only'] and not p['cudnn_benchmark']
assert not p['cudnn_allow_tf32'] and not p['matmul_allow_tf32']
assert p['cublas_workspace_config']==':4096:8'
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=os.environ.copy())
    assert result.returncode==0,result.stderr


def test_three_commands_are_same_protocol_and_only_classifier_activation_differs(tmp_path):
    from tools.run_l04_control import command
    a=command('python',tmp_path,tmp_path,'R0','m2')
    b=command('python',tmp_path,tmp_path,'R1','l04')
    c=command('python',tmp_path,tmp_path,'R2','l04',start=2)
    for cmd in (a,b,c):
        for key,value in [('--epochs','400'),('--seed','1'),('--conflict-lambda','4'),('--eval-profiles','full')]:
            assert cmd[cmd.index(key)+1]==value
        assert '--deterministic-execution' in cmd
    assert '--classifier-protection-start-round' not in b
    assert c[-3:]==['--classifier-protection-start-round','2','--deterministic-execution']


def test_repro_comparison_requires_weights_rng_slots_and_eval(tmp_path):
    import json
    from tools.check_e03_reproducibility import compare
    left=tmp_path/'a';right=tmp_path/'b';left.mkdir();right.mkdir()
    state=dict(full={'x':torch.ones(2)},rng={'torch':torch.tensor([1],dtype=torch.uint8)},evaluation={6:(1.,2.)},slots=[(1,6)])
    for i in range(1,4):
        torch.save(state,left/f'round{i}.pt');torch.save(state,right/f'round{i}.pt')
    target=tmp_path/'comparison.json';assert compare(left,right,target)['passed']
    state['rng']['torch'][0]=2;torch.save(state,right/'round3.pt')
    import pytest
    with pytest.raises(RuntimeError):compare(left,right,target)
    assert json.loads(target.read_text())['passed'] is False
