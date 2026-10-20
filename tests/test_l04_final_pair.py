"""First-round classifier-only intervention and unchanged subsequent M2."""
import copy

import torch

from adapterfl_methods import MethodHooks
from local12_aggregation import Local12Hooks
from models.Fed import Aggregation_AdaptiveFL
from test_adapterfl_methods import args_for,assert_states
from test_local12_aggregation import setup


def test_r3_first_round_l04_rest_m2_with_original_scores_and_checkpoint():
    _,models,u,n,w,s,old,e03=setup('l04')
    args=args_for('l04',conflict_lambda=4.,classifier_protection_end_round=1)
    r3=Local12Hooks(args,models)
    snapshot=copy.deepcopy((u,w,s))
    native=Aggregation_AdaptiveFL(u,n,w)
    first=r3.aggregate(u,n,w,s)
    for k in w:
        assert torch.equal(first[k],native[k] if k.startswith('classifier.') else e03[k])
    assert r3.round_diagnostics['conflict_weights']==old.round_diagnostics['conflict_weights']
    assert r3.round_diagnostics['cosine']==old.round_diagnostics['cosine']
    assert r3.round_diagnostics['classifier_protection_active'] is True
    assert r3.round_diagnostics['difference_from_e03_l2']>0
    saved=copy.deepcopy(r3.state_dict());restored=Local12Hooks(args,models);restored.load_state_dict(saved,w)
    for hook in (r3,restored):
        for _ in range(3):
            assert_states(hook.aggregate(u,n,w,s),e03)
            assert hook.round_diagnostics['classifier_protection_active'] is False
            assert hook.round_diagnostics['difference_from_e03_l2']==0
    for a,b in zip(u,snapshot[0]):assert_states(a,b)
    assert_states(w,snapshot[1])
    for a,b in zip(s,snapshot[2]):assert_states(a,b)


def test_b00_d_is_native_and_both_commands_are_strict_400(tmp_path):
    _,models,u,n,w,s,_,_=setup('l04')
    native=MethodHooks(args_for('baseline',conflict_lambda=4.),models)
    assert_states(native.aggregate(u,n,w,s),Aggregation_AdaptiveFL(u,n,w))
    from tools.run_l04_final_pair import jobs
    tasks=jobs('python',tmp_path,tmp_path)
    assert [t['id'] for t in tasks]==['B00-D','R3']
    for task in tasks:
        assert '--deterministic-execution' in task['argv']
        assert task['argv'][task['argv'].index('--epochs')+1]=='400'
        assert task['argv'][task['argv'].index('--seed')+1]=='1'
        assert task['argv'][task['argv'].index('--eval-profiles')+1]=='full'
    assert tasks[0]['argv'][tasks[0]['argv'].index('--method')+1]=='baseline'
    assert tasks[1]['argv'][-2:]==['--classifier-protection-end-round','1']


def test_factorial_contrasts_separate_initial_later_and_interaction():
    from tools.report_l04_control import factorial_contrasts
    rows=[dict(id=name,**{k:v for k in ('test_oracle_peak','final','last20','last50','final_ce')})
        for name,v in [('R0',10),('R3',11),('R2',12),('R1',14),('B00-D',9)]]
    c=factorial_contrasts(rows)['last20']
    assert c==dict(initial_native_without_later=1,initial_native_with_later=2,
        later_native_without_initial=2,later_native_with_initial=3,interaction=1,m2_minus_native=1)
