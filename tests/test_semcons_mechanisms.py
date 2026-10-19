import copy
import json

import torch

from adapterfl_methods import rng_state
from semcons_scope import ScopeHooks
from test_adapterfl_methods import assert_states, assert_rng
from test_semcons_scope import setup
from tools.probe_semcons_directions import observed, tensor_observation, references
from tools.analyze_semcons_mechanisms import m2_summary,history_summary,semantic_summary,tensor_summary,within_context_correlation


def test_tensor_observer_preserves_aggregate_history_rng(tmp_path):
    h,u,n,w,s = setup('s07')
    h.history = {f'{cid}:{r["id"]}':2 for r in h.regions for cid in (1,2,3)}
    reference = copy.deepcopy(h); before = rng_state()
    expected = reference.aggregate(u,n,w,s)
    actual = observed(ScopeHooks.aggregate,tmp_path/'tensor.jsonl')(h,u,n,w,s)
    assert_states(actual,expected); assert_rng(before,rng_state())
    assert h.history == reference.history and h.round_diagnostics == reference.round_diagnostics
    records = json.loads((tmp_path/'tensor.jsonl').read_text())['records']
    assert records and all(abs(sum(c['dot'] for c in r['cells'])-r['dot'])<1e-10 for r in records)


def test_component_negative_hidden_by_positive_region():
    h,u,n,w,s = setup('s07', profiles=(2,2),cids=(1,2),deltas=[[10.,1.,1.],[10.,-1.,1.]],response_values=(1.,1.))
    # Toy additive decomposition; real geometry is tested separately by test_semcons_scope.
    h.regions = [{'id':0,'profiles':[2],'numel':3,
                  'cells':[{'key':'w','bounds':((j,j+1),),'numel':1} for j in range(3)]}]
    records = tensor_observation(h,u,n,s)
    assert records[0]['cosine'] > 0
    assert any(c['cosine']<0 for c in records[0]['cells'])


def test_repeated_client_reference_excludes_all_own_slots():
    h,*_ = setup('s07',cids=(1,1,2))
    values={0:torch.tensor([1.],dtype=torch.float64),1:torch.tensor([3.],dtype=torch.float64),2:torch.tensor([-1.],dtype=torch.float64)}
    ref=references(h,values,[2,2,5])
    assert ref[0].item()==ref[1].item()==-1
    assert ref[2].item()==2


def test_m2_nonuniform_weights_do_not_claim_measured_parameter_effect():
    row={'round':1,'slots':[[1,0],[1,0],[2,0]],'server':{
        'cosine':[-.1,None,.2],'conflict_weights':[.7,1.,1.], 'common_support_fraction_full':.2}}
    result=m2_summary([row])
    assert result['negative_fraction']==.5 and result['active_weight_count']==1
    assert result['duplicate_client_rounds']==1
    assert 'actual_changed_coordinates' not in result


def test_history_reconstruction_and_sparse_participation_categories():
    h,u,n,w,s=setup('s07');h.aggregate(u,n,w,s)
    row={'round':1,'server':h.round_diagnostics,'slots':h.slots,'clients':[]}
    result=history_summary([row])
    assert result['history_reconstruction_matches']
    assert result['counts']['joint_cold_slots']>0
    assert all(r['category']=='first_client_participation' for r in result['joint_events'])


def test_positive_semantic_disagreement_can_coexist_with_own_gain():
    row={'round':1,'slots':[[7,0]],'clients':[{'functional_class_changes':[{
        'class':0,'true_class_probability_change':.1,'response_delta_l2':.2}]}],
        'server':{'regions':[{'region_id':0,'support_profiles':[0],
        'semantic':{'scores':{'7':.2},'valid_classes':{'7':[0]}}}]}}
    result=semantic_summary([row])
    assert result['positive_q_and_positive_own_probability_gain']==1


def test_tensor_statistics_do_not_equate_negative_cells_with_harm():
    cell={'cosine':-.2,'dot':-1.,'delta_square':2.,'reference_square':3.,'family':'conv'}
    cell2=dict(cell,cosine=.8,dot=9.,delta_square=8.)
    row={'round':2,'records':[{'region':0,'cosine':.5,'cells':[cell,cell2]}]}
    result=tensor_summary([row])['post_initialization_rounds']
    assert result['negative_regions']==0 and result['negative_tensor_cells']==1
    assert result['hidden_negative_delta_energy_fraction']==.2


def test_context_centering_removes_between_round_confound():
    rows=[{'round':r,'region':0,'q':10*r+i,'q_original':-10*r-i,
           'local_true_probability_gain':-10*r+i} for r in (0,1) for i in range(3)]
    result=within_context_correlation(rows,('round','region'))
    assert abs(result['pearson']-1)<1e-12 and result['records']==6
    assert abs(within_context_correlation(rows,('round','region'),'q_original')['pearson']+1)<1e-12


def test_same_tensor_rectangles_are_joined_before_counting_conflicts():
    a={'key':'conv.weight','cosine':-.5,'dot':-1.,'delta_square':2.,'reference_square':2.,'family':'conv'}
    b=dict(a,cosine=1.,dot=9.,delta_square=9.,reference_square=9.)
    result=tensor_summary([{'round':2,'records':[{'region':0,'cosine':.7,'cells':[a,b]}]}])
    assert result['all']['valid_tensor_cells']==1
    assert result['all']['negative_tensor_cells']==0
    assert result['raw_rectangle_post_initialization']['negative_tensor_cells']==1
