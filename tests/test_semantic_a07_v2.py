"""V2 mechanism correctness, no claim of global utility or accuracy improvement."""
import copy
import json
import math
from pathlib import Path
import pytest
import torch
from semcons_a07_v2.memory import task_rows,TierSemanticMemory,TierReliabilityMemory
from semcons_a07_v2.scoring import complementarity
from semcons_a07_v2.aggregation import blend_shares,fuse
from test_final40 import setup
from test_adapterfl_methods import assert_states
from semcons_scope import slices


def config():
    return json.loads(Path('configs/semcons_semantic_a07_v2.json').read_text())


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


def task(q,y):
    return dict(count=2,probability=torch.tensor([q,1-q] if y==0 else [1-q,q],dtype=torch.float64))


def observation(cid,profile,pre,post):
    return dict(client=cid,profile=profile,task_before={y:task(pre,y) for y in (0,1)},task_after={y:task(post,y) for y in (0,1)})


def records(pre=(.45,.45,.45),post=(.8,.6,.5)):
    return task_rows([observation(cid,p,pre[g],post[g]) for g,ids in enumerate(((1,2),(40,41),(70,71))) for p,cid in enumerate(ids)],(40,70),2)


def past():
    cfg=config();semantic=TierSemanticMemory(cfg);reliable=TierReliabilityMemory(cfg)
    for r in (2,3):
        updates=semantic.commit_tiers(r,records((.3,.3,.3),(.4,.4,.4)))
        reliable.commit_tiers(r,updates)
    return cfg,semantic,reliable


def decision():
    cfg,semantic,reliable=past()
    return complementarity(records(),semantic.tier_snapshot(),reliable.tier_snapshot(),reliable,4,[0,1,2],cfg)


@pytest.mark.parametrize('strength,scores',[(0.,{0:.1,1:.5}),(.2,{0:.1,1:.1}),(.2,{}),(.2,{0:float('nan'),1:.1})])
def test_exact_neutral_shares(strength,scores):
    pi={0:.3,1:.7}
    assert blend_shares(pi,scores,strength)==pi


def test_reliable_different_u_changes_nonnegative_normalized_shares():
    d=decision();assert d['valid'] and d['scores'][0]>d['scores'][1]>d['scores'][2]
    pi={0:.3,1:.2,2:.5};new=blend_shares(pi,d['scores'],.2)
    assert new!=pi and sum(new.values())==pytest.approx(1)
    assert all(math.isfinite(v) and v>=0 for v in new.values())
    assert set(d['classes'])=={0,1}


@pytest.mark.parametrize('strength',[-.1,1.1,float('nan')])
def test_invalid_lambda_rejected(strength):
    with pytest.raises(ValueError):blend_shares({0:.5,1:.5},{0:.1,1:.2},strength)


@pytest.mark.parametrize('case',['lambda_zero','missing','equal','first'])
def test_neutral_returns_original_a07_object_and_exact_state(case):
    h,u,n,w,_=setup('a07');h.aggregate(u,n,w);base=h.aggregate(u,n,w)
    d=decision()
    if case=='missing':d=dict(valid=False,reason='no_history',scores={},classes=[],details=[])
    if case=='equal':d['scores']={0:.2,1:.2,2:.2}
    out,report=fuse(base,w,u,n,h,d,0 if case=='lambda_zero' else .2,case=='first')
    assert out is base and report['changed_parameters']==0
    assert_states(base,out)


def test_synthetic_effective_fusion_preserves_bn_classifier_uncovered_and_within_tier_ratio():
    h,u,n,w,_=setup('a07',profiles=(0,2,3,4,5),cids=(1,2,40,41,70),lens=(2,6,10,14,20))
    h.aggregate(u,n,w);base=h.aggregate(u,n,w);original=copy.deepcopy(base)
    out,report=fuse(base,w,u,n,h,decision(),.2,False)
    assert report['changed_parameters']>0 and report['difference_from_a07_l2']>0
    assert_states(base,original)
    for key in h.bn_keys|{'classifier.weight','classifier.bias'}:
        assert torch.equal(out[key],base[key])
    for region in h.regions:
        if not any(p in region['profiles'] for _,p in h.slots):
            for cell in region['cells']:assert torch.equal(out[cell['key']][slices(cell)],w[cell['key']][slices(cell)])
    for detail in report['regions']:
        if detail['reason']!='semantic_source_blend':continue
        assert sum(detail['new_slot_shares'])==pytest.approx(1)
        members=detail['slots']
        for a,i in enumerate(members):
            for b,j in enumerate(members):
                if sum(h.slots[i][0]>=v for v in h.boundaries)==sum(h.slots[j][0]>=v for v in h.boundaries):
                    assert detail['new_slot_shares'][a]/detail['new_slot_shares'][b]==pytest.approx(n[i]/n[j])


def test_single_tier_regions_neutral():
    h,u,n,w,_=setup('a07',cids=(70,71,72,73));h.aggregate(u,n,w);base=h.aggregate(u,n,w)
    out,report=fuse(base,w,u,n,h,decision(),.2,False)
    assert out is base and report['changed_parameters']==0


def test_duplicate_identity_and_profiles_are_one_semantic_vote():
    rows=[observation(1,0,.1,.2),observation(1,2,.3,.4),observation(2,0,.2,.5)]
    current=task_rows(rows,(40,70),2)
    assert current[0,0]['support']==2 and current[0,0]['clients']==[1,2]
    assert current[0,0]['sources'][0]['slots']==2
    only=task_rows(rows[:2],(40,70),2)
    semantic=TierSemanticMemory(config());assert not semantic.commit_tiers(2,only)


def test_snapshots_past_only_initialization_and_expiry():
    cfg,semantic,reliable=past();anchors=semantic.tier_snapshot();history=reliable.tier_snapshot()
    fresh=TierSemanticMemory(cfg);assert fresh.commit_tiers(1,records())=={}
    semantic.commit_tiers(4,records())
    assert anchors[0,0]['round']==3
    assert complementarity(records(),anchors,history,reliable,4,[0,1,2],cfg)['valid']
    assert not complementarity(records(),semantic.tier_snapshot(),history,reliable,4,[0,1,2],cfg)['valid']
    assert not complementarity(records(),anchors,history,reliable,104,[0,1,2],cfg)['valid']


def test_both_memories_affect_score_but_s1_and_adapter_do_not():
    cfg,semantic,reliable=past();anchors=semantic.tier_snapshot();history=reliable.tier_snapshot()
    original=complementarity(records(),anchors,history,reliable,4,[0,1,2],cfg)
    assert original['valid']
    assert not complementarity(records(),{},history,reliable,4,[0,1,2],cfg)['valid']
    assert not complementarity(records(),anchors,{},reliable,4,[0,1,2],cfg)['valid']
    changed=copy.deepcopy(anchors)
    for value in changed.values():value['q']=.99
    zero=complementarity(records(),changed,history,reliable,4,[0,1,2],cfg)
    assert zero['valid'] and set(zero['scores'].values())=={0.}
    unstable=copy.deepcopy(history)
    for v in unstable.values():
        for r in v:r['change']=.8
    lower=complementarity(records(),anchors,unstable,reliable,4,[0,1,2],cfg)
    assert lower['scores'][0]<original['scores'][0]
    reliable.clients[1]=[(3,1e10)]  # old S1 bank: explicitly not knowledge reliability
    assert complementarity(records(),anchors,history,reliable,4,[0,1,2],cfg)==original
    rows=[observation(c,0,.2,.6) for c in (1,2,40,41)]
    baseline=task_rows(rows,(40,70),2)
    for row in rows:row['heads']={'unvalidated_embedding':'arbitrary'}
    assert task_rows(rows,(40,70),2)==baseline


@pytest.mark.parametrize('case',['no_history','single_tier','missing_tier','one_class','nonfinite'])
def test_missingness_is_not_tier_inferiority(case):
    cfg,sem,rel=past();current=records();anchors=sem.tier_snapshot();history=rel.tier_snapshot();tiers=[0,1,2]
    if case=='no_history':anchors={}
    if case=='single_tier':tiers=[0]
    if case=='missing_tier':current={key:v for key,v in current.items() if key[0]!=0}
    if case=='one_class':current={key:v for key,v in current.items() if key[1]==0}
    if case=='nonfinite':current[0,0]['post']=float('nan');current[0,1]['pre']=float('inf')
    assert not complementarity(current,anchors,history,rel,4,tiers,cfg)['valid']
