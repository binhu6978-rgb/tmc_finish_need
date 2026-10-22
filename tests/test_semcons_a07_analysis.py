"""Statistical plumbing and separation contracts, not evidence of real efficacy."""
import copy
import numpy as np
import pytest
from tools.analyze_semcons_a07 import auc, correlation, predict, shuffled, score_result, description
from semcons_a07.config import load


def row(i, split='development'):
    value = (i % 11-5)/10
    return dict(sender=i % 20 + (70 if split=='confirmation' else 0), receiver=(i+3)%20 + (70 if split=='confirmation' else 0),
        stage='early' if split=='development' else 'middle', split=split,
        sender_profile=i%7, receiver_profile=(i+1)%7, sender_tier=0, receiver_tier=1,
        direction_cosine=value, scaled_update_l2=1+i%3, alpha=.1,
        before_ce=2., own_gain=.01, sender_sample_mass=500,
        delta_ce=value*.03, negative_loo=value<0,
        evidence=dict(S0=value, S1=value, S2=value, valid_embedding_classes=[0,1], valid_task_classes=2,
            current_embedding_change=.2, current_task_change=.01),
        reliability={k:dict(support=0, ready=False, age=None, mean=None, std=None) for k in ('client','client_profile')})


def test_auc_and_rank_ties_signs():
    assert auc([0,1],[False,True]) == 1
    assert auc([1,0],[False,True]) == 0
    assert auc([1,1],[False,True]) == .5
    assert auc([1,2],[True,True]) is None
    assert correlation([1,2,3], [3,2,1], True) == pytest.approx(-1)


def test_predict_never_uses_heldout_labels():
    dev = [row(i) for i in range(60)]
    confirm = [row(i, 'confirmation') for i in range(20)]
    original = predict(dev, confirm, 'S1')
    for r in confirm:
        r['delta_ce'] = 10000
    assert np.array_equal(original, predict(dev, confirm, 'S1'))


def test_permutation_preserves_identities_validity_and_stratified_distribution():
    rows = [row(i) for i in range(140)]
    changed, count = shuffled(rows, 'S0', np.random.default_rng(3))
    assert count > 0
    for a,b in zip(rows,changed):
        assert a['sender']==b['sender'] and a['receiver']==b['receiver']
        assert a['evidence']['valid_embedding_classes']==b['evidence']['valid_embedding_classes']
        assert a['delta_ce']==b['delta_ce']
    for p in range(7):
        assert sorted(r['evidence']['S0'] for r in rows if r['sender_profile']==p) == sorted(r['evidence']['S0'] for r in changed if r['sender_profile']==p)


def test_insufficient_heldout_data_cannot_authorize_go():
    result = score_result([row(i) for i in range(40)], load(), 'S0')
    assert result['go'] is False
    assert result['confirmation']['pairs']==0
    assert result['reason']=='insufficient_disjoint_development_or_confirmation'


def test_pair_counts_not_class_or_head_pseudoreplicates():
    values = [row(i) for i in range(20)]
    d = description(values)
    assert d['pairs']==20 and d['senders']==20 and d['receivers']==20
    assert d['harmful']+d['beneficial']+d['zero']==20
