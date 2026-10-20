import csv
import json

import pytest
import torch

from tools.report_tonight10 import diagnostic_summary, validate


def completed_fixture(path):
    with (path/'accuracy_each_round.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','full_accuracy','full_ce']);w.writeheader()
        w.writerows({'round':i,'full_accuracy':40. if i==100 else 30.,'full_ce':1.} for i in range(1,401))
    (path/'completed.json').write_text(json.dumps({'success':True,'rounds_completed':400}))
    torch.save({'next_round':400,'rows':[{}]*400},path/'latest.pt')
    torch.save({'round':100,'full_accuracy':40.},path/'best_full.pt')


def test_completion_and_peak_validation(tmp_path):
    completed_fixture(tmp_path)
    assert validate(tmp_path)['best_round']==100
    torch.save({'round':101,'full_accuracy':40.},tmp_path/'best_full.pt')
    with pytest.raises(ValueError,match='first test-oracle peak'):validate(tmp_path)


def test_incomplete_not_counted_successful(tmp_path):
    completed_fixture(tmp_path)
    torch.save({'next_round':399,'rows':[{}]*399},tmp_path/'latest.pt')
    with pytest.raises(ValueError,match='Not400'):validate(tmp_path)


def test_diagnostic_summary_actual_conflicts_and_later_scale(tmp_path):
    p=tmp_path/'diag.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','metrics_json']);w.writeheader()
        for i in range(1,401):
            d={'conflict_slot_count':1 if i>300 else 0,'difference_from_e03_l2':1.,
                'changed_from_e03_fraction':.2,'conflict_weights':[.5]+[1.]*9,
                'update_l2_ratio_e03':50. if i==1 else .98,'effective_step_along_e03':.97,
                'update_cosine_e03':.99,'protected_components':[{}] if i>300 else []}
            w.writerow({'round':i,'metrics_json':json.dumps({'server':d,'slots':[(1,6)]*10})})
    d,s=diagnostic_summary(p)
    assert d['conflict_slots']==100 and d['protected_components']==100
    assert d['later_l2_ratio_e03_mean']==pytest.approx(.98)
    assert d['phases']['301-400']['conflict_slots']==100 and len(s)==400


def test_c2_projection_activity_is_distinct_from_t09_restoration(tmp_path):
    path=tmp_path/'c2.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['round','metrics_json']);writer.writeheader()
        for i in range(1,401):
            server={'conflict_slot_count':2 if i==2 else 0,'conflict_weights':[1.]*10,
                'adjusted_projection_slot_count':2 if i==2 else 0,
                'difference_from_e03_l2':100. if i==1 else .5}
            writer.writerow({'round':i,'metrics_json':json.dumps({'server':server,'slots':[(1,0)]*10})})
    result,_=diagnostic_summary(path)
    assert result['negative_projection_adjusted_slots']==2
    assert result['negative_projection_active_rounds']==1
    assert result['protected_components']==0 and result['restoration_active_rounds']==0
    assert result['later_same_input_e03_difference_l2_mean']==.5
