"""Fixed G03 refinements: same uploads, coverage, first-round Native and RNG."""
import contextlib
import copy
import csv
import io
from pathlib import Path

import pytest
import torch
from torch import nn

from native_group_aggregation import NativeGroupHooks,REFINEMENTS,balanced_weights,capability_weights,update_norm
from models.Fed import Aggregation_AdaptiveFL
from test_native_groups import case,active
from test_adapterfl_methods import args_for,assert_states,assert_rng,tiny_models


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(2)


@pytest.mark.parametrize('method',REFINEMENTS)
def test_warmup_snapshot_rng_uncovered_and_history_state(method):
    from adapterfl_methods import rng_state
    h,u,n,w,s=case(method)
    original=copy.deepcopy((u,w,s));before=rng_state()
    first=h.aggregate(u,n,w,s)
    assert_states(first,Aggregation_AdaptiveFL(u,n,w));assert_rng(before,rng_state())
    result=h.aggregate(u,n,w,s);assert_rng(before,rng_state())
    for a,b in zip(u,original[0]):assert_states(a,b)
    assert_states(w,original[1])
    restored,*_=case(method);restored.load_state_dict(h.state_dict(),w)
    assert_states(restored.aggregate(u,n,w,s),h.aggregate(u,n,w,s))
    # Only profiles 0 and 2 cover prefixes <=3; no update outside them.
    h,u,n,w,s=case(method,profiles=(0,2,2),cids=(1,45,75),lens=(2,3,5),values=(1.,.5,-.2))
    result=active(h,u,n,w,s)
    assert torch.equal(result['weight'][3:],w['weight'][3:])


def test_sqrt_inverse_population_and_equal_mass_exact_g03():
    slots=[(1,0),(2,0),(40,0),(70,0)];lens=[1,3,9,16];members=list(range(4))
    for mode,powers in [('p03',[2.,3.,4.]),('p04',[.5,1/3,.25]),('p05',[4.,3.,3.])]:
        weights=capability_weights(mode,members,slots,lens,(40,70));total=sum(weights)
        actual=[sum(weights[i] for i in ids)/total for ids in ([0,1],[2],[3])]
        assert actual==pytest.approx([x/sum(powers) for x in powers])
        assert weights[1]/weights[0]==pytest.approx(3.)
    lens=[1,3,4,4]
    for mode in ('p03','p04'):
        assert capability_weights(mode,members,slots,lens,(40,70))==balanced_weights('g03',members,slots,lens,(40,70))


def test_dedup_only_inside_tier_and_covered_region():
    h,u,n,w,s=case('p06',profiles=(0,2,2,6),cids=(1,1,2,70),lens=(2,2,6,10),values=(1.,3.,5.,7.))
    result=active(h,u,n,w,s)['weight']
    # Low tier: clients (mean(1,3),5) weighted (2,6), then half with high tier 7.
    assert result[0].item()==pytest.approx(((2*2+5*6)/8+7)/2)
    # Beyond profile0, client1 has one actual covering slot, not two.
    assert result[1].item()==pytest.approx(.01+((3*2+5*6)/8+7)/2)
    assert torch.equal(result[3:],Aggregation_AdaptiveFL(u,n,w)['weight'][3:])


def test_hierarchy_real_tier_profile_groups_and_single_tier_neutral():
    slots=[(1,0),(2,0),(3,2),(70,6)];lens=[1,3,6,8]
    w=capability_weights('p07',[0,1,2,3],slots,lens,(40,70))
    shares=[x/sum(w) for x in w]
    assert shares==pytest.approx([.25/4,.25*3/4,.25,.5])
    # Present low-tier profile2 vs high-tier profile6 only: equal tiers.
    assert capability_weights('p07',[2,3],slots,lens,(40,70))==pytest.approx([7.,7.])
    for mode in ('p03','p04','p05','p06','p07'):
        assert capability_weights(mode,[0,1,2],slots,lens,(40,70))==[1,3,6]


def test_fixed_region_scopes_and_scale_controls_same_input():
    outputs={};hooks={}
    for mode in ('g03','p01','p02','p08','p09'):
        h,u,n,w,s=case(mode);outputs[mode]=active(h,u,n,w,s);hooks[mode]=h
    native=Aggregation_AdaptiveFL(u,n,w)
    assert update_norm(outputs['p01'],w)==pytest.approx(update_norm(outputs['g03'],w),rel=2e-6)
    assert update_norm(outputs['p02'],w)==pytest.approx(update_norm(native,w),rel=2e-6)
    assert hooks['p01'].round_diagnostics['actual_update_cosine_native']==pytest.approx(1.,abs=1e-6)
    assert hooks['p02'].round_diagnostics['actual_update_cosine_g03']==pytest.approx(1.,abs=1e-6)
    assert torch.equal(outputs['p08']['weight'][:1],outputs['g03']['weight'][:1])
    assert torch.equal(outputs['p08']['weight'][1:],native['weight'][1:])
    assert torch.equal(outputs['p09']['weight'][:1],native['weight'][:1])
    assert torch.equal(outputs['p09']['weight'][1:],outputs['g03']['weight'][1:])


def test_p10_classifier_only_native_and_bn_still_g03():
    class Model(nn.Module):
        def __init__(self,n):
            super().__init__();self.conv=nn.Linear(n,n,bias=False)
            self.bn=nn.BatchNorm1d(n,track_running_stats=False);self.classifier=nn.Linear(n,2)
    models=[Model(i) for i in range(1,8)]
    w={k:torch.zeros_like(v) for k,v in models[-1].state_dict().items()}
    slots=[(1,0),(2,2),(45,4),(75,6)];n=[1,3,7,9]
    s=[{k:torch.zeros_like(v) for k,v in models[p].state_dict().items()} for c,p in slots]
    u=[{k:v+(i+1)*.1 for k,v in x.items()} for i,x in enumerate(s)]
    out={}
    for mode in ('g03','p10'):
        h=NativeGroupHooks(args_for(mode),models);h.set_slots(slots);out[mode]=active(h,u,n,w,s)
    native=Aggregation_AdaptiveFL(u,n,w)
    for k in w:
        assert torch.equal(out['p10'][k],native[k] if k.startswith('classifier.') else out['g03'][k])


def test_zero_direction_normal_neutral_all_modes():
    for mode in REFINEMENTS:
        h,u,n,w,s=case(mode,profiles=(6,6),cids=(1,70),lens=(1,1),values=(0.,0.))
        result=active(h,u,n,w,s);native=Aggregation_AdaptiveFL(u,n,w)
        # P05 unequal positive weights can round identical float32 uploads by
        # one ULP; do not mistake that for a new learning update or an error.
        assert torch.allclose(result['weight'],native['weight'],rtol=0,atol=8e-9)
        if mode in ('p01','p02'):assert_states(result,native)


def test_fixed_queue_has_only_requested_ten_and_no_historical_repeats():
    from tools.run_native_groups10 import jobs
    config=Path(__file__).resolve().parents[1]/'configs/g03_refinements10_400.json'
    batches=jobs(Path('python'),Path('/data'),Path('runs/p_new'),config)
    assert [[j['id'] for j in b] for b in batches]==[[f'P{i:02d}',f'P{i+1:02d}'] for i in (1,3,5,7,9)]
    for job in sum(batches,[]):
        assert '--deterministic-execution' in job['argv']
        assert job['argv'][job['argv'].index('--epochs')+1]=='400'
    assert len({j['output'] for b in batches for j in b})==10


def test_completed_reference_remains_usable_after_user_checkpoint_cleanup(tmp_path):
    import json
    import shutil
    from tools.report_native_groups10 import validate_reference
    source=Path(__file__).resolve().parents[1]/'docs/evidence/native_groups10_results/raw/G03'
    for name in ('completed.json','accuracy_each_round.csv'):shutil.copyfile(source/name,tmp_path/name)
    assert not list(tmp_path.glob('*.pt'))
    assert validate_reference(tmp_path)['test_oracle_peak']==pytest.approx(38.45,abs=1e-5)
    done=json.loads((tmp_path/'completed.json').read_text());done['rounds_completed']=2
    (tmp_path/'completed.json').write_text(json.dumps(done))
    with pytest.raises(ValueError,match='did not complete'):validate_reference(tmp_path)
    shutil.copyfile(source/'completed.json',tmp_path/'completed.json')
    with (tmp_path/'accuracy_each_round.csv').open() as f:lines=f.readlines()
    (tmp_path/'accuracy_each_round.csv').write_text(''.join(lines[:3]))
    with pytest.raises(ValueError,match='Expected completed 400'):validate_reference(tmp_path)


def test_original_loop_two_round_full_only_all_ten(tmp_path,monkeypatch):
    import Algorithm.Training_AdaptiveFL as training
    from utils.set_seed import set_random_seed
    data=torch.utils.data.TensorDataset(torch.arange(4*3*8*8).reshape(4,3,8,8).float()/1000,torch.tensor([0,1,0,1]))
    monkeypatch.setattr(training,'get_model_list',lambda args:(tiny_models(),[(.4,2,i+1.) for i in range(7)]))
    monkeypatch.setattr(training,'tqdm',lambda x:x)
    for mode in REFINEMENTS:
        args=args_for(mode,tmp_path/mode,epochs=2,eval_profiles='full',frac=.1);set_random_seed(1)
        with contextlib.redirect_stdout(io.StringIO()):training.AdaptiveFL(args,data,data,{i:[0,1,2,3] for i in range(100)})
        folder=next((tmp_path/mode/mode).iterdir())
        with (folder/'accuracy_each_round.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==2 and set(k for k in rows[0] if k.startswith('P'))=={'P6_accuracy','P6_ce'}
        state=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
        assert state['next_round']==2 and state['method_state']['rounds_seen']==2
