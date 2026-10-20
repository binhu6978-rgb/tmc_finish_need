"""Actual seven-shape aggregation/memory check with synthetic uploads, no SGD."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refinements',action='store_true');a=p.parse_args()
    import torch
    from utils.reproducibility import configure_deterministic_execution
    from utils.options import args_parser
    from models.Fed import get_model_list,Aggregation_AdaptiveFL,split_model
    from native_group_aggregation import NativeGroupHooks,MODES,REFINEMENTS,update_norm
    configure_deterministic_execution(True);torch.set_num_threads(2)
    from utils.set_seed import set_random_seed
    set_random_seed(1)
    sys.argv=['check'];args=args_parser();args.model='resnet';args.device=torch.device('cuda:0')
    with contextlib.redirect_stdout(io.StringIO()):models,_=get_model_list(args)
    previous=copy.deepcopy(models[-1].state_dict())
    profiles=[0,0,2,3,4,5,6,6,6,6];cids=[1,2,3,40,41,42,70,70,71,72];lens=[100+cid for cid in cids]
    sent=[{k:v.clone() for k,v in split_model(previous,models[p].state_dict()).items()} for p in profiles]
    uploads=[{k:v+(i-3)*.001 for k,v in s.items()} for i,s in enumerate(sent)]
    native=Aggregation_AdaptiveFL(uploads,lens,previous)
    torch.cuda.reset_peak_memory_stats();records={};outputs={}
    for mode in REFINEMENTS if a.refinements else MODES:
        args.method=mode;hook=NativeGroupHooks(args,models);hook.set_slots(zip(cids,profiles))
        first=hook.aggregate(uploads,lens,previous,sent)
        assert all(torch.equal(first[k],native[k]) for k in previous)
        del first
        out=hook.aggregate(uploads,lens,previous,sent)
        assert len(hook.regions)==9 and all(bool(torch.isfinite(v).all()) for v in out.values())
        d=hook.round_diagnostics
        records[mode]=dict(regions=len(hook.regions),actual_update_l2=d['actual_update_l2'],
            difference_from_native_l2=d['difference_from_native_l2'],scale_norm_relative_error=d['scale_norm_relative_error'],
            gain_min=min(r['gain'] for r in d['regions']),gain_max=max(r['gain'] for r in d['regions']))
        if mode in ('g07','g09'):outputs[mode]=update_norm(out,previous)
        if mode in ('g08','g10'):
            target=outputs['g07' if mode=='g08' else 'g09']
            assert abs(update_norm(out,previous)-target)<max(1e-7,target*1e-5)
        if mode in ('p01','p02'):
            target=d['matched_norm_target']
            assert abs(update_norm(out,previous)-target)<max(1e-7,target*1e-5)
            records[mode]['matched_norm_relative_error']=d['matched_norm_relative_error']
        if mode=='p10':
            assert torch.equal(out['classifier.weight'],native['classifier.weight'])
            assert torch.equal(out['classifier.bias'],native['classifier.bias'])
        del out,hook
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(type='synthetic uploads on actual seven ResNet shapes, not CIFAR training',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,checks=records,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20),indent=2))


if __name__=='__main__':main()
