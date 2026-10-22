"""V2 short correctness or explicitly requested formal entry, original A07 loop."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_final40_job import make_args
from tools import run_evidence_candidate as base
from tools.final40_data import datasets_for
from semcons_a07_v2.trainer import semantic_loop


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=('reference','neutral','active'),required=True)
    p.add_argument('--rounds',type=int,choices=(2,20,400),required=True)
    p.add_argument('--semantic-lambda',type=float)
    p.add_argument('--config',type=Path,default=ROOT/'configs/semcons_semantic_a07_v2.json')
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--execute-formal',action='store_true',help='Explicit operator opt-in; not used by short test suite')
    opt=p.parse_args();os.chdir(ROOT)
    if opt.rounds==400 and not opt.execute_formal:raise ValueError('400-round execution requires explicit opt-in; not authorized by the short suite')
    cfg=json.loads(opt.config.read_text())
    if opt.semantic_lambda is not None:cfg['semantic_lambda']=opt.semantic_lambda
    if not math.isfinite(cfg['semantic_lambda']) or not 0<=cfg['semantic_lambda']<=1:raise ValueError('Invalid semantic lambda')
    digest=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    args,condition,users=make_args('A07',opt.output_root,opt.prepared)
    args.epochs=opt.rounds;args.method_run_name=f'{opt.mode}_{opt.rounds}_seed1'
    from utils.reproducibility import configure_deterministic_execution
    policy=configure_deterministic_execution(True)
    base.torch.set_num_threads(2);args.device=base.torch.device('cuda:0')
    path=base.prepare_output(args);start=time.perf_counter()
    files=[*sorted((ROOT/'semcons_a07_v2').glob('*.py')),*sorted((ROOT/'semcons_a07').glob('*.py')),Path(__file__),opt.config]
    source={str(v.relative_to(ROOT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in files}
    try:
        base.set_random_seed(1);train,test=datasets_for(condition,opt.dataset_root)
        (path/'v2_manifest.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256=source,configuration=cfg,config_sha256=digest,partition_sha256=args._method_partition_sha256,
            argv=sys.argv,mode=opt.mode,rounds=opt.rounds,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            deterministic_execution=policy,torch=base.torch.__version__,save_pt=False,adapter_in_decision=False,
            effective_lambda=cfg['semantic_lambda'] if opt.mode=='active' else 0.,
            interpretation='New potential-complementarity hypothesis; not a validation of v1'),indent=2))
        unused=base.ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
        base.torch.cuda.reset_peak_memory_stats()
        with semantic_loop(cfg,train,users,path,opt.mode,args._final40_context['boundaries']):
            base.AdaptiveFL(args,train,test,users)
    except BaseException:
        import traceback
        (path/'failed.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));raise
    finally:
        if base.torch.cuda.is_initialized():
            (path/'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-start,
                peak_allocated_bytes=base.torch.cuda.max_memory_allocated(),peak_reserved_bytes=base.torch.cuda.max_memory_reserved()),indent=2))


if __name__=='__main__':main()
