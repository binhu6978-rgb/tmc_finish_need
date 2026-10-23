"""Two frozen 500-round trajectories; small preflight uses the same original loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_final40_job import make_args
from tools.final40_data import datasets_for
from tools import run_evidence_candidate as base
from semcons_a07_v2.pair500 import pair_loop,sampled_rounds,verify_pair


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--mode',choices=('neutral','active'))
    p.add_argument('--rounds',type=int,choices=(2,500,1000))
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/semantic_a07_pair500.json')
    p.add_argument('--execute1000',action='store_true')
    p.add_argument('--output-root',type=Path);p.add_argument('--prepared',type=Path)
    p.add_argument('--dataset-root',type=Path);p.add_argument('--no-diagnostic',action='store_true')
    p.add_argument('--pilot-proof',type=Path);p.add_argument('--resume',type=Path)
    p.add_argument('--verify-pilot',type=Path);p.add_argument('--verify-pair',type=Path)
    opt=p.parse_args();os.chdir(ROOT)
    if opt.verify_pilot:
        root=opt.verify_pilot
        paths=[root/x/'training/a07/neutral_2_seed1/regression.jsonl' for x in ('plain','diagnosed')]
        rows=[[json.loads(l) for l in v.read_text().splitlines()] for v in paths]
        assert len(rows[0])==3 and rows[0]==rows[1]
        result=dict(success=True,rounds=2,exact_models_slots_sgd_rng_official_test=True)
        (root/'pilot_pass.json').write_text(json.dumps(result,indent=2));print(result);return
    if opt.verify_pair:
        result=verify_pair(opt.verify_pair,500)
        (opt.verify_pair/'paired_verification.json').write_text(json.dumps(result,indent=2));print(result);return
    if not all((opt.mode,opt.rounds,opt.output_root,opt.prepared,opt.dataset_root)):p.error('Missing run arguments')
    if opt.rounds in (500,1000) and (not opt.pilot_proof or not json.loads(opt.pilot_proof.read_text())['success']):
        raise ValueError('Real two-round diagnostic regression must pass first')
    protocol=json.loads(opt.protocol.read_text())
    if opt.rounds>2 and opt.rounds!=protocol['rounds']:raise ValueError('Round budget differs from frozen protocol')
    if opt.rounds==1000 and (not opt.execute1000 or opt.mode!='active'):raise ValueError('Authorized1000 is a single active V2 run')
    cfg=json.loads((ROOT/'configs/semcons_semantic_a07_v2.json').read_text())
    assert cfg['semantic_lambda']==protocol['semantic_lambda']==.2
    cfg['diagnostic_rounds']={('pair500' if protocol['rounds']==500 else 'activation1000'):sampled_rounds(protocol)} if not opt.no_diagnostic else {}
    root=opt.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
    args,condition,users=make_args('A07',root/'training',opt.prepared)
    args.epochs=opt.rounds;args.method_run_name=f'{opt.mode}_{opt.rounds}_seed1'
    from utils.reproducibility import configure_deterministic_execution
    policy=configure_deterministic_execution(True);base.torch.set_num_threads(2)
    args.device=base.torch.device('cuda:0');path=base.prepare_output(args)
    files=[ROOT/v for v in ('capability_final40.py','native_group_aggregation.py','semcons_scope.py',
        'models/Update.py','models/Fed.py','models/resnet.py','models/test.py','Algorithm/Training_AdaptiveFL.py',
        'adapterfl_methods.py','tools/run_semantic_pair500.py')]
    files.append(opt.protocol.resolve())
    files+=list((ROOT/'semcons_a07').glob('*.py'))+list((ROOT/'semcons_a07_v2').glob('*.py'))
    source={str(v.relative_to(ROOT)):hashlib.sha256(v.read_bytes()).hexdigest() for v in files}
    identity=dict(source=source,mode=opt.mode,epochs=opt.rounds,cfg=cfg,protocol=protocol,
        partition=args._method_partition_sha256)
    manifest=dict(identity=identity,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        configuration_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
        argv=sys.argv,deterministic=policy,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch=base.torch.__version__,python=sys.version,pid=os.getpid(),
        original_partition=True,full_training_examples=50000,probe_sgd_independent=False,
        probe_semantic_measurement_disjoint=True,crd='omitted_by_user',
        recovery='one rolling complete state every50 rounds/final; no test-selected snapshot',
        resume_source=str(opt.resume) if opt.resume else None)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter()
    try:
        base.set_random_seed(1);train,test=datasets_for(condition,opt.dataset_root)
        assert len(train)==50000 and len(test)==10000
        unused=base.ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
        base.torch.cuda.reset_peak_memory_stats()
        with pair_loop(cfg,train,users,path,opt.mode,args._final40_context['boundaries'],protocol,
                       identity,not opt.no_diagnostic,opt.resume):
            base.AdaptiveFL(args,train,test,users)
        result=json.loads((path/'completed.json').read_text());assert result['success']
        (root/'completed.json').write_text(json.dumps(result,indent=2))
    except BaseException:
        import traceback
        (root/'failed.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));raise
    finally:
        (root/'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
            peak_allocated=base.torch.cuda.max_memory_allocated(),peak_reserved=base.torch.cuda.max_memory_reserved()),indent=2))


if __name__=='__main__':main()
