"""Fixed seven-profile, seed1 C0--C5 runs. No resume or long-run extension interface."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import numpy as np

from experiments.baseline import content_hash, load_data, load_partition, sha256, state_hash
from hgkr.core import Config, KernelHistory
from hgkr.protocol import model_costs
from hgkr.seven import MODES, initialize_models, make_schedule, run_round, shared_specs
from models.test import test_img
from semconsfl.aggregation import coverage
from semconsfl.measurement import isolated_rng
from utils.options import args_parser
from utils.set_seed import set_random_seed

PARTITION = 'data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
CODE_FILES = [
    'models/__init__.py','models/resnet.py','models/Fed.py','models/Update.py','models/test.py',
    'models/mobileNetV2.py','models/vgg.py','models/Nets.py','models/LSTM.py','optimizer/Adabelief.py',
    'utils/options.py','utils/set_seed.py','utils/scaler.py','experiments/__init__.py','experiments/baseline.py',
    'experiments/run_hgkr7.py','experiments/summarize_hgkr7.py','experiments/hgkr7_preflight.py',
    'hgkr/__init__.py','hgkr/core.py','hgkr/protocol.py','hgkr/seven.py',
    'semconsfl/__init__.py','semconsfl/config.py','semconsfl/aggregation.py','semconsfl/measurement.py',
    'configs/hgkr7_300.json',*[f'configs/hgkr7/{m}.json' for m in MODES]]


def write_json(path, value):
    path=Path(path); pending=Path(str(path)+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(pending,path)


def sources(): return {p:sha256(p) for p in CODE_FILES}


def parse(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,default='C0')
    p.add_argument('--config')
    p.add_argument('--dataset-root',default='data/cifar100')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--plan')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke-rounds',type=int,choices=(2,5))
    opt=p.parse_args(argv)
    previous=sys.argv
    try:
        sys.argv=[previous[0]]; args=args_parser()
    finally: sys.argv=previous
    args.model,args.dataset,args.algorithm='resnet','cifar100','HGKR7'
    args.width_ration,args.depth_saved=[.4,.66,1.],[2,3,4]
    args.seed,args.epochs,args.generate_data=1,opt.smoke_rounds or 300,0
    args.gpu=opt.gpu
    args.device=torch.device(f'cuda:{opt.gpu}' if opt.gpu>=0 else 'cpu')
    if opt.gpu>=0 and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    if opt.gpu<0 and not (opt.prepare or opt.smoke_rounds): raise ValueError('Formal runs require a GPU')
    mode_config=json.loads(Path(opt.config or f'configs/hgkr7/{opt.mode}.json').read_text())
    if mode_config!={'mode':opt.mode,'epochs':300,'seed':1,'common_config':'configs/hgkr7_300.json'}:
        raise ValueError('Expected fixed 300-round seed1 mode configuration')
    config=Config(**json.loads(Path(mode_config['common_config']).read_text()))
    return opt,args,config


def version():
    if Path('VERSION.json').exists(): return json.loads(Path('VERSION.json').read_text())
    try: return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    except (OSError,subprocess.CalledProcessError): return {'git_commit':'unavailable','source_identity':'see source hashes'}


def prepare(args, config, opt, train, users, partition, models, profiles):
    costs=model_costs(models)
    _,region=coverage(models)
    return {'protocol':'HGKR7-300-full-eval-v2','epochs':args.epochs,'seed':1,'config':vars(config),
            'partition':partition,'profiles':profiles,'costs':costs,'coverage':region,
            'shared_kernel_shapes':shared_specs(models),
            'initial_state_sha256':[state_hash(m.state_dict()) for m in models],
            'schedule':make_schedule(1,args.epochs,users,costs,config),
            'dataset_sha256':{n:sha256(Path(opt.dataset_root)/'cifar-100-python'/n) for n in ('train','test','meta')},
            'source_sha256':sources(),'version':version(),
            'training_args':{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items() if k not in ('device','gpu')},
            'evaluation':{'profiles':['P6'],'frequency':'every_round','batch_size':args.bs,
                          'rng':'isolated_from_training'},
            'test_protocol':'Original official test once per round, P6 Full only; peak is test-oracle peak'}


def evaluate_full(models, dataset, args):
    """One original P6 test, isolated from future SGD RNG and model mode state."""
    full=models[6]
    modes=[(module,module.training) for module in full.modules()]
    with isolated_rng():
        try:
            accuracy,ce=test_img(full,dataset,args)
        finally:
            for module,training in modes: module.training=training
    return {'profile':'P6','accuracy':accuracy,'ce':ce}


def save_weights(path, model, round_id, accuracy):
    temporary=Path(str(path)+'.pending')
    torch.save({'round_1based':round_id+1,'full_test':accuracy,
                'global_full':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},temporary)
    os.replace(temporary,path)


def main(argv=None):
    opt,args,config=parse(argv)
    train,test=load_data(opt.dataset_root)
    users,partition=load_partition(PARTITION,args,train.targets)
    set_random_seed(1); models,profiles=initialize_models(args)
    if opt.plan:
        plan=json.loads(Path(opt.plan).read_text())
        if plan['source_sha256']!=sources() or plan['epochs']!=args.epochs or plan['partition']!=partition:
            raise ValueError('Frozen plan source/data/epochs mismatch')
        if plan['initial_state_sha256']!=[state_hash(m.state_dict()) for m in models]:
            raise ValueError('Frozen plan initialization mismatch')
        if plan['config']!=vars(config): raise ValueError('Frozen config mismatch')
        if plan['shared_kernel_shapes']!={k:list(v) for k,v in shared_specs(models).items()}:
            raise ValueError('Shared shape intersection mismatch')
        if any(sha256(Path(opt.dataset_root)/'cifar-100-python'/n)!=h for n,h in plan['dataset_sha256'].items()):
            raise ValueError('Raw CIFAR data differs')
    else:
        plan=prepare(args,config,opt,train,users,partition,models,profiles)
    if len(plan['schedule'])!=args.epochs or [x['round'] for x in plan['schedule']]!=list(range(args.epochs)):
        raise ValueError('Plan does not contain exactly the requested communication rounds')
    if plan.get('evaluation')!={'profiles':['P6'],'frequency':'every_round','batch_size':128,
                               'rng':'isolated_from_training'}:
        raise ValueError('Expected Full-only per-round evaluation protocol')
    output=Path(opt.output_dir); output.mkdir(parents=True,exist_ok=False)
    if opt.prepare:
        write_json(output/'protocol.json',plan); print(output/'protocol.json'); return
    env={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,
         'cudnn':torch.backends.cudnn.version(),'gpu':str(args.device),
         'gpu_name':torch.cuda.get_device_name(args.gpu) if args.gpu>=0 else 'cpu',
         'deterministic':torch.are_deterministic_algorithms_enabled(),
         'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}
    write_json(output/'manifest.json',{'mode':opt.mode,'formal_300':not bool(opt.smoke_rounds),
               'plan_sha256':content_hash(plan),'protocol':plan,'environment':env,
               'command':[sys.executable,'-m','experiments.run_hgkr7',*(sys.argv[1:] if argv is None else argv)]})
    history=KernelHistory(shared_specs(models),config); masks,_=coverage(models)
    best=-float('inf'); started=time.perf_counter()
    try:
        for r,entry in enumerate(plan['schedule']):
            wall=time.perf_counter(); row=run_round(models,train,users,entry,args,config,history,opt.mode,masks)
            begin=time.perf_counter()
            row['full_test']=evaluate_full(models,test,args)
            row['timings']['test_seconds']=time.perf_counter()-begin
            row['timings']['wall_seconds']=time.perf_counter()-wall
            with (output/'rounds.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
            full=row['full_test']['accuracy']
            if full>best:
                best=full; save_weights(output/'best_full.pt',models[-1],r,row['full_test'])
            print(f'{opt.mode} round={r+1}/{args.epochs} Full={full:.4f} test-oracle-peak={best:.4f} resets={row["reset_total"]}',flush=True)
        save_weights(output/'final_full.pt',models[-1],args.epochs-1,row['full_test'])
        write_json(output/'completed.json',{'mode':opt.mode,'rounds_completed':r+1,
                   'formal_300_complete':r+1==300 and not bool(opt.smoke_rounds),'elapsed_seconds':time.perf_counter()-started,
                   'ended_utc':datetime.now(timezone.utc).isoformat(),'final_source_sha256':sources()})
    except BaseException as error:
        write_json(output/'failed.json',{'error':repr(error),'mode':opt.mode})
        raise


if __name__=='__main__': main()
