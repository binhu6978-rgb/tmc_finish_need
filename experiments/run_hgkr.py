"""HGKR-FL B0/B1/B2/B3. Independent of legacy A0/A1/A2/A3 training."""
import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from experiments.baseline import BASELINE_FILES, content_hash, load_data, load_partition, sha256, state_hash, write_json_new
from experiments.run_semconsfl import atomic_checkpoint
from hgkr.core import (Config, KernelHistory, anchored_aggregate, choose_kernels, clone_state,
                       reset_copy, shared_specs)
from hgkr.protocol import initialize_models, make_schedule, model_costs
from models.Fed import split_model
from models.Update import LocalUpdate_AdaptiveFL
from models.test import test_img
from semconsfl.aggregation import coverage, update_diagnostics
from semconsfl.measurement import capture_rng, restore_rng
from utils.options import args_parser
from utils.set_seed import set_random_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--mode', required=True, choices=['B0','B1','B2','B3'])
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hgkr-config', default='configs/hgkr_three_tier.json')
    parser.add_argument('--partition-path', default='data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    extra, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    previous = sys.argv
    try:
        sys.argv = [previous[0]] + remaining
        args = args_parser()
    finally:
        sys.argv = previous
    args.model, args.dataset, args.algorithm = 'resnet','cifar100','HGKR-FL'
    args.width_ration, args.depth_saved, args.generate_data = [.4,.66,1.], [2], 0
    required = dict(num_users=100, frac=.1, local_ep=5, local_bs=50, bs=128, optimizer='sgd',
                    lr=.01, lr_decay=.998, momentum=.5, weight_decay=1e-4, data_beta=.3,
                    iid=0, num_classes=100, num_channels=3, client_hetero_ration='4:3:3')
    if any(getattr(args,k)!=v for k,v in required.items()) or args.epochs < 1:
        raise ValueError('This frozen protocol retains the original data/training/test settings')
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    args.device = torch.device(f'cuda:{args.gpu}' if args.gpu>=0 else 'cpu')
    return args,extra,Config(**json.loads(Path(extra.hgkr_config).read_text()))


def sources():
    paths = [Path(p) for p in BASELINE_FILES]
    paths += [Path('experiments')/p for p in ('baseline.py','run_semconsfl.py','run_hgkr.py','hgkr_suite.py','summarize_hgkr.py')]
    paths += sorted(Path('hgkr').glob('*.py'))
    paths += [Path('semconsfl')/p for p in ('aggregation.py','measurement.py')]
    paths += [Path('configs/hgkr_three_tier.json')]
    return {p.as_posix():sha256(p) for p in paths}


def sync(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def run_round(models, dataset, users, entry, args, config, history, mode, masks):
    r = entry['round']
    profiles = entry['B0'] if mode=='B0' else entry['priority']
    old = clone_state(models[-1].state_dict())
    history_start = time.perf_counter()
    scores = history.scores(r)  # snapshot is read-only for the entire round
    score_seconds = time.perf_counter()-history_start
    uploads, starts, counts, logs = [],[],[],[]
    train_seconds, reset_seconds = 0.,0.
    for slot,(c,p) in enumerate(zip(entry['clients'],profiles)):
        local = LocalUpdate_AdaptiveFL(args=args,dataset=dataset,idxs=users[c])
        model = copy.deepcopy(models[p]).to(args.device)
        dispatched = clone_state(model.state_dict())
        sync(args.device)
        begin = time.perf_counter()
        selection, diagnostic = choose_kernels(scores,config,mode,args.seed,r,slot)
        reset_l2 = reset_copy(dispatched,old,history.specs,selection,args.seed,r,slot)
        model.load_state_dict(dispatched)
        sync(args.device)
        reset_seconds += time.perf_counter()-begin
        start = clone_state(model.state_dict())
        begin = time.perf_counter()
        weights = local.train(round=r,net=model)
        sync(args.device)
        train_seconds += time.perf_counter()-begin
        upload = clone_state(weights)
        delta_l2 = sum(float((upload[k]-start[k]).double().square().sum()) for k in upload)**.5
        uploads.append(upload)
        starts.append(start)
        counts.append(len(users[c]))
        logs.append({'slot':slot,'client':c,'profile':p,'n':len(users[c]),
                     'sgd_steps':5*(len(users[c])//50),'reset_l2':reset_l2,
                     'local_learning_delta_l2':delta_l2,'reset_count':sum(map(len,selection.values())),
                     'kernels':diagnostic})
        del model,dispatched
    # This equality also detects any copy/view error corrupting the live server.
    if any(not torch.equal(v,models[-1].state_dict()[k]) for k,v in old.items()):
        raise ValueError('Dispatch/reset changed the server Full model')
    sync(args.device)
    begin=time.perf_counter()
    active=any(x['reset_l2']>0 for x in logs)
    new=anchored_aggregate(old,uploads,starts,counts,r,active)
    if any(not torch.isfinite(v).all() for v in new.values()):
        raise ValueError('Nonfinite global state')
    region=update_diagnostics(old,new,new,masks,3)
    history.update(r,old,new)
    for model in models:
        model.load_state_dict(split_model(new,model.state_dict()))
    sync(args.device)
    server_seconds=time.perf_counter()-begin
    # No forward/semantic measurement. This is a scalar comparison to the reset impulse.
    return {'round':r,'mode':mode,'slots':logs,'profile_slots':[profiles.count(p) for p in range(3)],
            'reset_total':sum(x['reset_count'] for x in logs),'allocation':entry,
            'coverage_update_diagnostics':region,
            'timings':{'local_train_seconds':train_seconds,'selection_reset_seconds':reset_seconds,
                       'history_score_seconds':score_seconds,'server_seconds':server_seconds}}


def main(argv=None):
    args,extra,config=parse_args(argv)
    train,test=load_data(extra.dataset_root)
    users,partition=load_partition(extra.partition_path,args,train.targets)
    set_random_seed(args.seed)
    models,profiles=initialize_models(args)
    initial=[state_hash(m.state_dict()) for m in models]
    costs=model_costs(models)
    schedule=make_schedule(args.seed,args.epochs,users,costs,config)
    masks,regions=coverage(models)
    specs=shared_specs(models)
    manifest={'protocol':'HGKR-three-tier-v1','mode':extra.mode,'config':asdict(config),
              'args':{k:str(v) if isinstance(v,torch.device) else v for k,v in vars(args).items()},
              'partition':partition,'initial_state_sha256':initial,'profiles':profiles,'costs':costs,
              'dataset_sha256':{name:sha256(Path(extra.dataset_root)/'cifar-100-python'/name) for name in ('train','test','meta')},
              'coverage':regions,'shared_kernel_shapes':specs,'schedule':schedule,
              'participation_sha256':content_hash([x['clients'] for x in schedule]),
              'priority_sha256':content_hash([x['priority'] for x in schedule]),
              'implementation_sha256':sources(),
              'evaluation':'Original official test all three profiles every round; peaks are test-oracle peak; no validation split',
              'environment':{'python':platform.python_version(),'torch':torch.__version__,
                             'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                             'gpu':torch.cuda.get_device_name(args.gpu) if args.gpu>=0 else 'cpu',
                             'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
                             'deterministic':torch.are_deterministic_algorithms_enabled(),
                             'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}}
    manifest['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    fingerprint=content_hash({k:v for k,v in manifest.items() if k!='git_head'})
    output=Path(extra.output_dir)
    history=KernelHistory(specs,config)
    start=0
    if extra.resume:
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        if saved['fingerprint']!=fingerprint or saved['mode']!=extra.mode:
            raise ValueError('Checkpoint configuration/source mismatch')
        start=saved['next_round']
        rows=[json.loads(x) for x in (output/'rounds.jsonl').read_text().splitlines()]
        if [x['round'] for x in rows]!=list(range(start)) or saved['history']['last_round']!=start-1:
            raise ValueError('Incomplete checkpoint/log round boundary')
        history.load_state_dict(saved['history'])
        for m in models:
            m.load_state_dict(split_model(saved['global'],m.state_dict()))
        restore_rng(saved['rng'])
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json_new(output/'manifest.json',dict(manifest,fingerprint=fingerprint))
        if extra.audit_only:
            return
        (output/'rounds.jsonl').touch(exist_ok=False)
    if extra.audit_only:
        raise ValueError('Audit cannot resume')
    for r in range(start,args.epochs):
        wall=time.perf_counter()
        row=run_round(models,train,users,schedule[r],args,config,history,extra.mode,masks)
        begin=time.perf_counter()
        row['official_test']=[dict(zip(('accuracy','ce'),test_img(m,test,args))) for m in models]
        row['timings']['test_seconds']=time.perf_counter()-begin
        row['timings']['round_wall_seconds']=time.perf_counter()-wall
        with (output/'rounds.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        atomic_checkpoint(output/'latest.pt',{'fingerprint':fingerprint,'mode':extra.mode,'next_round':r+1,
                          'global':clone_state(models[-1].state_dict()),'history':history.state_dict(),'rng':capture_rng()})
        print(f"round={r+1} mode={extra.mode} full_test={row['official_test'][-1]['accuracy']:.4f} resets={row['reset_total']}",flush=True)


if __name__=='__main__':
    main()
