"""Short concurrent GPU capacity probe; no CIFAR performance result or resume."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from hgkr.core import Config, KernelHistory
from hgkr.seven import initialize_models, run_round, shared_specs
from models.Fed import split_model
from semconsfl.aggregation import coverage
from utils.set_seed import set_random_seed


def probe(gpu, output, peer_output=None):
    if gpu<0 or not torch.cuda.is_available(): raise ValueError('A CUDA GPU is required')
    set_random_seed(1)
    args=SimpleNamespace(width_ration=[.4,.66,1.],depth_saved=[2,3,4],model='resnet',num_channels=3,
                         num_classes=100,device=torch.device(f'cuda:{gpu}'),gpu=gpu,seed=1,
                         local_ep=5,local_bs=50,optimizer='sgd',lr=.01,lr_decay=.998,
                         momentum=.5,weight_decay=1e-4,dataset='cifar100')
    torch.cuda.set_device(gpu)
    # The allocator must exist before resetPeakMemoryStats (not just CUDA RNG).
    warmup=torch.empty(1,device=args.device)
    del warmup
    torch.cuda.reset_peak_memory_stats(gpu)
    models,_=initialize_models(args)
    for model in models: model.load_state_dict(split_model(models[-1].state_dict(),model.state_dict()))
    dataset=TensorDataset(torch.randn(50,3,32,32),torch.arange(50)%100)
    clients=list(range(70,80)); users={c:list(range(50)) for c in clients}
    history=KernelHistory(shared_specs(models),Config()); history.last_round=0
    masks,_=coverage(models)
    row=run_round(models,dataset,users,{'round':1,'clients':clients,'random':[6]*10,'priority':[6]*10},
                  args,Config(),history,'C1',masks)
    # Only P6 is evaluated on a test-sized batch; all seven remain in training.
    with torch.no_grad():
        models[6].eval()
        result=models[6](torch.zeros(128,3,32,32,device=args.device))['output']
        if not torch.isfinite(result).all(): raise ValueError('Nonfinite forward')
    result={'passed':True,'gpu':gpu,'synthetic_only':True,'full_slots':len(row['slots']),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(gpu),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(gpu),
            'GPU_name':torch.cuda.get_device_name(gpu)}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    # Keep this worker's CUDA allocator reservation alive until its peer has also
    # completed the stress path, so a fast worker cannot hide two-process pressure.
    if peer_output:
        deadline=time.monotonic()+180
        while not Path(peer_output).exists():
            if time.monotonic()>deadline: raise RuntimeError('Peer memory probe did not complete')
            time.sleep(.1)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); p.add_argument('--output',required=True)
    p.add_argument('--peer-output')
    opt=p.parse_args(); probe(opt.gpu,opt.output,opt.peer_output)
