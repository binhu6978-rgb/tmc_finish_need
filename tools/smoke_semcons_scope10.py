"""1-3 rounds of smoke only. Calls the original seven-model AdaptiveFL loop."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Algorithm.Training_AdaptiveFL import AdaptiveFL
from utils.options import args_parser
from utils.get_datatset_new import get_dataset,_build_record_path
from utils.set_seed import set_random_seed
from models.resnet import ResNet18_cifar


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode',choices=[f's{i:02d}' for i in range(10)],required=True)
    parser.add_argument('--rounds',type=int,choices=[1,2,3],default=2)
    parser.add_argument('--gpu',type=int,choices=[-1,0],default=-1)
    parser.add_argument('--full-data',action='store_true',help='Use the original full partition/test set; default is labelled smoke subsets')
    parser.add_argument('--output-root',default='runs/semcons_scope_preflight/smoke')
    opt=parser.parse_args();original_argv=sys.argv
    try:
        sys.argv=['main_fed_ori.py','--method',opt.mode,'--epochs',str(opt.rounds),'--seed','1',
                  '--gpu',str(opt.gpu),'--generate_data','0','--data_beta','.3','--eval-profiles','full',
                  '--conflict-lambda','4','--method-run-name','smoke_'+str(opt.rounds),
                  '--method-output-root',opt.output_root]
        args=args_parser()
    finally:
        sys.argv=original_argv
    torch.set_num_threads(2)
    args.model='resnet';args.dataset='cifar100';args.algorithm='AdaptiveFL'
    args.device=torch.device('cpu' if opt.gpu==-1 else 'cuda:0')
    set_random_seed(1)
    record=Path(_build_record_path(args))
    if not record.is_file():raise FileNotFoundError('Original saved split required')
    train,test,users=get_dataset(args)
    original_hash=hashlib.sha256(record.read_bytes()).hexdigest()
    if not opt.full_data:
        users={cid:list(indices)[:50] for cid,indices in users.items()}
        test=torch.utils.data.Subset(test,range(200))
    actual_hash=original_hash if opt.full_data else hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest()
    args._method_partition_sha256=actual_hash
    args.scope_smoke_provenance={'kind':'implementation smoke, not a performance experiment',
        'original_partition_sha256':original_hash,'full_data':opt.full_data,'examples_per_client':sorted({len(v) for v in users.values()}),
        'test_examples':len(test),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(args.scope_smoke_provenance),flush=True)
    # Match the original entry's otherwise unused initial Full construction.
    unused=ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
    AdaptiveFL(args,train,test,users)


if __name__=='__main__':main()
