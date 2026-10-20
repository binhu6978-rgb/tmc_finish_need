"""Thin local/server data-root adapter calling the ORIGINAL AdaptiveFL loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torchvision import datasets,transforms
from Algorithm.Training_AdaptiveFL import AdaptiveFL
from models.resnet import ResNet18_cifar
from utils.options import args_parser
from utils.set_seed import set_random_seed
import utils.get_datatset_new as data
from adapterfl_methods import prepare_output


def main():
    extra=argparse.ArgumentParser(add_help=False);extra.add_argument('--dataset-root',type=Path,required=True)
    extra.add_argument('--local-reference',action='store_true',help='Native M2 with read-only local-suite norm logging')
    opt,remaining=extra.parse_known_args()
    argv=sys.argv
    try:
        sys.argv=['main_fed_ori.py',*remaining];args=args_parser()
    finally:sys.argv=argv
    if args.method not in ('baseline','m2','c1','c2',*[f't{i:02d}' for i in range(1,11)],*[f'l{i:02d}' for i in range(1,12)],*[f'g{i:02d}' for i in range(1,11)],*[f'p{i:02d}' for i in range(1,11)]) or args.seed!=1 or args.epochs!=400 or args.generate_data!=0 or args.eval_profiles!='full':
        raise ValueError('Fixed seed1 full-only400 protocol required')
    args.model='resnet';args.dataset='cifar100';args.algorithm='AdaptiveFL'
    args._local12_reference=opt.local_reference
    if opt.local_reference and args.method!='m2':raise ValueError('Local REF must use original M2')
    from utils.reproducibility import configure_deterministic_execution
    policy=configure_deterministic_execution(args.deterministic_execution)
    args.device=torch.device('cuda:0');torch.set_num_threads(2)
    path=prepare_output(args);started=time.perf_counter()
    try:
        set_random_seed(1)
        record=Path(data._build_record_path(args))
        if not record.is_file():raise FileNotFoundError('Original cached partition required')
        args._method_partition_sha256=hashlib.sha256(record.read_bytes()).hexdigest()
        if args._method_partition_sha256!='41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41':
            raise ValueError('Original B00/E03 partition differs')
        def loaded():
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                (.5071,.4867,.4408),(.2675,.2565,.2761))])
            return (datasets.CIFAR100(str(opt.dataset_root),train=True,download=False,transform=transform),
                    datasets.CIFAR100(str(opt.dataset_root),train=False,download=False,transform=transform))
        original=data.prepare_cifar100
        try:
            data.prepare_cifar100=loaded
            train,test,users=data.get_dataset(args)
        finally:data.prepare_cifar100=original
        if len(train)!=50000 or len(test)!=10000 or len(users)!=100:raise ValueError('Original full data required')
        (path/'run_environment.json').write_text(json.dumps({'pid':os.getpid(),'argv':argv,'dataset_root':str(opt.dataset_root),
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'train_examples':len(train),'test_examples':len(test),
            'deterministic_execution':policy},indent=2))
        # Match the original entry's unused Full construction BEFORE get_model_list.
        unused=ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
        torch.cuda.reset_peak_memory_stats()
        AdaptiveFL(args,train,test,users)
    except BaseException:
        import traceback
        (path/'failed.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        raise
    finally:
        if torch.cuda.is_initialized():
            (path/'resources.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-started,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2))


if __name__=='__main__':main()
