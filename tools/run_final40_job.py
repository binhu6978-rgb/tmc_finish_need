"""One fixed-budget job: original AdaptiveFL loop with a prepared condition."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_evidence_candidate as base
from tools.final40_data import config, datasets_for, validate_partition, context_for


def parse_job(job_id):
    if job_id in sum(config()['gpu0_batches'], []):
        return 'A', job_id.lower()
    if job_id in sum(config()['gpu1_batches'], []):
        condition, method = job_id.split('-')
        return condition, 'baseline' if method == 'Native' else method.lower()
    raise ValueError('Not one of the exact forty fixed jobs')


def make_args(job_id, output_root, prepared):
    name, method = parse_job(job_id)
    condition = config()['conditions'][name]
    saved = sys.argv
    try:
        sys.argv = ['main_fed_ori.py', '--method', method, '--epochs', '400', '--seed', '1',
            '--gpu', '0', '--generate_data', '0', '--data_beta', str(condition['alpha']),
            '--dataset', condition['dataset'], '--num_classes', str(condition['classes']),
            '--client_hetero_ration', condition['ratio'], '--eval-profiles', 'full',
            '--deterministic-execution', '--no-save-pt', '--method-run-name', job_id + '_full_only_400_seed1',
            '--method-output-root', str(output_root)]
        args = base.args_parser()
    finally:
        sys.argv = saved
    args.model = 'resnet'
    args.algorithm = 'AdaptiveFL'
    record = prepared / (name + '.json')
    users = validate_partition(json.loads(record.read_text()), condition)
    prepared_manifest = json.loads((prepared / 'manifest.json').read_text())[name]
    sha = hashlib.sha256(record.read_bytes()).hexdigest()
    if sha != prepared_manifest['sha256']:
        raise ValueError('Prepared training partition changed')
    args._method_partition_sha256 = sha
    args._final40_context = context_for(name, condition, users)
    if args._final40_context != prepared_manifest['context']:
        raise ValueError('Prepared capability/data mass context changed')
    return args, condition, users


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    opt = parser.parse_args()
    os.chdir(ROOT)
    args, condition, users = make_args(opt.job, opt.output_root, opt.prepared)
    from utils.reproducibility import configure_deterministic_execution
    policy = configure_deterministic_execution(True)
    base.torch.set_num_threads(2)
    args.device = base.torch.device('cuda:0')
    path = base.prepare_output(args)
    started = time.perf_counter()
    try:
        base.set_random_seed(1)
        train, test = datasets_for(condition, opt.dataset_root)
        (path / 'run_environment.json').write_text(json.dumps(dict(pid=os.getpid(), argv=sys.argv,
            job=opt.job, condition=condition, dataset_root=str(opt.dataset_root),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=base.torch.cuda.get_device_name(),
            torch=base.torch.__version__, train_examples=len(train), test_examples=len(test),
            deterministic_execution=policy, prepared_partition=str(opt.prepared), save_pt=False), indent=2))
        # Preserve original entry's unused Full initialization before seven models.
        unused = base.ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)
        base.torch.cuda.reset_peak_memory_stats()
        base.AdaptiveFL(args, train, test, users)
    except BaseException:
        import traceback
        (path / 'failed.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2))
        raise
    finally:
        if base.torch.cuda.is_initialized():
            (path / 'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter() - started,
                peak_allocated_bytes=base.torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=base.torch.cuda.max_memory_reserved()), indent=2))


if __name__ == '__main__':
    main()
