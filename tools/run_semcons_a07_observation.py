"""Original A07 loop, optional read-only instrumentation, fixed real-data budgets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_final40_job import make_args
from tools import run_evidence_candidate as base
from tools.final40_data import datasets_for
from semcons_a07.config import load, DEFAULT
from semcons_a07.trainer import instrument_original_loop


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=('reference', 'observe'), required=True)
    parser.add_argument('--rounds', type=int, choices=(2, 20, 400), required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--equivalence-evidence', type=Path)
    opt = parser.parse_args()
    os.chdir(ROOT)
    cfg = load(opt.config)
    if opt.rounds == 400:
        if opt.mode != 'observe' or opt.equivalence_evidence is None:
            raise ValueError('400 rounds is observation only and requires a successful paired regression')
        proof = json.loads(opt.equivalence_evidence.read_text())
        if not proof['exact_match'] or proof['rounds'] < 20 or proof['config_sha256'] != cfg['config_sha256']:
            raise ValueError('Real 20-round equivalence has not passed for this configuration')
    args, condition, users = make_args('A07', opt.output_root, opt.prepared)
    args.epochs = opt.rounds
    args.method_run_name = f'{opt.mode}_{opt.rounds}_seed1'
    from utils.reproducibility import configure_deterministic_execution
    policy = configure_deterministic_execution(True)
    base.torch.set_num_threads(2); args.device = base.torch.device('cuda:0')
    path = base.prepare_output(args); started = time.perf_counter()
    files = [*sorted((ROOT/'semcons_a07').glob('*.py')), Path(__file__), opt.config]
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if opt.rounds == 400:
        if proof['source_sha256'] != source or proof['partition_sha256'] != args._method_partition_sha256:
            raise ValueError('Observer source/partition differs from the real regression proof')
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha for name, sha in proof['frozen_source_sha256'].items()):
            raise ValueError('Frozen A07 source differs from the real regression proof')
    try:
        base.set_random_seed(1)
        train, test = datasets_for(condition, opt.dataset_root)
        (path/'observation_manifest.json').write_text(json.dumps(dict(mode=opt.mode, rounds=opt.rounds,
            commit=commit, source_sha256=source, config=cfg, argv=sys.argv, pid=os.getpid(),
            partition_sha256=args._method_partition_sha256, deterministic_execution=policy,
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=base.torch.__version__,
            train_examples=len(train), test_examples=len(test), formal_gate=False,
            selection='fixed train-side diagnostic design; Full test is never a score input'), indent=2))
        unused = base.ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes)
        base.torch.cuda.reset_peak_memory_stats()
        with instrument_original_loop(cfg, train, users, path, opt.mode=='observe'):
            base.AdaptiveFL(args, train, test, users)
    except BaseException:
        import traceback
        (path/'failed.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2))
        raise
    finally:
        if base.torch.cuda.is_initialized():
            (path/'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
                peak_allocated_bytes=base.torch.cuda.max_memory_allocated(), peak_reserved_bytes=base.torch.cuda.max_memory_reserved()), indent=2))


if __name__ == '__main__':
    main()
