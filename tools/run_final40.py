"""Two independent fixed queues; at most two original training jobs per GPU."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_local12 import environment


def read_config():
    return json.loads((ROOT / 'configs/capability_final40_400.json').read_text())


def spec(job_id):
    if job_id in sum(read_config()['gpu0_batches'], []):
        return 'A', job_id.lower()
    if job_id in sum(read_config()['gpu1_batches'], []):
        condition, method = job_id.split('-')
        return condition, 'baseline' if method == 'Native' else method.lower()
    raise ValueError('Not a fixed job')


def jobs(gpu, python, dataset, root):
    result = []
    for pair in read_config()[f'gpu{gpu}_batches']:
        batch = []
        for name in pair:
            condition, method = spec(name)
            output_root = root / f'gpu{gpu}'
            batch.append(dict(id=name, condition=condition, method=method, physical_gpu=gpu,
                output=str(output_root / method / (name + '_full_only_400_seed1')),
                argv=[str(python), '-u', 'tools/run_final40_job.py', '--job', name,
                    '--dataset-root', str(dataset), '--prepared', str(root / 'prepared'),
                    '--output-root', str(output_root)]))
        result.append(batch)
    return result


def save_status(path, state):
    state['updated'] = time.time()
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(state, indent=2))
    pending.replace(path)


def run_queue(gpu, python, dataset, root):
    from tools.report_native_groups10 import validate_run
    output = root / f'gpu{gpu}'
    output.mkdir(exist_ok=False)
    state = dict(parent_pid=os.getpid(), physical_gpu=gpu, state='starting', completed=[], failed=[], jobs={})
    def save():
        save_status(output / 'status.json', state)
    save()
    env = environment(python)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED='1')
    for number, batch in enumerate(jobs(gpu, python, dataset, root), 1):
        (output / f'gpu_before_batch{number}.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 3500:
            state.update(state='resource_blocked', batch=number, free_mib=free)
            save()
            return 2
        concurrency = 2 if free >= 8000 else 1
        state.update(state='running', batch=number, concurrency=concurrency, free_mib=free)
        save()
        for offset in range(0, 2, concurrency):
            active = []
            for job in batch[offset:offset + concurrency]:
                out = (output / (job['id'] + '.stdout.log')).open('x')
                err = (output / (job['id'] + '.stderr.log')).open('x')
                try:
                    child = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdout=out, stderr=err)
                except Exception as exc:
                    out.close(); err.close()
                    state['jobs'][job['id']] = dict(job, state='launch_failed', error=str(exc))
                    state['failed'].append(job['id']); save()
                    break
                state['jobs'][job['id']] = dict(job, state='running', pid=child.pid)
                active.append((job, child, out, err))
                save()
            for job, child, out, err in active:
                code = child.wait()
                out.close(); err.close()
                error = None
                try:
                    if code:
                        raise RuntimeError(f'Exit {code}; original stderr and written logs retained')
                    validate_run(Path(job['output']))
                except Exception as exc:
                    error = str(exc)
                state['jobs'][job['id']].update(state='failed' if error else 'completed', exit_code=code, error=error)
                state['failed' if error else 'completed'].append(job['id'])
                save()
            if state['failed']:
                state['state'] = 'failed'; save()
                return 1
    state['state'] = 'completed'; save()
    return 0


def execute(python, dataset, root):
    if not (root / 'prepared/manifest.json').is_file():
        raise ValueError('Prepare all conditions before training')
    if any((root / name).exists() for name in ('status.json', 'gpu0', 'gpu1', 'report')):
        raise FileExistsError('Never overwrite or duplicate an existing suite')
    state = dict(parent_pid=os.getpid(), state='starting', queues={}, requested_jobs=40,
                 epochs=400, seed=1, save_pt=False)
    save_status(root / 'status.json', state)
    children = []
    for gpu in (0, 1):
        out = (root / f'queue_gpu{gpu}.stdout.log').open('x')
        err = (root / f'queue_gpu{gpu}.stderr.log').open('x')
        argv = [str(python), '-u', 'tools/run_final40.py', '--queue', str(gpu), '--python', str(python),
                '--dataset-root', str(dataset), '--output-root', str(root)]
        child = subprocess.Popen(argv, cwd=ROOT, env=environment(python), stdout=out, stderr=err)
        state['queues'][str(gpu)] = dict(pid=child.pid, state='running', argv=argv)
        children.append((gpu, child, out, err))
        state['state'] = 'running'; save_status(root / 'status.json', state)
    # Independent queue processes progress even while the parent waits on one.
    for gpu, child, out, err in children:
        code = child.wait(); out.close(); err.close()
        state['queues'][str(gpu)].update(state='completed' if code == 0 else 'failed', exit_code=code)
        save_status(root / 'status.json', state)
    if any(q['exit_code'] for q in state['queues'].values()):
        state['state'] = 'failed'; save_status(root / 'status.json', state)
        return 1
    state['state'] = 'training_completed'; save_status(root / 'status.json', state)
    try:
        from tools.report_final40 import report
        report(root)
    except Exception:
        import traceback
        state.update(state='report_failed', error=traceback.format_exc())
        save_status(root / 'status.json', state)
        raise
    state['state'] = 'completed'; save_status(root / 'status.json', state)
    return 0


def main():
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--queue', type=int, choices=[0, 1], help=argparse.SUPPRESS)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('runs/capability_final40_400_seed1_20261021'))
    args = parser.parse_args(); os.chdir(ROOT)
    if args.prepare:
        args.output_root.mkdir(parents=True, exist_ok=False)
        from tools.final40_data import prepare
        prepare(args.dataset_root, args.output_root / 'prepared')
    elif args.queue is not None:
        raise SystemExit(run_queue(args.queue, args.python, args.dataset_root, args.output_root))
    elif args.execute:
        raise SystemExit(execute(args.python, args.dataset_root, args.output_root))
    else:
        for gpu in (0, 1):
            for batch in jobs(gpu, args.python, args.dataset_root, args.output_root):
                for job in batch:
                    print('CUDA_VISIBLE_DEVICES=' + str(gpu) + ' ' + shlex.join(job['argv']))


if __name__ == '__main__':
    main()
