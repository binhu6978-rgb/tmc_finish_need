"""Only 2/20-round paired real A07 regression; never starts a Gate experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    opt = p.parse_args()
    opt.output_root.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), status='running', phase='B', batches=[], formal_gate=False)
    def save():
        pending = opt.output_root/'status.pending'
        pending.write_text(json.dumps(state, indent=2)); pending.replace(opt.output_root/'status.json')
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'], text=True)
        (opt.output_root/'gpu_before.csv').write_text(gpu)
        for rounds in (2, 20):
            batch = dict(rounds=rounds, jobs=[]); state['batches'].append(batch); save()
            running = []
            for mode in ('reference', 'observe'):
                command = [sys.executable, '-u', 'tools/run_semcons_a07_observation.py', '--mode', mode,
                    '--rounds', str(rounds), '--dataset-root', str(opt.dataset_root), '--prepared', str(opt.prepared),
                    '--output-root', str(opt.output_root)]
                log = (opt.output_root/f'{mode}_{rounds}.log').open('x')
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job = dict(mode=mode, pid=child.pid, argv=command, status='running')
                batch['jobs'].append(job); running.append((child, log, job)); save()
            # Both independent tasks finish; no retries, deletion, or parameter changes.
            for child, log, job in running:
                job['exit_code'] = child.wait(); log.close()
                job['status'] = 'completed' if child.returncode == 0 else 'failed'; save()
            if any(j['exit_code'] for j in batch['jobs']):
                raise RuntimeError(f'Real {rounds}-round regression process failed; inspect preserved logs')
            command = [sys.executable, 'tools/check_semcons_a07_equivalence.py',
                str(opt.output_root/'a07'/f'reference_{rounds}_seed1'), str(opt.output_root/'a07'/f'observe_{rounds}_seed1'),
                '--output', str(opt.output_root/f'equivalence_{rounds}.json')]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        state['status'] = 'completed'; state['equivalence_passed'] = True; save()
    except BaseException:
        import traceback
        state.update(status='failed', error=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    main()
