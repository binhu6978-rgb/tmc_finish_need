"""Fixed commands and two-process queue. Training stays in main_fed_ori.py."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/adapterfl_full_only_400.json'


def plan(output, gpus, python, config=CONFIG):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    entries = {e['id']: e for e in cfg['experiments']}
    batches = []
    for ids in cfg['batches']:
        batch = []
        for slot, name in enumerate(ids):
            entry = entries[name]
            run_name = name + '_full_only_400_seed1'
            argv = [python, '-u', str(ROOT/'main_fed_ori.py'), '--method', entry['method'],
                    '--method-run-name', run_name, '--method-output-root', str(output/'experiments'),
                    *cfg['common_args'], *entry['args']]
            batch.append({'id': name, 'method': entry['method'], 'gpu': gpus[slot], 'argv': argv,
                          'output_dir': str(output/'experiments'/entry['method']/run_name),
                          'stdout': str(output/(name+'.stdout.log'))})
        batches.append(batch)
    return batches


def gpu_inventory(gpus):
    # A startup/batch-boundary resource check, never a per-round monitor.
    inventory = subprocess.check_output(['nvidia-smi'], text=True)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    known = {int(row[0]): (row[1].strip(), int(row[2])) for row in csv.reader(gpu_rows.splitlines())}
    for gpu in set(gpus):
        if gpu not in known:
            raise RuntimeError(f'Physical GPU {gpu} is unavailable')
        uuid, free = known[gpu]
        if any(row and row[0].strip() == uuid for row in csv.reader(apps.splitlines())):
            raise RuntimeError(f'GPU {gpu} has another compute process; nothing was killed or started')
        if free < 8000:
            raise RuntimeError(f'GPU {gpu} has only {free} MiB free; refusing to change training parameters')
    return inventory


def wait_for_prior(path, state, save):
    if path is None:
        return
    while True:
        prior = json.loads(path.read_text())
        if prior['state'] == 'completed':
            if any(j.get('state') != 'completed' for j in prior['jobs'].values()) or len(prior['jobs']) != 7:
                raise RuntimeError('Prior seven-experiment queue has incomplete jobs')
            return
        if prior['state'] == 'failed':
            raise RuntimeError('Prior queue failed; no new experiment started')
        pid = prior['parent_pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise RuntimeError('Prior queue exited without completion; no new experiment started')
        if state['state'] != 'waiting_for_prior_suite':
            state.update(state='waiting_for_prior_suite', prior_status=str(path), prior_parent_pid=pid)
            save()
        time.sleep(60)


def validate(job):
    path = Path(job['output_dir'])
    done = json.loads((path/'completed.json').read_text())
    with (path/'accuracy_each_round.csv').open() as f:
        rows = list(csv.DictReader(f))
    if not (done.get('success') and done['rounds_completed'] == 400 and len(rows) == 400
            and [int(r['round']) for r in rows] == list(range(1, 401))
            and (path/'latest.pt').is_file() and (path/'best_full.pt').is_file()):
        raise RuntimeError(f"{job['id']} did not complete 400 rounds")


def run_batch(batch, state, save, validator=validate):
    children = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            try:
                for job in batch:
                    log = open(job['stdout'], 'x', encoding='utf-8')
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(job['gpu']), MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
                    try:
                        process = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close(); raise
                    children.append((job, process, log))
                    state['jobs'][job['id']] = dict(job, pid=process.pid, state='running')
                    save()
                    futures[executor.submit(process.wait)] = job
                for future in as_completed(futures):
                    job = futures[future]
                    code = future.result()
                    state['jobs'][job['id']].update(returncode=code, state='failed' if code else 'checking_completion')
                    save()
                    if code:
                        raise RuntimeError(f"{job['id']} exited {code}; original error is in {job['stdout']}")
                    validator(job)
                    state['jobs'][job['id']].update(state='completed', rounds_completed=400)
                    save()
            finally:
                # Only this invocation's children. Never kill pre-existing jobs.
                for job, process, log in children:
                    if process.poll() is None:
                        if os.name == 'posix':
                            os.killpg(process.pid, signal.SIGTERM)
                        else:
                            process.terminate()
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        state['jobs'][job['id']].update(state='stopped_after_batch_failure', returncode=process.returncode)
                    log.close()
                save()
    except BaseException:
        raise


def summarize(batches, output, config=CONFIG):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if cfg.get('report',{}).get('kind')=='semcons_scope':
        if __package__:
            from .report_semcons_scope10 import report_scope
        else:
            from report_semcons_scope10 import report_scope
        return report_scope(batches,output,cfg['report'])
    if cfg.get('report'):
        if __package__:
            from .report_next10_m2_m3 import report_next10
        else:
            from report_next10_m2_m3 import report_next10
        return report_next10(batches, output, cfg['report'])
    records = []
    for batch in batches:
        for job in batch:
            data = json.loads((Path(job['output_dir'])/'summary.json').read_text())
            records.append({'id': job['id'], 'method': job['method'], **{k: data[k] for k in
                ('rounds_completed', 'best_full_accuracy', 'best_round', 'final_full_accuracy', 'final_full_ce',
                 'last20_full_mean', 'last20_full_std', 'last50_full_mean', 'elapsed_seconds')}})
    with (output/'comparison.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    text = ['# Full-only 400-round exploratory results', '',
            'best_full_accuracy is an official-test **test-oracle peak**, not independent validation-selected performance.', '',
            '|ID|Method|test-oracle peak|Round|Final|Last20|', '|---|---|---:|---:|---:|---:|']
    text += [f"|{r['id']}|{r['method']}|{r['best_full_accuracy']:.4f}|{r['best_round']}|{r['final_full_accuracy']:.4f}|{r['last20_full_mean']:.4f}|" for r in records]
    text += ['', 'Compare each candidate with the new B00 using isolated evaluation RNG. Historical 300-round runs use legacy RNG consumption.',
             'These ten prespecified exploratory configurations use the official test set; no online tuning or claim of unbiased held-out selection.']
    (output/'report.md').write_text('\n'.join(text)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--gpus', default='0,1', help='Physical GPUs for the two slots, e.g. 0,0')
    parser.add_argument('--allow-shared-gpu', action='store_true', help='Explicit authorization after confirming single-GPU two-job resources')
    parser.add_argument('--after-suite', type=Path, help='Wait for the old seven-experiment suite status.json to complete')
    args = parser.parse_args(argv)
    gpus = [int(g) for g in args.gpus.split(',')]
    if len(gpus) != 2 or min(gpus) < 0:
        parser.error('--gpus requires two nonnegative physical GPU indices')
    if args.execute and len(set(gpus)) == 1 and not args.allow_shared_gpu:
        parser.error('Two jobs sharing a GPU need --allow-shared-gpu after resource validation')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    output = (args.output_root or ROOT/cfg.get('default_output_dir','runs/adapterfl_full_only_400_seed1')).resolve()
    batches = plan(output, gpus, sys.executable, args.config)
    for i, batch in enumerate(batches, 1):
        print(f'# Batch {i}', flush=True)
        for job in batch:
            print('CUDA_VISIBLE_DEVICES='+str(job['gpu'])+' '+shlex.join(job['argv'])+
                  ' > '+shlex.quote(job['stdout'])+' 2>&1', flush=True)
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=False)
    state = {'state': 'starting', 'parent_pid': os.getpid(), 'jobs': {}, 'plan': batches,
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'configuration_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(), 'config_path': str(args.config.resolve())}
    def save():
        temp = output/'status.pending'; temp.write_text(json.dumps(state, indent=2)); temp.replace(output/'status.json')
    def interrupt(signum, frame):
        raise RuntimeError(f'Queue interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    save()
    try:
        wait_for_prior(args.after_suite, state, save)
        for i, batch in enumerate(batches, 1):
            inventory = gpu_inventory(gpus)
            (output/f'batch{i}_nvidia_smi.txt').write_text(inventory)
            state.update(state='running', batch=i); save()
            run_batch(batch, state, save)
        summarize(batches, output, args.config)
        state.update(state='completed'); save()
    except BaseException as exc:
        state.update(state='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()
