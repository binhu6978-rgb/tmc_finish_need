"""One fixed A07 observational trajectory and offline decision, never a Gate queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--equivalence-evidence', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    opt = p.parse_args()
    proof = json.loads(opt.equivalence_evidence.read_text())
    if not proof['exact_match'] or proof['rounds'] != 20:
        raise ValueError('Stage B has not passed')
    opt.output_root.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), status='running', phase='C', formal_gate_started=False,
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    def save():
        pending=opt.output_root/'status.pending'
        pending.write_text(json.dumps(state, indent=2)); pending.replace(opt.output_root/'status.json')
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    command = [sys.executable,'-u','tools/run_semcons_a07_observation.py','--mode','observe','--rounds','400',
        '--dataset-root',str(opt.dataset_root),'--prepared',str(opt.prepared),'--output-root',str(opt.output_root),
        '--equivalence-evidence',str(opt.equivalence_evidence)]
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
        (opt.output_root/'gpu_before.csv').write_text(gpu)
        with (opt.output_root/'observation_400.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid, argv=command); save()
            code = child.wait(); state['exit_code']=code; save()
        if code:
            raise RuntimeError('Observational trajectory failed; no retry or invented recovery')
        analysis=[sys.executable,'tools/analyze_semcons_a07.py',str(opt.output_root/'a07/observe_400_seed1'),
                  '--output',str(opt.output_root/'analysis')]
        subprocess.run(analysis, cwd=ROOT, env=env, check=True)
        result=json.loads((opt.output_root/'analysis/mechanism_summary.json').read_text())
        state.update(status='completed', scientific_decision=result['status'],
                     next='Stop semantic Gate experiments' if result['status']=='No-Go' else 'Review scientific Go evidence before implementing fixed formal comparison')
        save()
    except BaseException:
        import traceback
        state.update(status='failed', error=traceback.format_exc()); save(); raise


if __name__ == '__main__':
    main()
