"""Exactly six short real runs, 2/20 reference-neutral-active. No 400 dispatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    opt=p.parse_args();opt.output_root.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),status='running',jobs=[],scope='only real2/20 correctness; no400')
    def save():(opt.output_root/'status.json').write_text(json.dumps(state,indent=2))
    save();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
        (opt.output_root/'gpu_before.csv').write_text(gpu)
        for rounds in (2,20):
            for modes in (('reference','neutral'),('active',)):
                children=[]
                for mode in modes:
                    command=[sys.executable,'-u','tools/run_semantic_a07_v2.py','--mode',mode,'--rounds',str(rounds),
                        '--dataset-root',str(opt.dataset_root),'--prepared',str(opt.prepared),'--output-root',str(opt.output_root)]
                    log=(opt.output_root/f'{mode}_{rounds}.log').open('x')
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    record=dict(mode=mode,rounds=rounds,pid=child.pid,argv=command,status='running')
                    state['jobs'].append(record);children.append((child,log,record));save()
                for child,log,record in children:
                    code=child.wait();log.close();record.update(exit_code=code,status='completed' if code==0 else 'failed');save()
                if any(r['exit_code'] for _,_,r in children):raise RuntimeError('Short run failed; no retry or parameter change')
            command=[sys.executable,'tools/check_semantic_a07_v2.py',str(opt.output_root),'--rounds',str(rounds),'--output',str(opt.output_root/f'check_{rounds}.json')]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
        state['status']='completed';save()
    except BaseException:
        import traceback
        state.update(status='failed',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
