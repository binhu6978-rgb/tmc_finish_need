"""Last two original-entry controls; reuse completed R0/R1/R2, no new engine."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_l04_control import command
from tools.run_local12 import environment


def jobs(python,dataset,root):
    result=[]
    for name,method in [('B00-D','baseline'),('R3','l04')]:
        run=name+'_full_only_400_seed1_deterministic'
        argv=command(python,dataset,root,run,method)
        if name=='R3':argv+=['--classifier-protection-end-round','1']
        result.append(dict(id=name,method=method,argv=argv,output=str(root/method/run)))
    return result


def execute(python,dataset,root,prior):
    from tools.report_tonight10 import validate
    # Completion check only; never launch or overwrite the existing three.
    for name,method in [('R0','m2'),('R1','l04'),('R2','l04')]:
        validate(prior/method/(name+'_full_only_400_seed1_deterministic'))
    root.mkdir(parents=True,exist_ok=False)
    state=dict(parent_pid=os.getpid(),state='starting',completed=[],failed=[],jobs={})
    def save():
        state['updated']=time.time();p=root/'status.pending';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
    save()
    (root/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    concurrency=2 if free>=6000 else 1
    state.update(concurrency=concurrency,free_mib_at_start=free);save()
    env=environment(python);env['PYTHONHASHSEED']='1'
    tasks=jobs(python,dataset,root)
    for start in range(0,2,concurrency):
        active=[]
        for job in tasks[start:start+concurrency]:
            out=(root/(job['id']+'.stdout.log')).open('x');err=(root/(job['id']+'.stderr.log')).open('x')
            child=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=out,stderr=err)
            state['jobs'][job['id']]=dict(job,pid=child.pid,state='running');state['state']='running';save()
            active.append((job,child,out,err))
        for job,child,out,err in active:
            code=child.wait();out.close();err.close();error=None
            try:
                if code:raise RuntimeError(f'Process exit {code}; stderr and checkpoint preserved')
                validate(Path(job['output']))
            except Exception as exc:error=str(exc)
            state['jobs'][job['id']].update(state='failed' if error else 'completed',exit_code=code,error=error)
            state['failed' if error else 'completed'].append(job['id']);save()
        if state['failed']:
            state['state']='failed';save();return 1
    state['state']='training_completed';save()
    from tools.report_l04_control import report
    try:report(prior,extra_root=root)
    except Exception as exc:
        state.update(state='report_failed',error=str(exc));save();raise
    state['state']='completed';save();return 0


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('runs/e03_l04_final_pair400_seed1'))
    p.add_argument('--prior-root',type=Path,default=Path('runs/e03_l04_control400_seed1'))
    a=p.parse_args();os.chdir(ROOT)
    if a.execute:raise SystemExit(execute(a.python,a.dataset_root,a.output_root,a.prior_root))
    for job in jobs(a.python,a.dataset_root,a.output_root):print(shlex.join(job['argv']))


if __name__=='__main__':main()
