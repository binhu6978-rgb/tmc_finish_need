"""Five fixed pairs calling the original AdaptiveFL entry; no live monitoring."""
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
from tools.run_local12 import environment


def jobs(python,dataset,root,config_path=None):
    config=json.loads((config_path or ROOT/'configs/native_groups10_400.json').read_text())
    return [[dict(id=name,method=name.lower(),output=str(root/name.lower()/(name+'_full_only_400_seed1_deterministic')),
        argv=[str(python),'-u','tools/run_evidence_candidate.py','--dataset-root',str(dataset),'--method',name.lower(),
            '--epochs','400','--seed','1','--gpu','0','--generate_data','0','--data_beta','.3','--eval-profiles','full',
            '--deterministic-execution','--method-run-name',name+'_full_only_400_seed1_deterministic',
            '--method-output-root',str(root)]) for name in batch] for batch in config['batches']]


def execute(python,dataset,root,config_path=None):
    from tools.report_tonight10 import validate
    config=json.loads((config_path or ROOT/'configs/native_groups10_400.json').read_text())
    validate((ROOT/config['baseline_csv']).parent)
    if 'reference_csv' in config:validate((ROOT/config['reference_csv']).parent)
    root.mkdir(parents=True,exist_ok=False)
    state=dict(parent_pid=os.getpid(),state='starting',completed=[],failed=[],jobs={})
    def save():
        state['updated']=time.time();p=root/'status.pending';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
    save();env=environment(python);env['PYTHONHASHSEED']='1'
    for number,batch in enumerate(jobs(python,dataset,root,config_path),1):
        (root/f'gpu_before_batch{number}.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<3500:
            state.update(state='resource_blocked',batch=number,free_mib=free);save();return 2
        concurrency=2 if free>=8000 else 1
        state.update(state='running',batch=number,concurrency=concurrency,free_mib=free);save()
        for start in range(0,len(batch),concurrency):
            active=[]
            for job in batch[start:start+concurrency]:
                out=(root/(job['id']+'.stdout.log')).open('x');err=(root/(job['id']+'.stderr.log')).open('x')
                try:child=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=out,stderr=err)
                except Exception as exc:
                    out.close();err.close();state['jobs'][job['id']]=dict(job,state='launch_failed',error=str(exc))
                    state['failed'].append(job['id']);save();break
                state['jobs'][job['id']]=dict(job,pid=child.pid,state='running');save();active.append((job,child,out,err))
            for job,child,out,err in active:
                code=child.wait();out.close();err.close();error=None
                try:
                    if code:raise RuntimeError(f'Process exit {code}; original stderr/checkpoint retained')
                    validate(Path(job['output']))
                except Exception as exc:error=str(exc)
                state['jobs'][job['id']].update(state='failed' if error else 'completed',exit_code=code,error=error)
                state['failed' if error else 'completed'].append(job['id']);save()
            if state['failed']:state['state']='failed';save();return 1
    state['state']='training_completed';save()
    from tools.report_native_groups10 import report
    try:report(root,config_path)
    except Exception as exc:state.update(state='report_failed',error=str(exc));save();raise
    state['state']='completed';save();return 0


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable));p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('runs/native_groups10_full400_seed1'))
    p.add_argument('--config',type=Path,default=None,help='Fixed suite JSON; original G01-G10 remains default')
    a=p.parse_args();os.chdir(ROOT)
    if a.execute:raise SystemExit(execute(a.python,a.dataset_root,a.output_root,a.config))
    for batch in jobs(a.python,a.dataset_root,a.output_root,a.config):
        for job in batch:print(shlex.join(job['argv']))


if __name__=='__main__':main()
