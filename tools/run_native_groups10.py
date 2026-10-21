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
            '--method-output-root',str(root),*(['--no-save-pt'] if not config.get('save_pt',True) else [])]) for name in batch] for batch in config['batches']]


def migration_resume_job(job,previous):
    """Only a user-requested, checkpoint-aligned GPU move can continue a job."""
    if previous['state']!='paused_for_gpu_migration':
        raise ValueError('Only explicitly paused GPU-migration jobs can continue; no failure retry')
    return dict(job,argv=[*job['argv'],'--method-resume',str(Path(job['output'])/'latest.pt')],
        previous_pid=previous['pid'],resume_round=previous['saved_round'])


def execute(python,dataset,root,config_path=None,physical_gpu=None,continue_suite=False):
    from tools.report_native_groups10 import validate_reference,validate_run
    config=json.loads((config_path or ROOT/'configs/native_groups10_400.json').read_text())
    if physical_gpu is None:physical_gpu=config.get('physical_gpu',0)
    validate_reference((ROOT/config['baseline_csv']).parent)
    if 'reference_csv' in config:validate_reference((ROOT/config['reference_csv']).parent)
    if continue_suite:
        state=json.loads((root/'status.json').read_text())
        if state['state']!='migration_paused' or state['failed']:
            raise ValueError('Continuation requires a deliberate GPU-migration pause, not a failed suite')
        state['previous_parent_pid']=state['parent_pid'];state['parent_pid']=os.getpid()
    else:
        root.mkdir(parents=True,exist_ok=False)
        state=dict(parent_pid=os.getpid(),state='starting',completed=[],failed=[],jobs={})
    state['physical_gpu']=physical_gpu
    def save():
        state['updated']=time.time();p=root/'status.pending';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
    save();env=environment(python);env['PYTHONHASHSEED']='1';env['CUDA_VISIBLE_DEVICES']=str(physical_gpu)
    for number,batch in enumerate(jobs(python,dataset,root,config_path),1):
        batch=[job for job in batch if job['id'] not in state['completed']]
        if not batch:continue
        gpu_record=f'gpu_before_batch{number}'+(f'_continued_gpu{physical_gpu}' if continue_suite else '')+'.txt'
        (root/gpu_record).write_text(subprocess.check_output(['nvidia-smi'],text=True))
        free=int(subprocess.check_output(['nvidia-smi','--id='+str(physical_gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<3500:
            state.update(state='resource_blocked',batch=number,free_mib=free);save();return 2
        concurrency=2 if free>=8000 else 1
        state.update(state='running',batch=number,concurrency=concurrency,free_mib=free);save()
        for start in range(0,len(batch),concurrency):
            active=[]
            for job in batch[start:start+concurrency]:
                old=state['jobs'].get(job['id'])
                if old:job=migration_resume_job(job,old)
                mode='a' if old else 'x'
                out=(root/(job['id']+'.stdout.log')).open(mode);err=(root/(job['id']+'.stderr.log')).open(mode)
                try:child=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=out,stderr=err)
                except Exception as exc:
                    out.close();err.close();state['jobs'][job['id']]=dict(job,state='launch_failed',error=str(exc))
                    state['failed'].append(job['id']);save();break
                state['jobs'][job['id']]=dict(job,pid=child.pid,state='running');save();active.append((job,child,out,err))
            for job,child,out,err in active:
                code=child.wait();out.close();err.close();error=None
                try:
                    if code:raise RuntimeError(f'Process exit {code}; original stderr/checkpoint retained')
                    validate_run(Path(job['output']))
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
    p.add_argument('--physical-gpu',type=int,choices=[0,1],default=None)
    p.add_argument('--continue-suite',action='store_true',help='Continue only a checkpoint-aligned user-requested GPU migration')
    a=p.parse_args();os.chdir(ROOT)
    if a.execute:raise SystemExit(execute(a.python,a.dataset_root,a.output_root,a.config,a.physical_gpu,a.continue_suite))
    for batch in jobs(a.python,a.dataset_root,a.output_root,a.config):
        for job in batch:print(shlex.join(job['argv']))


if __name__=='__main__':main()
