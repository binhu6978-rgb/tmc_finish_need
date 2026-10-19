"""Six local batches of two original-entry processes; no training engine."""
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


def environment(python):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',KMP_DUPLICATE_LIB_OK='TRUE')
    home=Path(python).resolve().parent
    if os.name=='nt':env['PATH']=os.pathsep.join(str(home/p) for p in ('Library/bin','','Scripts'))+os.pathsep+env.get('PATH','')
    return env


def jobs(config,python,dataset,output):
    return [[{'id':name,'method':'m2' if name=='REF' else name.lower(),
        'output':str(output/('m2' if name=='REF' else name.lower())/(name+'_full_only_400_seed1')),
        'argv':[str(python),'-u','tools/run_evidence_candidate.py','--dataset-root',str(dataset),
            *(['--local-reference'] if name=='REF' else []),'--method','m2' if name=='REF' else name.lower(),
            '--epochs','400','--seed','1','--gpu','0','--generate_data','0','--data_beta','.3',
            '--eval-profiles','full','--conflict-lambda','4','--method-run-name',name+'_full_only_400_seed1',
            '--method-output-root',str(output)]} for name in batch] for batch in config['batches']]


def run(batches,output,env):
    from tools.report_tonight10 import validate
    output.mkdir(parents=True,exist_ok=False)
    state={'parent_pid':os.getpid(),'state':'starting','completed':[],'failed':[],'batches':batches,'jobs':{}}
    def save():
        state['updated']=time.time();p=output/'status.pending';p.write_text(json.dumps(state,indent=2));p.replace(output/'status.json')
    save()
    for number,batch in enumerate(batches,1):
        # A boundary-only resource check, not a live monitoring service.
        gpu=subprocess.check_output(['nvidia-smi'],text=True)
        (output/f'gpu_before_batch{number}.txt').write_text(gpu)
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        concurrency=2 if free>=6000 else 1
        state.update(state='running',batch=number,concurrency=concurrency,free_mib_at_batch_start=free);save()
        for start in range(0,len(batch),concurrency):
            active=[]
            for job in batch[start:start+concurrency]:
                out=(output/(job['id']+'.stdout.log')).open('x',encoding='utf-8')
                err=(output/(job['id']+'.stderr.log')).open('x',encoding='utf-8')
                try:
                    p=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=out,stderr=err,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                except Exception as error:
                    out.close();err.close();state['failed'].append(job['id'])
                    state['jobs'][job['id']]=dict(job,state='launch_failed',error=str(error));save();break
                state['jobs'][job['id']]=dict(job,pid=p.pid,state='running');save()
                active.append((job,p,out,err))
            for job,p,out,err in active:
                code=p.wait();out.close();err.close();error=None
                if code==0:
                    try:validate(Path(job['output']))
                    except Exception as exc:error=str(exc)
                else:error=f'Process exit{code}; original stderr and checkpoint retained'
                state['jobs'][job['id']].update(state='failed' if error else 'completed',exit_code=code,error=error)
                state['failed' if error else 'completed'].append(job['id']);save()
            if state['failed']:
                state['state']='failed';save();return 1
    state['state']='training_completed';save()
    from tools.report_local12 import report
    try:report(output)
    except Exception as error:
        state.update(state='report_failed',report_error=str(error));save();raise
    state['state']='completed';save();return 0


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=Path('runs/e03_local12_full400_seed1'))
    a=p.parse_args();os.chdir(ROOT)
    config=json.loads((ROOT/'configs/e03_local12_400.json').read_text())
    batches=jobs(config,a.python,a.dataset_root,a.output_root)
    if not a.execute or a.dry_run:
        for batch in batches:
            for job in batch:print(subprocess.list2cmdline(job['argv']) if os.name=='nt' else shlex.join(job['argv']))
        return
    env=environment(a.python);os.environ.update(env)
    raise SystemExit(run(batches,a.output_root,env))


if __name__=='__main__':main()
