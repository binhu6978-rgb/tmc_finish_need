"""Three serial original-entry controls, gated by real GPU reproducibility."""
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


def command(python,dataset,root,name,method,strict=True,start=1,audit=None,mode='rounds'):
    argv=[str(python),'-u']
    if audit is not None:argv+=['tools/check_e03_reproducibility.py','--mode',mode,'--audit-output',str(audit)]
    else:argv+=['tools/run_evidence_candidate.py']
    argv+=['--dataset-root',str(dataset),'--method',method,'--epochs','400','--seed','1','--gpu','0',
        '--generate_data','0','--data_beta','.3','--eval-profiles','full','--conflict-lambda','4',
        '--method-run-name',name,'--method-output-root',str(root)]
    if method=='m2':argv+=['--local-reference']
    if start!=1:argv+=['--classifier-protection-start-round',str(start)]
    if strict:argv+=['--deterministic-execution']
    return argv


def preflight(python,dataset,root,legacy=False):
    root.mkdir(parents=True,exist_ok=False)
    env=environment(python);env['PYTHONHASHSEED']='1'
    (root/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    from tools.check_e03_reproducibility import compare
    policies=[False,True] if legacy else [True]
    for strict in policies:
        prefix='strict' if strict else 'legacy'
        for name,method in [('REF','m2'),('L02','l02')]:
            run=prefix+'_'+name
            argv=command(python,dataset,root,run,method,strict,audit=root/run)
            with (root/(run+'.stdout.log')).open('x') as out,(root/(run+'.stderr.log')).open('x') as err:
                subprocess.run(argv,cwd=ROOT,env=env,stdout=out,stderr=err,check=True)
        target=root/(prefix+'_comparison.json')
        try:compare(root/(prefix+'_REF'),root/(prefix+'_L02'),target)
        except RuntimeError:
            if strict:raise
    gate=json.loads((root/'strict_comparison.json').read_text())
    import torch
    gate.update(host=__import__('socket').gethostname(),torch=torch.__version__,
        policy='strict deterministic execution, TF32 off, cuBLAS4096:8',timestamp=time.time())
    (root/'passed.json').write_text(json.dumps(gate,indent=2))


def execute(python,dataset,root,gate_path):
    gate=json.loads(gate_path.read_text())
    if not gate['passed']:raise ValueError('Require passed real GPU preflight')
    root.mkdir(parents=True,exist_ok=False)
    (root/'preflight_evidence.json').write_text(json.dumps(gate,indent=2))
    state=dict(parent_pid=os.getpid(),state='running',completed=[],failed=[],jobs={})
    def save():
        state['updated']=time.time();p=root/'status.pending';p.write_text(json.dumps(state,indent=2));p.replace(root/'status.json')
    save();env=environment(python);env['PYTHONHASHSEED']='1'
    (root/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    from tools.report_tonight10 import validate
    for name,method,start in [('R0','m2',1),('R1','l04',1),('R2','l04',2)]:
        run=name+'_full_only_400_seed1_deterministic'
        argv=command(python,dataset,root,run,method,start=start)
        with (root/(name+'.stdout.log')).open('x') as out,(root/(name+'.stderr.log')).open('x') as err:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=out,stderr=err)
            state['jobs'][name]=dict(pid=child.pid,argv=argv,state='running');save()
            code=child.wait()
        try:
            if code:raise RuntimeError(f'Process exit {code}')
            validate(root/method/run)
        except Exception as exc:
            state['jobs'][name].update(state='failed',error=str(exc));state['failed'].append(name)
            state['state']='failed';save();return 1
        state['jobs'][name]['state']='completed';state['completed'].append(name);save()
    state['state']='training_completed';save()
    from tools.report_l04_control import report
    try:report(root)
    except Exception as exc:
        state.update(state='report_failed',error=str(exc));save();raise
    state['state']='completed';save();return 0


def main():
    p=argparse.ArgumentParser(__doc__)
    group=p.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--execute',action='store_true')
    p.add_argument('--include-legacy-pair',action='store_true');p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gate',type=Path)
    a=p.parse_args();os.chdir(ROOT)
    if a.preflight:preflight(a.python,a.dataset_root,a.output_root,a.include_legacy_pair)
    elif a.execute:
        if a.gate is None:p.error('--execute requires --gate')
        raise SystemExit(execute(a.python,a.dataset_root,a.output_root,a.gate))
    else:
        for name,method,start in [('R0','m2',1),('R1','l04',1),('R2','l04',2)]:
            print(shlex.join(command(a.python,a.dataset_root,a.output_root,name+'_full_only_400_seed1_deterministic',method,start=start)))


if __name__=='__main__':main()
