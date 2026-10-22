"""One authorized paired run; no automatic retries, no algorithm selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TESTS=['tests/test_semantic_pair500.py','tests/test_semantic_iv_e_holdout.py',
       'tests/test_semantic_a07_iv_e.py','tests/test_semantic_a07_v2.py','tests/test_semcons_a07.py',
       'tests/test_semcons_a07_analysis.py','tests/test_semcons_a07_regression_evidence.py',
       'tests/test_semcons_a07_final_evidence.py','tests/test_final40.py','tests/test_adapterfl_methods.py',
       'tests/test_fd_semconsfl.py','tests/test_semcons_scope.py','tests/test_receiver_ce_probe.py']


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);opt=p.parse_args();os.chdir(ROOT)
    root=opt.root.resolve();root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    status=dict(pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        physical_gpu=0,started=time.time(),phase='preflight',processes=[],success=False)
    def save():
        temp=root/'status.pending';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'status.json')
    def run(command,name):
        with (root/f'{name}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            item=dict(name=name,pid=child.pid,argv=command);status['processes'].append(item);save()
            item['returncode']=child.wait();save()
            if item['returncode']:raise RuntimeError(f'{name} failed; logs retained, no retry')
    common=[sys.executable,'-u','tools/run_semantic_pair500.py','--prepared',str(opt.prepared),
        '--dataset-root',str(opt.dataset_root)]
    save()
    try:
        with (root/'gpu_storage.txt').open('x') as f:
            subprocess.run(['nvidia-smi'],stdout=f,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['df','-h',str(root)],stdout=f,stderr=subprocess.STDOUT,check=True)
        # About35GB quota headroom was verified separately; no large upload archives.
        if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<2*1024**3:raise RuntimeError('Less than2GB filesystem headroom')
        run([sys.executable,'-m','pytest','-q',*TESTS],'pytest_linux')
        for name in ('plain','diagnosed'):
            command=common+['--mode','neutral','--rounds','2','--output-root',str(root/'pilot'/name)]
            if name=='plain':command+=['--no-diagnostic']
            run(command,'pilot_'+name)
        run([sys.executable,'tools/run_semantic_pair500.py','--verify-pilot',str(root/'pilot')],'pilot_verification')
        status['phase']='formal500';save()
        children=[]
        for mode in ('neutral','active'):
            command=common+['--mode',mode,'--rounds','500','--output-root',str(root/mode),
                '--pilot-proof',str(root/'pilot/pilot_pass.json')]
            log=(root/f'{mode}_500.log').open('x');child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            item=dict(name=mode+'_500',pid=child.pid,argv=command);status['processes'].append(item)
            children.append((child,log,item));save()
        for child,log,item in children:
            item['returncode']=child.wait();log.close();save()
        if any(item['returncode'] for child,log,item in children):raise RuntimeError('A formal trajectory failed; no rerun')
        status['phase']='report';save()
        run([sys.executable,'tools/report_semantic_pair500.py',str(root)],'report')
        status.update(phase='completed',success=True,finished=time.time());save()
    except BaseException:
        import traceback
        status.update(phase='failed',error=traceback.format_exc(),finished=time.time());save();raise


if __name__=='__main__':main()
