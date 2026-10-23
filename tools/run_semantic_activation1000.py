"""Single fixed V2 run with real2-round preflight, no extra trained comparator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_semantic_pair500_suite import TESTS


def verify_smoke(new,reference):
    a=[json.loads(l) for l in new.read_text().splitlines()]
    b=[json.loads(l) for l in reference.read_text().splitlines()][:3]
    assert len(a)==3 and a==b
    return dict(success=True,rounds=2,models_slots_local_sgd_rng_official_test_exact_to_frozen_active500=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--reference500',type=Path,required=True)
    opt=p.parse_args();os.chdir(ROOT);root=opt.root.resolve();root.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    status=dict(pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        physical_gpu=0,formal_runs=1,epochs=1000,phase='preflight',success=False,processes=[],started=time.time())
    def save():
        temp=root/'status.pending';temp.write_text(json.dumps(status,indent=2));temp.replace(root/'status.json')
    def run(command,name):
        with (root/f'{name}.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            record=dict(name=name,pid=child.pid,argv=command);status['processes'].append(record);save()
            record['returncode']=child.wait();save()
            if record['returncode']:raise RuntimeError(f'{name} failed, no automatic retry')
    common=[sys.executable,'-u','tools/run_semantic_pair500.py','--mode','active',
        '--protocol','configs/semantic_a07_activation1000.json','--prepared',str(opt.prepared),
        '--dataset-root',str(opt.dataset_root)]
    save()
    try:
        info=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv'],text=True)
        (root/'gpu_start.csv').write_text(info)
        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<4096:raise RuntimeError('GPU0 has less than4096MiB free; do not change training settings')
        if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<2*1024**3:raise RuntimeError('Insufficient filesystem headroom')
        run([sys.executable,'-m','pytest','-q',*TESTS,'tests/test_semantic_pair500_report.py','tests/test_semantic_activation1000.py'],'pytest_linux')
        run(common+['--rounds','2','--output-root',str(root/'smoke')],'smoke2')
        proof=verify_smoke(root/'smoke/training/a07/active_2_seed1/regression.jsonl',
            opt.reference500/'active/training/a07/active_500_seed1/regression.jsonl')
        (root/'smoke_pass.json').write_text(json.dumps(proof,indent=2))
        status['phase']='formal1000';save()
        run(common+['--rounds','1000','--execute1000','--pilot-proof',str(root/'smoke_pass.json'),
            '--output-root',str(root/'active')],'active1000')
        status['phase']='activation_report';save()
        run([sys.executable,'tools/report_semantic_activation.py',str(root/'active')],'activation_report')
        status.update(phase='completed',success=True,finished=time.time());save()
    except BaseException:
        import traceback
        status.update(phase='failed',error=traceback.format_exc(),finished=time.time());save();raise


if __name__=='__main__':main()
