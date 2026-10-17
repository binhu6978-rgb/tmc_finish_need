"""One-shot smoke gate, then two pairs of 100 rounds; at most two owned children."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.baseline import content_hash
from experiments.run_hgkr import sources


def write(path, data):
    temp=Path(str(path)+'.pending')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def validate_pairing(paths, rounds):
    manifests={p.name.split('_')[-1]:json.loads((p/'manifest.json').read_text()) for p in paths}
    rows={p.name.split('_')[-1]:[json.loads(x) for x in (p/'rounds.jsonl').read_text().splitlines()] for p in paths}
    reference=manifests['B1']
    for mode,m in manifests.items():
        for key in ('partition','dataset_sha256','initial_state_sha256','implementation_sha256','schedule','config','costs'):
            if m[key]!=reference[key]: raise ValueError(f'Pair mismatch {mode}: {key}')
        if [r['round'] for r in rows[mode]]!=list(range(rounds)):
            raise ValueError(f'Incomplete {mode}')
        if mode in ('B0','B1') and any(r['reset_total'] for r in rows[mode]):
            raise ValueError('Unexpected baseline reset')
        if any(r['reset_total'] for r in rows[mode][:4]):
            raise ValueError('Reset before mature old history')
    if [r['reset_total'] for r in rows['B2']] != [r['reset_total'] for r in rows['B3']]:
        raise ValueError('Random/history actual reset counts differ')
    for r in range(rounds):
        schedules=[[ (s['client'],s['profile']) for s in rows[m][r]['slots']] for m in ('B1','B2','B3')]
        if not schedules[0]==schedules[1]==schedules[2]: raise ValueError('Priority schedule diverged')
    return {'passed':True,'rounds':rounds,'arms':list(manifests),
            'same_partition':True,'same_initial_states':True,'same_source':True,
            'same_participation':True,'same_priority_allocation':True,'same_reset_count_B2_B3':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--launch100',action='store_true')
    opt=parser.parse_args()
    root=Path(opt.output_root).resolve(); root.mkdir(parents=True,exist_ok=False)
    frozen=sources()
    status={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'state':'running',
            'source_sha256':frozen,'commands':[],'completed':[],'gpu_all_arms':0}
    write(root/'status.json',status)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    try:
        stages=[('smoke',2)]+([('pair100',100)] if opt.launch100 else [])
        for stage,epochs in stages:
            paths=[root/f'{stage}_{mode}' for mode in ('B0','B1','B2','B3')]
            for modes in [('B0','B1'),('B2','B3')]:
                children=[]
                try:
                    for mode in modes:
                        if sources()!=frozen: raise ValueError('Frozen training source changed')
                        output=root/f'{stage}_{mode}'
                        cmd=[sys.executable,'-u','-m','experiments.run_hgkr','--mode',mode,
                             '--dataset-root',opt.dataset_root,'--output-dir',str(output),
                             '--epochs',str(epochs),'--seed','1','--gpu','0',
                             '--hgkr-config','configs/hgkr_three_tier.json']
                        out=(root/f'{stage}_{mode}.stdout.log').open('w',encoding='utf-8')
                        err=(root/f'{stage}_{mode}.stderr.log').open('w',encoding='utf-8')
                        p=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        children.append((mode,p,out,err))
                        status['commands'].append({'stage':stage,'mode':mode,'argv':cmd,'pid':p.pid,'output':str(output)})
                    status.update(stage=stage,active=list(modes)); write(root/'status.json',status)
                    while any(p.poll() is None for _,p,_,_ in children):
                        if any(p.poll() not in (None,0) for _,p,_,_ in children):
                            raise RuntimeError('A child failed; inspect its stderr log')
                        time.sleep(5)
                    for mode,p,_,_ in children:
                        if p.returncode!=0: raise RuntimeError(f'{stage}/{mode} exit {p.returncode}')
                        status['completed'].append(f'{stage}_{mode}')
                    write(root/'status.json',status)
                finally:
                    for _,p,out,err in children:
                        if p.poll() is None: p.terminate(); p.wait()
                        out.close(); err.close()
            write(root/f'{stage}_pairing_check.json',validate_pairing(paths,epochs))
        status.update(state='completed',active=[],ended_utc=datetime.now(timezone.utc).isoformat())
        write(root/'status.json',status)
        if opt.launch100:
            from experiments.summarize_hgkr import summarize
            summarize(root)
    except BaseException as error:
        status.update(state='failed',error=repr(error),ended_utc=datetime.now(timezone.utc).isoformat())
        write(root/'status.json',status)
        raise


if __name__=='__main__': main()
