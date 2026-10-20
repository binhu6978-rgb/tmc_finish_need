import csv
import json
from pathlib import Path
import subprocess
import sys

import torch

from tools.run_local12 import ROOT,environment


def test_complete_report_uses_local_ref_and_writes_real_plots(tmp_path):
    for number,name in enumerate(['REF']+[f'L{i:02d}' for i in range(1,12)]):
        method='m2' if name=='REF' else name.lower();p=tmp_path/method/(name+'_full_only_400_seed1');p.mkdir(parents=True)
        with (p/'accuracy_each_round.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['round','full_accuracy','full_ce']);w.writeheader()
            w.writerows({'round':i,'full_accuracy':100. if name=='L02' and i==100 else 30+number+(1 if i==100 else 0),'full_ce':2.} for i in range(1,401))
        server={'conflict_slot_count':1,'conflict_weights':[1.]*10,'difference_from_e03_l2':.1,
            'changed_from_e03_fraction':.01,'update_l2_ratio_e03':.99,'effective_step_along_e03':.98,
            'update_cosine_e03':.99,'actual_update_l2':.5}
        with (p/'method_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['round','metrics_json']);w.writeheader()
            w.writerows({'round':i,'metrics_json':json.dumps({'server':server,'slots':[(1,0)]*10})} for i in range(1,401))
        (p/'completed.json').write_text(json.dumps({'success':True,'rounds_completed':400}))
        (p/'manifest.json').write_text(json.dumps({'initial_state_sha256':['same']*7}))
        torch.save({'next_round':400,'rows':[{}]*400},p/'latest.pt')
        torch.save({'round':100,'full_accuracy':100. if name=='L02' else 31+number},p/'best_full.pt')
    result=subprocess.run([sys.executable,str(ROOT/'tools/report_local12.py'),'--root',str(tmp_path)],
        cwd=ROOT,env=environment(sys.executable),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((tmp_path/'report/summary.json').read_text())
    assert summary['comparison'][1]['test_oracle_peak_minus_local_REF_pp']==1.
    assert summary['primary_control'].startswith('local REF')
    text=(tmp_path/'report/report.md').read_text(encoding='utf-8')
    assert '新机制最高观察峰值：L11' in text
    assert (tmp_path/'report/accuracy.png').stat().st_size>1000
    assert (tmp_path/'report/ce.png').stat().st_size>1000
