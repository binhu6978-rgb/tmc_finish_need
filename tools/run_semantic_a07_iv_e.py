"""Only 2/20 real V2 rounds with independent read-only recipient probes."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import adapterfl_methods as methods
from semcons_a07.trainer import OriginalHooks
from semcons_a07_v2.trainer import semantic_loop
from semcons_a07_v2.iv_e_diagnostics import MechanismDiagnostic
from tools import run_semantic_a07_v2 as original_entry


@contextmanager
def diagnosed_loop(cfg,dataset,users,path,mode,boundaries):
    with semantic_loop(cfg,dataset,users,path,mode,boundaries) as observer:
        diagnostic=MechanismDiagnostic(dataset,users,path);diagnostic.observer=observer
        old_log=methods.RunLog
        class Log(old_log):
            def __init__(self,args,models):
                super().__init__(args,models);diagnostic.models=models
                normal=self.hooks.aggregate
                def aggregate(uploads,lens,previous,dispatched=None):
                    shadow=copy.copy(self.hooks);first=shadow.rounds_seen==0
                    actual=normal(uploads,lens,previous,dispatched)
                    # Read-only same-input original A07; shadow owns its round counter.
                    base=OriginalHooks.aggregate(shadow,uploads,lens,previous,dispatched)
                    diagnostic.run(base,actual,previous,uploads,lens,self.hooks,observer.decision,first)
                    return actual
                self.hooks.aggregate=aggregate
        try:
            methods.RunLog=Log;yield observer
        finally:methods.RunLog=old_log


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--rounds',type=int,choices=(2,20),required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    opt=p.parse_args();root=opt.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/semcons_semantic_a07_v2.json').read_text())
    cfg['diagnostic_rounds']={'iv_e':list(range(1,opt.rounds+1))}
    cfg_path=root/'observation_config.json';cfg_path.write_text(json.dumps(cfg,indent=2))
    (root/'protocol.json').write_text((ROOT/'configs/semantic_a07_v2_iv_e.json').read_text())
    (root/'diagnostic_manifest.json').write_text(json.dumps(dict(
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),argv=sys.argv,
        budget=opt.rounds,reference=str(opt.reference),no_pt=True,no_official_test_data_in_diagnostic=True),indent=2))
    saved=sys.argv;old=original_entry.semantic_loop
    try:
        original_entry.semantic_loop=diagnosed_loop
        sys.argv=['tools/run_semantic_a07_v2.py','--mode','active','--rounds',str(opt.rounds),
            '--config',str(cfg_path),'--output-root',str(root/'training'),
            '--dataset-root',str(opt.dataset_root),'--prepared',str(opt.prepared)]
        original_entry.main()
        actual=root/'training/a07'/f'active_{opt.rounds}_seed1'
        a=[json.loads(l) for l in (actual/'regression.jsonl').read_text().splitlines()]
        b=[json.loads(l) for l in (opt.reference/'regression.jsonl').read_text().splitlines()]
        assert len(a)==len(b)==opt.rounds+1 and a==b,'Diagnostic changed V2 trajectory'
        assert not list(root.rglob('*.pt'))
        (root/'diagnostic_completed.json').write_text(json.dumps(dict(success=True,rounds=opt.rounds,
            exact_old_v2_states_uploads_rng_evaluations=True,pt_files=0),indent=2))
    finally:sys.argv=saved;original_entry.semantic_loop=old


if __name__=='__main__':main()
