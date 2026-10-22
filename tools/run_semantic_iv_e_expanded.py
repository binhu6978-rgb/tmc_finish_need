"""Fixed 2-round checks or 120-round heldout diagnostic, never formal400."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_final40_job import make_args
from tools.final40_data import datasets_for,context_for
from tools import run_evidence_candidate as base
from models.Update import DatasetSplit
import adapterfl_methods as methods
from semcons_a07.trainer import OriginalHooks
from semcons_a07_v2.trainer import semantic_loop
from semcons_a07_v2.iv_e_holdout import split_users,HeldoutDiagnostic


@contextmanager
def observed(cfg,dataset,users,evaluation,path,boundaries,enabled,rounds):
    with semantic_loop(cfg,dataset,users,path,'active',boundaries) as observer:
        old_log=methods.RunLog;diag=HeldoutDiagnostic(dataset,users,evaluation,path);diag.observer=observer
        class Log(old_log):
            def __init__(self,args,models):
                super().__init__(args,models);diag.models=models;normal=self.hooks.aggregate
                def aggregate(uploads,lens,previous,dispatched=None):
                    shadow=copy.copy(self.hooks);first=shadow.rounds_seen==0
                    actual=normal(uploads,lens,previous,dispatched)
                    if observer.round_number in rounds:
                        a07=OriginalHooks.aggregate(shadow,uploads,lens,previous,dispatched)
                        diag.run(a07,actual,previous,uploads,lens,self.hooks,observer.decision,first)
                    return actual
                if enabled:self.hooks.aggregate=aggregate
        try:
            methods.RunLog=Log;yield observer
        finally:methods.RunLog=old_log


def pilot_check(root):
    paths=[root/f'{mode}_2/training/a07/{mode}_2_holdout_seed1' for mode in ('reference','diagnostic')]
    rows=[[json.loads(l) for l in (p/'regression.jsonl').read_text().splitlines()] for p in paths]
    assert len(rows[0])==3 and rows[0]==rows[1]
    manifests=[json.loads((p.parent.parent.parent/'manifest.json').read_text()) for p in paths]
    assert manifests[0]['training_sha256']==manifests[1]['training_sha256']
    result=dict(success=True,rounds=2,exact_models_sgd_rng_validation=True,training_sha256=manifests[0]['training_sha256'])
    (root/'pilot_pass.json').write_text(json.dumps(result,indent=2));print(result)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--verify-pilot',type=Path)
    p.add_argument('--mode',choices=('reference','diagnostic'));p.add_argument('--rounds',type=int,choices=(2,120))
    p.add_argument('--output-root',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--dataset-root',type=Path)
    p.add_argument('--pilot-proof',type=Path);opt=p.parse_args()
    if opt.verify_pilot:pilot_check(opt.verify_pilot);return
    if not all((opt.mode,opt.rounds,opt.output_root,opt.prepared,opt.dataset_root)):p.error('All training arguments required')
    if opt.rounds==120 and (not opt.pilot_proof or not json.loads(opt.pilot_proof.read_text())['success']):raise ValueError('Real exact pilot must pass before120')
    root=opt.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
    args,condition,original=make_args('A07',root/'training',opt.prepared)
    protocol=json.loads((ROOT/'configs/semantic_a07_iv_e_expanded.json').read_text())
    cfg=json.loads((ROOT/'configs/semcons_semantic_a07_v2.json').read_text())
    stages=protocol['stages'];rounds=set(range(1,3)) if opt.rounds==2 else {r for a,b in stages.values() for r in range(a,b+1)}
    cfg['diagnostic_rounds']={'expanded':sorted(rounds)} if opt.mode=='diagnostic' else {}
    from utils.reproducibility import configure_deterministic_execution
    policy=configure_deterministic_execution(True);base.torch.set_num_threads(2);args.device=base.torch.device('cuda:0')
    args.epochs=opt.rounds;args.method_run_name=f'{opt.mode}_{opt.rounds}_holdout_seed1'
    base.set_random_seed(1);train,unused_official_test=datasets_for(condition,opt.dataset_root)
    users,evaluation=split_users(train,original)
    train_sha=hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest()
    if opt.rounds==120:assert json.loads(opt.pilot_proof.read_text())['training_sha256']==train_sha
    args._method_partition_sha256=train_sha;args._final40_context=context_for('A',condition,users)
    pooled=sorted(i for indices in evaluation.values() for i in indices);validation=DatasetSplit(train,pooled)
    (root/'split_map.json').write_text(json.dumps(dict(training=users,evaluation=evaluation)))
    (root/'manifest.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),argv=sys.argv,
        protocol=protocol,measurement_configuration=cfg,training_sha256=train_sha,
        original_partition_sha256='41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41',
        training_examples=sum(map(len,users.values())),evaluation_examples=len(pooled),
        per_client_counts={c:dict(training=len(users[c]),evaluation=len(evaluation[c])) for c in users},
        deterministic=policy,full_evaluation='pooled heldout validation, not official test',
        original_full_training_trajectory=False,no_pt=True),indent=2))
    path=base.prepare_output(args);begin=time.perf_counter();base.torch.cuda.reset_peak_memory_stats()
    try:
        unused=base.ResNet18_cifar(num_channels=args.num_channels,num_classes=args.num_classes)
        with observed(cfg,train,users,evaluation,path,args._final40_context['boundaries'],opt.mode=='diagnostic',rounds):
            base.AdaptiveFL(args,train,validation,users)
        assert not list(root.rglob('*.pt'))
        (root/'completed.json').write_text(json.dumps(dict(success=True,rounds=opt.rounds,diagnostic_rounds=sorted(rounds) if opt.mode=='diagnostic' else [],pt_files=0),indent=2))
    except BaseException:
        import traceback
        (root/'failed.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));raise
    finally:
        (root/'resources.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-begin,
            peak_allocated=base.torch.cuda.max_memory_allocated(),peak_reserved=base.torch.cuda.max_memory_reserved()),indent=2))


if __name__=='__main__':main()
